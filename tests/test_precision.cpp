#include <Cellerator/ops/numeric/binary16.hh>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

import velocity.precision;

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

template<class F> void expect_invalid(F&& f, const char* message) {
    try { f(); expect(false, message); }
    catch (const std::invalid_argument&) {}
    catch (...) { expect(false, message); }
}

template<class F> void expect_runtime(F&& f, const char* message) {
    try { f(); expect(false, message); }
    catch (const std::runtime_error&) {}
    catch (...) { expect(false, message); }
}

template<class T> bool close(T actual, T expected, double tolerance = 2e-6) {
    const double a = static_cast<double>(actual), b = static_cast<double>(expected);
    return std::abs(a - b) <= tolerance * (1.0 + std::abs(b));
}

template<class T> struct AnalyticFit {
    T gamma = 0, r2 = 0;
    velocity::FitStatus status = velocity::FitStatus::insufficient_data;
    bool selected = false;
};

template<class T, std::size_t N>
T normalized_score(const std::array<T, N>& x, const std::array<T, N>& y,
                   T max_x, T max_y, std::size_t i) {
    const T safe_x = max_x > T(1e-3) ? max_x : T(1e-3);
    const T safe_y = max_y > T(1e-3) ? max_y : T(1e-3);
    return x[i] / safe_x + y[i] / safe_y;
}

template<class T, std::size_t N>
T analytic_percentile(std::vector<T> values, T q) {
    std::sort(values.begin(), values.end());
    const T position = T(values.size() - 1) * (q / T(100));
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = std::min(lower + 1, values.size() - 1);
    const T fraction = position - T(lower);
    return values[lower] + fraction * (values[upper] - values[lower]);
}

template<class T, std::size_t N>
AnalyticFit<T> analytic_upper_fit(const std::array<T, N>& x, const std::array<T, N>& y,
                                  const std::uint8_t* mask, T cutoff,
                                  const velocity::FitOptions& options, bool enabled) {
    T max_x = -std::numeric_limits<T>::infinity();
    T max_y = -std::numeric_limits<T>::infinity();
    for (std::size_t i = 0; i < N; ++i) if (!mask || mask[i]) {
        max_x = std::max(max_x, x[i]);
        max_y = std::max(max_y, y[i]);
    }

    std::size_t count = 0;
    T xx = 0, xy = 0;
    for (std::size_t i = 0; i < N; ++i) if (!mask || mask[i]) {
        if (normalized_score(x, y, max_x, max_y, i) < cutoff) continue;
        ++count;
        xx += x[i] * x[i];
        xy += x[i] * y[i];
    }
    if (!count) return {};
    if (!(xx > 0)) return {0, 0, velocity::FitStatus::singular, false};

    const T gamma = xy / xx;
    T mean_y = 0, low_y = y[0], high_y = y[0];
    for (T value : y) {
        mean_y += value;
        low_y = std::min(low_y, value);
        high_y = std::max(high_y, value);
    }
    mean_y /= T(N);
    if (low_y == high_y) return {gamma, 0, velocity::FitStatus::constant_response, false};

    T residual = 0, total = 0;
    for (std::size_t i = 0; i < N; ++i) {
        const T e = y[i] - gamma * x[i];
        const T d = y[i] - mean_y;
        residual += e * e;
        total += d * d;
    }
    if (!(total > 0)) return {gamma, 0, velocity::FitStatus::constant_response, false};
    const T r2 = T(1) - residual / total;
    const bool selected = enabled && gamma > T(options.min_ratio) && r2 > T(options.min_r2);
    return {gamma, r2, velocity::FitStatus::ok, selected};
}

template<class T, std::size_t N>
struct AnalyticScreen {
    T low = 0, high = 0, delta = 0, prediction_change = 0, r2_change = 0;
    AnalyticFit<T> expanded, contracted;
    bool status_change = false, selection_change = false;
};

template<class T, std::size_t N>
AnalyticScreen<T, N> analytic_screen(const std::array<T, N>& x, const std::array<T, N>& y,
    const std::uint8_t* mask, const velocity::GeneFit& baseline,
    const velocity::FitOptions& fit, velocity::precision::SensitivityOptions options,
    bool enabled) {
    std::vector<T> scores;
    T max_x = -std::numeric_limits<T>::infinity();
    T max_y = -std::numeric_limits<T>::infinity();
    for (std::size_t i = 0; i < N; ++i) if (!mask || mask[i]) {
        max_x = std::max(max_x, x[i]);
        max_y = std::max(max_y, y[i]);
    }
    for (std::size_t i = 0; i < N; ++i) if (!mask || mask[i])
        scores.push_back(normalized_score(x, y, max_x, max_y, i));

    AnalyticScreen<T, N> out;
    out.high = analytic_percentile<T, N>(scores, T(fit.percentile_high));
    const T magnitude = std::max(T(1), std::abs(out.high));
    out.delta = T(options.band) * T(1.0 / 2048.0) * magnitude;
    out.expanded = analytic_upper_fit(x, y, mask, out.high - out.delta, fit, enabled);
    out.contracted = analytic_upper_fit(x, y, mask, out.high + out.delta, fit, enabled);
    out.status_change = out.expanded.status != baseline.status || out.contracted.status != baseline.status;
    out.selection_change = out.expanded.selected != baseline.selected ||
                           out.contracted.selected != baseline.selected;

    T residual2 = 0, response2 = 0, prediction2 = 0;
    T expanded_change2 = 0, contracted_change2 = 0;
    for (std::size_t i = 0; i < N; ++i) {
        const T prediction = T(baseline.gamma) * x[i] + T(baseline.offset);
        const T residual = y[i] - prediction;
        residual2 += residual * residual;
        response2 += y[i] * y[i];
        prediction2 += prediction * prediction;
        if (out.expanded.status == velocity::FitStatus::ok) {
            const T d = out.expanded.gamma * x[i] - prediction;
            expanded_change2 += d * d;
        }
        if (out.contracted.status == velocity::FitStatus::ok) {
            const T d = out.contracted.gamma * x[i] - prediction;
            contracted_change2 += d * d;
        }
    }
    const T epsilon = std::numeric_limits<T>::epsilon();
    const T floor = T(32) * epsilon * std::max(std::sqrt(response2), std::sqrt(prediction2));
    const T denominator = std::max(std::sqrt(residual2), floor);
    out.prediction_change = std::sqrt(std::max(expanded_change2, contracted_change2)) / denominator;
    if (out.expanded.status == velocity::FitStatus::ok)
        out.r2_change = std::max(out.r2_change, std::abs(out.expanded.r2 - T(baseline.r2)));
    if (out.contracted.status == velocity::FitStatus::ok)
        out.r2_change = std::max(out.r2_change, std::abs(out.contracted.r2 - T(baseline.r2)));
    return out;
}

template<class T>
void test_analytic_screen() {
    constexpr std::size_t n = 5;
    const std::array<T, n> x{T(1), T(2), T(3), T(4), T(10)};
    const std::array<T, n> y{T(1), T(3), T(4.5), T(6), T(10)};
    velocity::FitOptions fit;
    fit.trimming = velocity::Trimming::upper;
    fit.percentile_high = 50;
    fit.fit_intercept = false;
    fit.min_r2 = -10;
    fit.min_ratio = 0;
    velocity::precision::SensitivityOptions options{128, 100, 100};
    const velocity::GeneFit baseline{1.3, .2, .61, velocity::FitStatus::ok, true};
    T scratch[n]{};

    const auto expected = analytic_screen(x, y, nullptr, baseline, fit, options, true);
    const auto report = velocity::precision::sensitivity_gene(x.data(), y.data(), n, 1,
        [&] {
            if constexpr (std::is_same_v<T, float>)
                return velocity::GeneFit32{T(baseline.gamma), T(baseline.offset), T(baseline.r2), baseline.status, baseline.selected};
            else return baseline;
        }(), fit, scratch, nullptr, true, options);
    const double tolerance = std::is_same_v<T, float> ? 2e-5 : 2e-12;
    expect(close(report.low, T(0), tolerance) && close(report.high, T(.75), tolerance) &&
           close(report.delta, T(1.0 / 16.0), tolerance),
           "screen reports the independently calculated score cutoff and perturbation");
    expect(close(expected.expanded.gamma, T(1.1), tolerance) &&
           close(expected.contracted.gamma, T(31.0 / 29.0), tolerance) &&
           close(report.prediction_change, expected.prediction_change, tolerance) &&
           close(report.r2_change, expected.r2_change, tolerance),
           "screen reports changes implied by independent upper-trim fit calculations and supplied baseline");
    expect(report.expanded_status == expected.expanded.status &&
           report.contracted_status == expected.contracted.status &&
           report.status_change == expected.status_change &&
           report.selection_change == expected.selection_change &&
           report.state == velocity::precision::SensitivityState::stable,
           "ordinary float/double screen remains advisory and stable when changes are below limits");

    const std::uint8_t mask[n]{1, 1, 1, 0, 1};
    const auto masked_expected = analytic_screen(x, y, mask, baseline, fit, options, true);
    const auto masked = velocity::precision::sensitivity_gene(x.data(), y.data(), n, 1,
        [&] {
            if constexpr (std::is_same_v<T, float>)
                return velocity::GeneFit32{T(baseline.gamma), T(baseline.offset), T(baseline.r2), baseline.status, baseline.selected};
            else return baseline;
        }(), fit, scratch, mask, true, options);
    expect(close(masked.high, analytic_percentile<T, n>(std::vector<T>{T(.2), T(.5), T(.75), T(2)}, T(50)), tolerance) &&
           !close(masked.high, report.high, tolerance) &&
           close(masked.prediction_change, masked_expected.prediction_change, tolerance) &&
           close(masked.r2_change, masked_expected.r2_change, tolerance),
           "fit mask changes cutoff and fitting rows while prediction diagnostics still cover every row");

    const auto disabled = velocity::precision::sensitivity_gene(x.data(), y.data(), n, 1,
        [&] {
            if constexpr (std::is_same_v<T, float>)
                return velocity::GeneFit32{T(baseline.gamma), T(baseline.offset), T(baseline.r2), baseline.status, true};
            else return baseline;
        }(), fit, scratch, mask, false, options);
    expect(disabled.selection_change,
           "disabled gene selection is reported as a selection change from the caller baseline");

    const auto at_cutoff = analytic_upper_fit(x, y, nullptr, T(.75), fit, true);
    velocity::GeneFit base0;
    base0.gamma = double(at_cutoff.gamma);
    base0.r2 = double(at_cutoff.r2);
    base0.status = at_cutoff.status;
    base0.selected = at_cutoff.selected;
    velocity::GeneFit32 base032{static_cast<float>(at_cutoff.gamma), 0,
        static_cast<float>(at_cutoff.r2), at_cutoff.status, at_cutoff.selected};
    velocity::precision::SensitivityOptions zero_band{0, 1e-4, 1e-4};
    const auto zero = [&] {
        if constexpr (std::is_same_v<T, float>)
            return velocity::precision::sensitivity_gene(x.data(), y.data(), n, 1, base032, fit,
                scratch, nullptr, true, zero_band);
        else
            return velocity::precision::sensitivity_gene(x.data(), y.data(), n, 1, base0, fit,
                scratch, nullptr, true, zero_band);
    }();
    expect(zero.state == velocity::precision::SensitivityState::stable && zero.delta == T(0) &&
           zero.expanded_status == velocity::FitStatus::ok &&
           zero.contracted_status == velocity::FitStatus::ok &&
           close(zero.prediction_change, T(0), tolerance) && close(zero.r2_change, T(0), tolerance),
           "zero-width screen gives the same fit as the supplied baseline with no reported change");
}

template<class T>
void test_constant_cutoff_and_invalid_inputs() {
    constexpr std::size_t n = 4;
    const std::array<T, n> x{T(1), T(2), T(3), T(4)};
    const std::array<T, n> y{T(4), T(3), T(2), T(1)};
    velocity::FitOptions fit;
    fit.trimming = velocity::Trimming::upper;
    fit.percentile_high = 50;
    fit.fit_intercept = false;
    fit.min_r2 = -4;
    fit.min_ratio = 0;
    T scratch[n]{};
    const double gamma = 2.0 / 3.0, r2 = -7.0 / 3.0;
    velocity::precision::SensitivityOptions zero_band{0, 1e-5, 1e-5};
    auto base = [&] {
        if constexpr (std::is_same_v<T, float>)
            return velocity::GeneFit32{T(gamma), T(0), T(r2), velocity::FitStatus::ok, true};
        else return velocity::GeneFit{gamma, 0, r2, velocity::FitStatus::ok, true};
    }();
    const auto exact = velocity::precision::sensitivity_gene(x.data(), y.data(), n, 1,
        base, fit, scratch, nullptr, true, zero_band);
    expect(close(exact.high, T(1.25), 2e-6) && exact.delta == T(0) &&
           exact.state == velocity::precision::SensitivityState::stable &&
           close(exact.prediction_change, T(0), 2e-6) && close(exact.r2_change, T(0), 2e-6),
           "exactly constant normalized scores retain the caller's matching baseline at zero band");

    velocity::precision::SensitivityOptions stress{4096, 100, 100};
    const auto stressed = velocity::precision::sensitivity_gene(x.data(), y.data(), n, 1,
        base, fit, scratch, nullptr, true, stress);
    expect(stressed.status_change && stressed.selection_change &&
           stressed.contracted_status == velocity::FitStatus::insufficient_data &&
           stressed.state == velocity::precision::SensitivityState::sensitive,
           "large sensitivity band crossing an exact-score tie changes eligibility and fit status");

    velocity::FitOptions no_trim = fit;
    no_trim.trimming = velocity::Trimming::none;
    const auto untrimmed = velocity::precision::sensitivity_gene(
        static_cast<const T*>(nullptr), static_cast<const T*>(nullptr), 0, 1,
        base, no_trim, static_cast<T*>(nullptr));
    expect(untrimmed.state == velocity::precision::SensitivityState::no_trimming &&
           untrimmed.expanded_status == base.status && untrimmed.contracted_status == base.status,
           "untrimmed fits need no score scratch and are reported as no-trimming");

    auto invalid_base = base;
    invalid_base.status = velocity::FitStatus::singular;
    const auto degenerate = velocity::precision::sensitivity_gene(
        x.data(), y.data(), n, 1, invalid_base, fit, scratch);
    expect(degenerate.state == velocity::precision::SensitivityState::unscoreable &&
           degenerate.expanded_status == velocity::FitStatus::singular &&
           degenerate.contracted_status == velocity::FitStatus::singular &&
           !degenerate.status_change && !degenerate.selection_change,
           "a degenerate supplied fit is unscoreable without fabricating a status change");

    velocity::precision::SensitivityOptions bad;
    bad.band = -1;
    expect_invalid([&] { (void)velocity::precision::sensitivity_gene(x.data(), y.data(), n, 1, base, fit, scratch, nullptr, true, bad); },
                   "negative sensitivity band is rejected");
    bad = {};
    bad.prediction_limit = std::numeric_limits<double>::quiet_NaN();
    expect_invalid([&] { (void)velocity::precision::sensitivity_gene(x.data(), y.data(), n, 1, base, fit, scratch, nullptr, true, bad); },
                   "nonfinite sensitivity limit is rejected");
    bad = {};
    expect_invalid([&] { (void)velocity::precision::sensitivity_gene(x.data(), y.data(), n, 0, base, fit, scratch); },
                   "zero input stride is rejected");
    expect_invalid([&] { (void)velocity::precision::sensitivity_gene(x.data(), y.data(), n, std::numeric_limits<std::ptrdiff_t>::max(), base, fit, scratch); },
                   "row-offset overflow from an oversized stride is rejected before dereference");
    auto bad_fit = fit;
    bad_fit.percentile_low = 80;
    bad_fit.percentile_high = 20;
    expect_invalid([&] { (void)velocity::precision::sensitivity_gene(x.data(), y.data(), n, 1, base, bad_fit, scratch); },
                   "invalid fitting percentiles are rejected by the sensitivity boundary");
}

template<class T, class ArrayLike> std::vector<T> values(const ArrayLike& array) {
    auto v = array.view().template dense<T>();
    std::vector<T> out(v.extent(0));
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = v.data()[std::ptrdiff_t(i) * v.stride(0)];
    return out;
}

std::vector<bool> boolean_values(const anndatapp::Array& array) {
    auto v = array.view().dense<bool>();
    std::vector<bool> out(v.extent(0));
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = v.data()[std::ptrdiff_t(i) * v.stride(0)];
    return out;
}

std::vector<std::uint8_t> byte_values(const anndatapp::Array& array) {
    auto v = array.view().dense<std::uint8_t>();
    std::vector<std::uint8_t> out(v.extent(0));
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = v.data()[std::ptrdiff_t(i) * v.stride(0)];
    return out;
}

struct PromotionData {
    static constexpr std::size_t n = 8, g = 3;
    static std::array<float, n * g> make_spliced() {
        std::array<float, n * g> out{};
        for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < g; ++j)
            out[i * g + j] = j == 2 ? float(i + 1) : 1.0f + .7f * float(i) + .3f * float(j) + .02f * float(i * j);
        return out;
    }
    static std::array<float, n * g> make_unspliced() {
        std::array<float, n * g> out{};
        constexpr float noise[n]{.13f, -.20f, .11f, -.07f, .17f, -.15f, .06f, -.10f};
        constexpr float tiny[n]{-.0004f, .0004f, -.0004f, .0004f, -.0004f, .0004f, -.0004f, .0004f};
        constexpr float slopes[g]{.6f, 1.2f, 0};
        for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < g; ++j) {
            const float x = j == 2 ? float(i + 1) : 1.0f + .7f * float(i) + .3f * float(j) + .02f * float(i * j);
            out[i * g + j] = j == 2 ? 2.0001f * x + tiny[i]
                                    : slopes[j] * x + .5f + .2f * float(j) + noise[i] * float(j + 1);
        }
        return out;
    }
    static std::array<std::uint16_t, n * g> as_half(const std::array<float, n * g>& x) {
        std::array<std::uint16_t, n * g> out{};
        for (std::size_t i = 0; i < x.size(); ++i) out[i] = cellerator::numeric::round_f16(x[i]);
        return out;
    }
    static std::array<float, n * n> identity() {
        std::array<float, n * n> out{};
        for (std::size_t i = 0; i < n; ++i) out[i * n + i] = 1;
        return out;
    }

    std::array<float, n * g> full_s = make_spliced();
    std::array<float, n * g> full_u = make_unspliced();
    std::array<std::uint16_t, n * g> half_s = as_half(full_s);
    std::array<std::uint16_t, n * g> half_u = as_half(full_u);
    std::array<float, n * n> weights = identity();
    anndatapp::Array full_s_array = anndatapp::Array::borrow_dense(full_s.data(), {n, g}, {std::ptrdiff_t(g), 1});
    anndatapp::Array full_u_array = anndatapp::Array::borrow_dense(full_u.data(), {n, g}, {std::ptrdiff_t(g), 1});
    velocity::NumericView half_s_view = velocity::NumericView::borrow_dense_f16(
        half_s.data(), {n, g}, {std::ptrdiff_t(g), 1});
    velocity::NumericView half_u_view = velocity::NumericView::borrow_dense_f16(
        half_u.data(), {n, g}, {std::ptrdiff_t(g), 1});
    anndatapp::Array graph_array = anndatapp::Array::borrow_dense(weights.data(), {n, n}, {std::ptrdiff_t(n), 1});
    velocity::PreparedInput full_input{{full_s_array.view(), full_u_array.view(), graph_array.view(), anndatapp::Axis::gather({2, 0, 1})}};
    velocity::PreparedInput half_input{{half_s_view, half_u_view, graph_array.view(), anndatapp::Axis::gather({2, 0, 1})}};

    velocity::FitOptions fit_options() const {
        velocity::FitOptions o;
        o.trimming = velocity::Trimming::none;
        o.fit_intercept = false;
        o.min_r2 = 0;
        o.min_ratio = 0;
        return o;
    }
};

struct SourceContext {
    PromotionData* data = nullptr;
    std::array<std::size_t, PromotionData::g> calls{};
    bool unavailable_gene0 = true;
    bool fail_gene2_after_first = false;

    static bool acquire(void* context, std::size_t source_gene, velocity::precision::OriginalColumn& column) {
        auto& self = *static_cast<SourceContext*>(context);
        if (source_gene >= PromotionData::g) return false;
        const auto call = ++self.calls[source_gene];
        if (source_gene == 0 && self.unavailable_gene0) return false;
        if (source_gene != 2 || (self.fail_gene2_after_first && call > 1)) return false;
        auto s = anndatapp::Array::borrow_dense(self.data->full_s.data() + source_gene,
            {PromotionData::n, 1}, {std::ptrdiff_t(PromotionData::g), 1});
        auto u = anndatapp::Array::borrow_dense(self.data->full_u.data() + source_gene,
            {PromotionData::n, 1}, {std::ptrdiff_t(PromotionData::g), 1});
        column = {s.view(), u.view()};
        return true;
    }

    velocity::precision::OriginalSource source(std::size_t staging = 0) {
        return {this, acquire, staging};
    }
};

struct HalfSourceContext {
    std::array<std::uint16_t, PromotionData::n> spliced{};
    std::array<std::uint16_t, PromotionData::n> unspliced{};
    std::size_t calls = 0;

    static bool acquire(void* context, std::size_t gene, velocity::precision::OriginalColumn& column) {
        auto& self = *static_cast<HalfSourceContext*>(context);
        if (gene != 2) return false;
        ++self.calls;
        auto s = velocity::NumericView::borrow_dense_f16(self.spliced.data(),
            {PromotionData::n, 1}, {1, 1});
        auto u = velocity::NumericView::borrow_dense_f16(self.unspliced.data(),
            {PromotionData::n, 1}, {1, 1});
        column = {s, u};
        return true;
    }
};

velocity::RunOptions run32() {
    velocity::RunOptions o;
    o.workspace_bytes = 1u << 20;
    o.block_genes = 1;
    o.output_dtype = velocity::StorageType::float32;
    o.compute_precision = velocity::ComputePrecision::float32;
    return o;
}

velocity::StochasticRunOptions stochastic_run32() {
    velocity::StochasticRunOptions o;
    o.workspace_bytes = 1u << 20;
    o.block_genes = 1;
    o.output_dtype = velocity::StorageType::float32;
    o.compute_precision = velocity::ComputePrecision::float32;
    return o;
}

template<class T>
void compare_fit_at(const velocity::FitResult& actual, const velocity::FitResult& expected,
                   std::size_t gene, double tolerance, bool& ok) {
    const auto ga = values<T>(actual.gamma), ge = values<T>(expected.gamma);
    const auto oa = values<T>(actual.offset), oe = values<T>(expected.offset);
    const auto ra = values<T>(actual.r2), re = values<T>(expected.r2);
    const auto sa = byte_values(actual.status), se = byte_values(expected.status);
    const auto xa = boolean_values(actual.genes), xe = boolean_values(expected.genes);
    ok &= close(ga[gene], ge[gene], tolerance) && close(oa[gene], oe[gene], tolerance) &&
          close(ra[gene], re[gene], tolerance) && sa[gene] == se[gene] && xa[gene] == xe[gene];
}

void test_sensitivity_workspace_budget_and_binding() {
    PromotionData data;
    auto options = run32();
    velocity::Workspace workspace(data.half_input, options);
    auto fit = data.fit_options();
    velocity::ReferenceStages32 stages(data.half_input, workspace, fit);
    constexpr std::size_t staging = 64;
    const auto score = PromotionData::n * sizeof(float);
    const auto budget = workspace.scratch_bytes() + score + staging;
    velocity::precision::SensitivityWorkspace32 scratch(data.half_input, workspace, stages, budget, staging);
    expect(scratch.data() != nullptr && scratch.scratch_bytes() == score,
           "sensitivity workspace allocates only its row score buffer under the exact aggregate budget");
    expect_invalid([&] {
        velocity::precision::SensitivityWorkspace32 too_small(data.half_input, workspace, stages, budget - 1, staging);
    }, "sensitivity workspace rejects a budget one byte below stage plus score plus staging");

    velocity::PreparedInput other_input{{data.half_s_view, data.half_u_view,
        data.graph_array.view(), anndatapp::Axis::gather({2, 0, 1})}};
    expect_invalid([&] {
        velocity::precision::SensitivityWorkspace32 wrong_input(other_input, workspace, stages, budget, staging);
    }, "sensitivity workspace rejects stages bound to a different prepared input object");
    velocity::Workspace other_workspace(data.half_input, options);
    expect_invalid([&] {
        velocity::precision::SensitivityWorkspace32 wrong_workspace(data.half_input, other_workspace, stages, budget, staging);
    }, "sensitivity workspace rejects stages bound to another workspace");

    auto stochastic_options = stochastic_run32();
    velocity::StochasticWorkspace stochastic_workspace(data.half_input, stochastic_options);
    velocity::StochasticOptions so;
    so.initialization = fit;
    velocity::StochasticReferenceStages32 stochastic_stages(data.half_input, stochastic_workspace, so);
    const auto stochastic_budget = stochastic_workspace.scratch_bytes() + score + staging;
    velocity::precision::SensitivityWorkspace32 stochastic_scratch(
        data.half_input, stochastic_workspace, stochastic_stages, stochastic_budget, staging);
    expect(stochastic_scratch.data() != nullptr && stochastic_scratch.scratch_bytes() == score,
           "stochastic sensitivity workspace accepts its correctly bound stage within budget");
    expect_invalid([&] {
        velocity::precision::SensitivityWorkspace32 too_small(
            data.half_input, stochastic_workspace, stochastic_stages, stochastic_budget - 1, staging);
    }, "stochastic sensitivity workspace charges caller staging against the same aggregate budget");
}

void test_steady_selective_promotion() {
    PromotionData data;
    const auto run = run32();
    const auto fit = data.fit_options();
    velocity::Workspace workspace(data.half_input, run), all_workspace(data.full_input, run), half_workspace(data.half_input, run);
    velocity::ReferenceStages32 reference(data.half_input, workspace, fit);
    SourceContext context{&data};
    const std::uint8_t requested[PromotionData::g]{1, 0, 1};
    velocity::precision::PromotionOptions promotion;
    promotion.source_gene_mask = requested;
    promotion.source_gene_count = PromotionData::g;
    promotion.workspace_bytes = 1u << 20;
    auto source = context.source(2 * PromotionData::n * sizeof(float));
    velocity::precision::SelectiveStages32 selective(data.half_input, workspace, reference, source, promotion);
    const auto selected = velocity::steady_state_with(selective, data.half_input.n_genes(), workspace.block_genes());
    const auto all = velocity::reference::steady_state(data.full_input, all_workspace, fit);
    const auto half = velocity::reference::steady_state(data.half_input, half_workspace, fit);

    const auto status = selective.promotion_status();
    const bool statuses_ok = selective.promotion_count() == 1 && status &&
        status[0] == std::uint8_t(velocity::precision::PromotionStatus::promoted) &&
        status[1] == std::uint8_t(velocity::precision::PromotionStatus::unavailable) &&
        status[2] == std::uint8_t(velocity::precision::PromotionStatus::not_requested);
    expect(data.half_input.gene_index(0) == 2 && data.half_input.gene_index(1) == 0 &&
           data.half_input.gene_index(2) == 1 && statuses_ok,
           "source-gene requests map through the prepared permutation and statuses remain in prepared order");
    bool diagnostics_ok = true;
    compare_fit_at<float>(selected, all, 0, 1e-6, diagnostics_ok);
    compare_fit_at<float>(selected, half, 1, 1e-6, diagnostics_ok);
    compare_fit_at<float>(selected, half, 2, 1e-6, diagnostics_ok);
    expect(diagnostics_ok,
           "promoted gene matches all-original diagnostics while unavailable and unrequested columns retain half-input results");
    expect(context.calls[2] == 1 && context.calls[0] == 1 && context.calls[1] == 0,
           "steady-state acquisition calls the source only for requested genes");

    SourceContext array_source_context{&data};
    velocity::precision::OriginalArraySource array_source(data.full_s_array.view(), data.full_u_array.view());
    velocity::Workspace array_workspace(data.half_input, run);
    velocity::ReferenceStages32 array_reference(data.half_input, array_workspace, fit);
    velocity::precision::SelectiveStages32 array_selective(data.half_input, array_workspace,
        array_reference, array_source.source(), promotion);
    const auto array_result = velocity::steady_state_with(
        array_selective, data.half_input.n_genes(), array_workspace.block_genes());
    bool array_ok = true;
    compare_fit_at<float>(array_result, all, 0, 1e-6, array_ok);
    expect(array_ok && array_selective.promotion_count() == 2,
           "dense OriginalArraySource acquires the requested permuted columns without staging copies");

    auto tight = promotion;
    tight.workspace_bytes = workspace.scratch_bytes();
    auto staged_source = context.source(1);
    expect_invalid([&] {
        velocity::precision::SelectiveStages32 rejected(data.half_input, workspace, reference, staged_source, tight);
    }, "selective promotion rejects caller staging beyond the remaining numerical workspace budget");
    tight.workspace_bytes = workspace.scratch_bytes() - 1;
    auto no_staging_source = context.source();
    expect_invalid([&] {
        velocity::precision::SelectiveStages32 rejected(data.half_input, workspace, reference, no_staging_source, tight);
    }, "selective promotion rejects a budget below the reference stage workspace");
}

void test_stochastic_selective_promotion() {
    PromotionData data;
    const auto run = stochastic_run32();
    velocity::StochasticRunOptions all_run = run, half_run = run;
    velocity::StochasticOptions stochastic;
    stochastic.initialization = data.fit_options();

    double original_xy = 0, original_xx = 0;
    for (std::size_t i = 0; i < PromotionData::n; ++i) {
        const double x = data.full_s[i * PromotionData::g + 2];
        const double y = data.full_u[i * PromotionData::g + 2];
        original_xy += x * y;
        original_xx += x * x;
    }
    const double original_gamma = original_xy / original_xx;
    expect(original_gamma > 2.0, "float original fixture retains a slope above its exact half-precision baseline");
    stochastic.initialization.min_ratio = (2.0 + original_gamma) / 2.0;

    velocity::StochasticWorkspace workspace(data.half_input, run),
        all_workspace(data.full_input, all_run), half_workspace(data.half_input, half_run);
    velocity::StochasticReferenceStages32 reference(data.half_input, workspace, stochastic);
    SourceContext context{&data};
    const std::uint8_t requested[PromotionData::g]{1, 0, 1};
    velocity::precision::PromotionOptions promotion;
    promotion.source_gene_mask = requested;
    promotion.source_gene_count = PromotionData::g;
    promotion.workspace_bytes = 1u << 20;
    auto source = context.source(2 * PromotionData::n * sizeof(float));
    velocity::precision::SelectiveStochasticStages32 selective(
        data.half_input, workspace, reference, source, promotion);
    const auto selected = velocity::stochastic_with(
        selective, data.half_input.n_genes(), workspace.block_genes());
    const auto all = velocity::reference::stochastic(data.full_input, all_workspace, stochastic);
    const auto half = velocity::reference::stochastic(data.half_input, half_workspace, stochastic);

    const auto promotion_status = selective.promotion_status();
    const bool mask_ok = selective.promotion_count() == 1 && promotion_status &&
        promotion_status[0] == std::uint8_t(velocity::precision::PromotionStatus::promoted) &&
        promotion_status[1] == std::uint8_t(velocity::precision::PromotionStatus::unavailable) &&
        promotion_status[2] == std::uint8_t(velocity::precision::PromotionStatus::not_requested);
    const auto selected_genes = boolean_values(selected.initialization.genes);
    const auto all_genes = boolean_values(all.initialization.genes);
    const auto half_genes = boolean_values(half.initialization.genes);
    const auto selected_status = byte_values(selected.status);
    bool results_ok = mask_ok && selected_genes[0] && all_genes[0] && !half_genes[0] &&
        selected_status[0] != std::uint8_t(velocity::StochasticStatus::not_selected);
    const auto sg = values<float>(selected.gamma), ag = values<float>(all.gamma), hg = values<float>(half.gamma);
    const auto so = values<float>(selected.offset), ao = values<float>(all.offset), ho = values<float>(half.offset);
    const auto so2 = values<float>(selected.offset2), ao2 = values<float>(all.offset2), ho2 = values<float>(half.offset2);
    const auto ss1 = values<float>(selected.first_scale), as1 = values<float>(all.first_scale), hs1 = values<float>(half.first_scale);
    const auto ss2 = values<float>(selected.second_scale), as2 = values<float>(all.second_scale), hs2 = values<float>(half.second_scale);
    const auto s_obj = values<float>(selected.objective), a_obj = values<float>(all.objective), h_obj = values<float>(half.objective);
    const auto s_status = byte_values(selected.status), a_status = byte_values(all.status), h_status = byte_values(half.status);
    for (std::size_t j = 0; j < PromotionData::g; ++j) {
        const bool promoted = j == 0;
        const auto& gamma = promoted ? ag : hg;
        const auto& offset = promoted ? ao : ho;
        const auto& offset2 = promoted ? ao2 : ho2;
        const auto& first = promoted ? as1 : hs1;
        const auto& second = promoted ? as2 : hs2;
        const auto& objective = promoted ? a_obj : h_obj;
        const auto& status = promoted ? a_status : h_status;
        const auto& genes = promoted ? all_genes : half_genes;
        results_ok &= close(sg[j], gamma[j], 2e-5) && close(so[j], offset[j], 2e-5) &&
            close(so2[j], offset2[j], 2e-5) && close(ss1[j], first[j], 2e-5) &&
            close(ss2[j], second[j], 2e-5) && close(s_obj[j], objective[j], 2e-5) &&
            s_status[j] == status[j] && selected_genes[j] == genes[j];
    }
    expect(results_ok,
           "stochastic promotion is applied before initialization selection and its outputs match the corresponding all-original or half baseline");
    expect(context.calls[2] == 2 && context.calls[0] == 1 && context.calls[1] == 0,
           "stochastic source callbacks run for requested genes during initialization and again for promoted genes on refinement");

    SourceContext no_request_context{&data};
    velocity::precision::PromotionOptions no_request;
    no_request.workspace_bytes = 1u << 20;
    velocity::StochasticWorkspace no_request_workspace(data.half_input, run);
    velocity::StochasticReferenceStages32 no_request_reference(data.half_input, no_request_workspace, stochastic);
    velocity::precision::SelectiveStochasticStages32 no_request_stages(data.half_input, no_request_workspace,
        no_request_reference, no_request_context.source(), no_request);
    (void)velocity::stochastic_with(no_request_stages, data.half_input.n_genes(), no_request_workspace.block_genes());
    expect(no_request_stages.promotion_count() == 0 &&
           no_request_context.calls[0] + no_request_context.calls[1] + no_request_context.calls[2] == 0,
           "an absent source-gene request mask never invokes the original-source callback");

    SourceContext disappearing{&data};
    disappearing.fail_gene2_after_first = true;
    velocity::StochasticWorkspace disappearing_workspace(data.half_input, run);
    velocity::StochasticReferenceStages32 disappearing_reference(data.half_input, disappearing_workspace, stochastic);
    velocity::precision::SelectiveStochasticStages32 disappearing_stages(data.half_input, disappearing_workspace,
        disappearing_reference, disappearing.source(), promotion);
    expect_runtime([&] {
        (void)velocity::stochastic_with(disappearing_stages, data.half_input.n_genes(), disappearing_workspace.block_genes());
    }, "an original promoted gene disappearing before refinement raises instead of mixing precisions");
    expect(disappearing.calls[2] == 2,
           "the disappearing-source case reaches the second-pass reacquisition boundary");
}

void test_steady_no_request_does_not_acquire() {
    PromotionData data;
    const auto run = run32();
    velocity::Workspace workspace(data.half_input, run);
    velocity::ReferenceStages32 reference(data.half_input, workspace, data.fit_options());
    SourceContext context{&data};
    velocity::precision::PromotionOptions promotion;
    promotion.workspace_bytes = workspace.scratch_bytes();
    velocity::precision::SelectiveStages32 selective(data.half_input, workspace, reference,
        context.source(std::numeric_limits<std::size_t>::max()), promotion);
    (void)velocity::steady_state_with(selective, data.half_input.n_genes(), workspace.block_genes());
    expect(selective.promotion_count() == 0 &&
           context.calls[0] + context.calls[1] + context.calls[2] == 0,
           "no request bypasses source acquisition and its declared staging budget");

    velocity::Workspace zero_workspace(data.half_input, run);
    velocity::ReferenceStages32 zero_reference(data.half_input, zero_workspace, data.fit_options());
    const std::uint8_t zero_mask[PromotionData::g]{};
    promotion.source_gene_mask = zero_mask;
    promotion.source_gene_count = PromotionData::g;
    promotion.workspace_bytes = zero_workspace.scratch_bytes();
    velocity::precision::SelectiveStages32 zero_selective(data.half_input, zero_workspace, zero_reference,
        context.source(std::numeric_limits<std::size_t>::max()), promotion);
    (void)velocity::steady_state_with(zero_selective, data.half_input.n_genes(), zero_workspace.block_genes());
    expect(zero_selective.promotion_count() == 0 &&
           context.calls[0] + context.calls[1] + context.calls[2] == 0,
           "an all-zero mask bypasses source acquisition and its declared staging budget");

    velocity::Workspace missing_workspace(data.half_input, run);
    velocity::ReferenceStages32 missing_reference(data.half_input, missing_workspace, data.fit_options());
    const std::uint8_t requested[PromotionData::g]{0, 0, 1};
    promotion.source_gene_mask = requested;
    promotion.workspace_bytes = missing_workspace.scratch_bytes();
    velocity::precision::OriginalSource missing{nullptr, nullptr, std::numeric_limits<std::size_t>::max()};
    velocity::precision::SelectiveStages32 missing_selective(data.half_input, missing_workspace,
        missing_reference, missing, promotion);
    (void)velocity::steady_state_with(missing_selective, data.half_input.n_genes(), missing_workspace.block_genes());
    expect(missing_selective.promotion_count() == 0 &&
           missing_selective.promotion_status()[0] == std::uint8_t(velocity::precision::PromotionStatus::unavailable),
           "missing original provider keeps the half path even if it advertises staging");
}

void test_fp16_original_is_unavailable() {
    PromotionData data;
    const auto run = run32();
    const auto fit = data.fit_options();
    velocity::Workspace workspace(data.half_input, run), baseline_workspace(data.half_input, run);
    velocity::ReferenceStages32 reference(data.half_input, workspace, fit);
    const auto baseline = velocity::reference::steady_state(data.half_input, baseline_workspace, fit);

    HalfSourceContext context;
    for (std::size_t i = 0; i < PromotionData::n; ++i) {
        context.spliced[i] = cellerator::numeric::round_f16(float(i + 20));
        context.unspliced[i] = cellerator::numeric::round_f16(float(3 * i + 7));
    }
    const std::uint8_t requested[PromotionData::g]{0, 0, 1};
    velocity::precision::PromotionOptions promotion;
    promotion.source_gene_mask = requested;
    promotion.source_gene_count = PromotionData::g;
    promotion.workspace_bytes = 1u << 20;
    velocity::precision::OriginalSource source{&context, HalfSourceContext::acquire, 0};
    velocity::precision::SelectiveStages32 selective(data.half_input, workspace, reference, source, promotion);
    const auto result = velocity::steady_state_with(selective, data.half_input.n_genes(), workspace.block_genes());

    const auto status = selective.promotion_status();
    bool same_as_half = selective.promotion_count() == 0 && status && context.calls == 1 &&
        status[0] == std::uint8_t(velocity::precision::PromotionStatus::unavailable) &&
        status[1] == std::uint8_t(velocity::precision::PromotionStatus::not_requested) &&
        status[2] == std::uint8_t(velocity::precision::PromotionStatus::not_requested);
    for (std::size_t j = 0; j < PromotionData::g; ++j)
        compare_fit_at<float>(result, baseline, j, 1e-6, same_as_half);
    expect(same_as_half,
           "a true callback carrying FP16 columns is unavailable and cannot widen or replace half-baseline moments");

    velocity::Workspace array_workspace(data.half_input, run);
    velocity::ReferenceStages32 array_reference(data.half_input, array_workspace, fit);
    velocity::precision::OriginalArraySource half_source(data.half_s_view, data.half_u_view);
    velocity::precision::SelectiveStages32 array_selective(
        data.half_input, array_workspace, array_reference, half_source.source(), promotion);
    const auto array_result = velocity::steady_state_with(
        array_selective, data.half_input.n_genes(), array_workspace.block_genes());
    const auto array_status = array_selective.promotion_status();
    bool array_same = array_selective.promotion_count() == 0 && array_status &&
        array_status[0] == std::uint8_t(velocity::precision::PromotionStatus::unavailable);
    for (std::size_t j = 0; j < PromotionData::g; ++j)
        compare_fit_at<float>(array_result, baseline, j, 1e-6, array_same);
    expect(array_same, "FP16 array source remains available to the uniform path without false promotion");
}

}

int main() {
    test_analytic_screen<double>();
    test_analytic_screen<float>();
    test_constant_cutoff_and_invalid_inputs<double>();
    test_constant_cutoff_and_invalid_inputs<float>();
    test_sensitivity_workspace_budget_and_binding();
    test_steady_selective_promotion();
    test_stochastic_selective_promotion();
    test_steady_no_request_does_not_acquire();
    test_fp16_original_is_unavailable();
    if (!failures) std::cout << "precision tests passed\n";
    return failures ? 1 : 0;
}
