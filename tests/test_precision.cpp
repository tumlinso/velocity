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

import velocitypp.precision;

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
    velocitypp::FitStatus status = velocitypp::FitStatus::insufficient_data;
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
                                  const velocitypp::FitOptions& options, bool enabled) {
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
    if (!(xx > 0)) return {0, 0, velocitypp::FitStatus::singular, false};

    const T gamma = xy / xx;
    T mean_y = 0, low_y = y[0], high_y = y[0];
    for (T value : y) {
        mean_y += value;
        low_y = std::min(low_y, value);
        high_y = std::max(high_y, value);
    }
    mean_y /= T(N);
    if (low_y == high_y) return {gamma, 0, velocitypp::FitStatus::constant_response, false};

    T residual = 0, total = 0;
    for (std::size_t i = 0; i < N; ++i) {
        const T e = y[i] - gamma * x[i];
        const T d = y[i] - mean_y;
        residual += e * e;
        total += d * d;
    }
    if (!(total > 0)) return {gamma, 0, velocitypp::FitStatus::constant_response, false};
    const T r2 = T(1) - residual / total;
    const bool selected = enabled && gamma > T(options.min_ratio) && r2 > T(options.min_r2);
    return {gamma, r2, velocitypp::FitStatus::ok, selected};
}

template<class T, std::size_t N>
struct AnalyticScreen {
    T low = 0, high = 0, delta = 0, prediction_change = 0, r2_change = 0;
    AnalyticFit<T> expanded, contracted;
    bool status_change = false, selection_change = false;
};

template<class T, std::size_t N>
AnalyticScreen<T, N> analytic_screen(const std::array<T, N>& x, const std::array<T, N>& y,
    const std::uint8_t* mask, const velocitypp::GeneFit& baseline,
    const velocitypp::FitOptions& fit, velocitypp::precision::SensitivityOptions options,
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
        if (out.expanded.status == velocitypp::FitStatus::ok) {
            const T d = out.expanded.gamma * x[i] - prediction;
            expanded_change2 += d * d;
        }
        if (out.contracted.status == velocitypp::FitStatus::ok) {
            const T d = out.contracted.gamma * x[i] - prediction;
            contracted_change2 += d * d;
        }
    }
    const T epsilon = std::numeric_limits<T>::epsilon();
    const T floor = T(32) * epsilon * std::max(std::sqrt(response2), std::sqrt(prediction2));
    const T denominator = std::max(std::sqrt(residual2), floor);
    out.prediction_change = std::sqrt(std::max(expanded_change2, contracted_change2)) / denominator;
    if (out.expanded.status == velocitypp::FitStatus::ok)
        out.r2_change = std::max(out.r2_change, std::abs(out.expanded.r2 - T(baseline.r2)));
    if (out.contracted.status == velocitypp::FitStatus::ok)
        out.r2_change = std::max(out.r2_change, std::abs(out.contracted.r2 - T(baseline.r2)));
    return out;
}

template<class T>
void test_analytic_screen() {
    constexpr std::size_t n = 5;
    const std::array<T, n> x{T(1), T(2), T(3), T(4), T(10)};
    const std::array<T, n> y{T(1), T(3), T(4.5), T(6), T(10)};
    velocitypp::FitOptions fit;
    fit.trimming = velocitypp::Trimming::upper;
    fit.percentile_high = 50;
    fit.fit_intercept = false;
    fit.min_r2 = -10;
    fit.min_ratio = 0;
    velocitypp::precision::SensitivityOptions options{128, 100, 100};
    const velocitypp::GeneFit baseline{1.3, .2, .61, velocitypp::FitStatus::ok, true};
    T scratch[n]{};

    const auto expected = analytic_screen(x, y, nullptr, baseline, fit, options, true);
    const auto report = velocitypp::precision::sensitivity_gene(x.data(), y.data(), n, 1,
        [&] {
            if constexpr (std::is_same_v<T, float>)
                return velocitypp::GeneFit32{T(baseline.gamma), T(baseline.offset), T(baseline.r2), baseline.status, baseline.selected};
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
           report.state == velocitypp::precision::SensitivityState::stable,
           "ordinary float/double screen remains advisory and stable when changes are below limits");

    const std::uint8_t mask[n]{1, 1, 1, 0, 1};
    const auto masked_expected = analytic_screen(x, y, mask, baseline, fit, options, true);
    const auto masked = velocitypp::precision::sensitivity_gene(x.data(), y.data(), n, 1,
        [&] {
            if constexpr (std::is_same_v<T, float>)
                return velocitypp::GeneFit32{T(baseline.gamma), T(baseline.offset), T(baseline.r2), baseline.status, baseline.selected};
            else return baseline;
        }(), fit, scratch, mask, true, options);
    expect(close(masked.high, analytic_percentile<T, n>(std::vector<T>{T(.2), T(.5), T(.75), T(2)}, T(50)), tolerance) &&
           !close(masked.high, report.high, tolerance) &&
           close(masked.prediction_change, masked_expected.prediction_change, tolerance) &&
           close(masked.r2_change, masked_expected.r2_change, tolerance),
           "fit mask changes cutoff and fitting rows while prediction diagnostics still cover every row");

    const auto disabled = velocitypp::precision::sensitivity_gene(x.data(), y.data(), n, 1,
        [&] {
            if constexpr (std::is_same_v<T, float>)
                return velocitypp::GeneFit32{T(baseline.gamma), T(baseline.offset), T(baseline.r2), baseline.status, true};
            else return baseline;
        }(), fit, scratch, mask, false, options);
    expect(disabled.selection_change,
           "disabled gene selection is reported as a selection change from the caller baseline");

    const auto at_cutoff = analytic_upper_fit(x, y, nullptr, T(.75), fit, true);
    velocitypp::GeneFit base0;
    base0.gamma = double(at_cutoff.gamma);
    base0.r2 = double(at_cutoff.r2);
    base0.status = at_cutoff.status;
    base0.selected = at_cutoff.selected;
    velocitypp::GeneFit32 base032{static_cast<float>(at_cutoff.gamma), 0,
        static_cast<float>(at_cutoff.r2), at_cutoff.status, at_cutoff.selected};
    velocitypp::precision::SensitivityOptions zero_band{0, 1e-4, 1e-4};
    const auto zero = [&] {
        if constexpr (std::is_same_v<T, float>)
            return velocitypp::precision::sensitivity_gene(x.data(), y.data(), n, 1, base032, fit,
                scratch, nullptr, true, zero_band);
        else
            return velocitypp::precision::sensitivity_gene(x.data(), y.data(), n, 1, base0, fit,
                scratch, nullptr, true, zero_band);
    }();
    expect(zero.state == velocitypp::precision::SensitivityState::stable && zero.delta == T(0) &&
           zero.expanded_status == velocitypp::FitStatus::ok &&
           zero.contracted_status == velocitypp::FitStatus::ok &&
           close(zero.prediction_change, T(0), tolerance) && close(zero.r2_change, T(0), tolerance),
           "zero-width screen gives the same fit as the supplied baseline with no reported change");
}

template<class T>
void test_constant_cutoff_and_invalid_inputs() {
    constexpr std::size_t n = 4;
    const std::array<T, n> x{T(1), T(2), T(3), T(4)};
    const std::array<T, n> y{T(4), T(3), T(2), T(1)};
    velocitypp::FitOptions fit;
    fit.trimming = velocitypp::Trimming::upper;
    fit.percentile_high = 50;
    fit.fit_intercept = false;
    fit.min_r2 = -4;
    fit.min_ratio = 0;
    T scratch[n]{};
    const double gamma = 2.0 / 3.0, r2 = -7.0 / 3.0;
    velocitypp::precision::SensitivityOptions zero_band{0, 1e-5, 1e-5};
    auto base = [&] {
        if constexpr (std::is_same_v<T, float>)
            return velocitypp::GeneFit32{T(gamma), T(0), T(r2), velocitypp::FitStatus::ok, true};
        else return velocitypp::GeneFit{gamma, 0, r2, velocitypp::FitStatus::ok, true};
    }();
    const auto exact = velocitypp::precision::sensitivity_gene(x.data(), y.data(), n, 1,
        base, fit, scratch, nullptr, true, zero_band);
    expect(close(exact.high, T(1.25), 2e-6) && exact.delta == T(0) &&
           exact.state == velocitypp::precision::SensitivityState::stable &&
           close(exact.prediction_change, T(0), 2e-6) && close(exact.r2_change, T(0), 2e-6),
           "exactly constant normalized scores retain the caller's matching baseline at zero band");

    velocitypp::precision::SensitivityOptions stress{4096, 100, 100};
    const auto stressed = velocitypp::precision::sensitivity_gene(x.data(), y.data(), n, 1,
        base, fit, scratch, nullptr, true, stress);
    expect(stressed.status_change && stressed.selection_change &&
           stressed.contracted_status == velocitypp::FitStatus::insufficient_data &&
           stressed.state == velocitypp::precision::SensitivityState::sensitive,
           "large sensitivity band crossing an exact-score tie changes eligibility and fit status");

    velocitypp::FitOptions no_trim = fit;
    no_trim.trimming = velocitypp::Trimming::none;
    const auto untrimmed = velocitypp::precision::sensitivity_gene(
        static_cast<const T*>(nullptr), static_cast<const T*>(nullptr), 0, 1,
        base, no_trim, static_cast<T*>(nullptr));
    expect(untrimmed.state == velocitypp::precision::SensitivityState::no_trimming &&
           untrimmed.expanded_status == base.status && untrimmed.contracted_status == base.status,
           "untrimmed fits need no score scratch and are reported as no-trimming");

    auto invalid_base = base;
    invalid_base.status = velocitypp::FitStatus::singular;
    const auto degenerate = velocitypp::precision::sensitivity_gene(
        x.data(), y.data(), n, 1, invalid_base, fit, scratch);
    expect(degenerate.state == velocitypp::precision::SensitivityState::unscoreable &&
           degenerate.expanded_status == velocitypp::FitStatus::singular &&
           degenerate.contracted_status == velocitypp::FitStatus::singular &&
           !degenerate.status_change && !degenerate.selection_change,
           "a degenerate supplied fit is unscoreable without fabricating a status change");

    velocitypp::precision::SensitivityOptions bad;
    bad.band = -1;
    expect_invalid([&] { (void)velocitypp::precision::sensitivity_gene(x.data(), y.data(), n, 1, base, fit, scratch, nullptr, true, bad); },
                   "negative sensitivity band is rejected");
    bad = {};
    bad.prediction_limit = std::numeric_limits<double>::quiet_NaN();
    expect_invalid([&] { (void)velocitypp::precision::sensitivity_gene(x.data(), y.data(), n, 1, base, fit, scratch, nullptr, true, bad); },
                   "nonfinite sensitivity limit is rejected");
    bad = {};
    expect_invalid([&] { (void)velocitypp::precision::sensitivity_gene(x.data(), y.data(), n, 0, base, fit, scratch); },
                   "zero input stride is rejected");
    expect_invalid([&] { (void)velocitypp::precision::sensitivity_gene(x.data(), y.data(), n, std::numeric_limits<std::ptrdiff_t>::max(), base, fit, scratch); },
                   "row-offset overflow from an oversized stride is rejected before dereference");
    auto bad_fit = fit;
    bad_fit.percentile_low = 80;
    bad_fit.percentile_high = 20;
    expect_invalid([&] { (void)velocitypp::precision::sensitivity_gene(x.data(), y.data(), n, 1, base, bad_fit, scratch); },
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
    static std::array<double, n * g> make_spliced() {
        std::array<double, n * g> out{};
        for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < g; ++j)
            out[i * g + j] = j == 2 ? double(i + 1) : 1.0 + .7 * double(i) + .3 * double(j) + .02 * double(i * j);
        return out;
    }
    static std::array<double, n * g> make_unspliced() {
        std::array<double, n * g> out{};
        constexpr double noise[n]{.13, -.20, .11, -.07, .17, -.15, .06, -.10};
        constexpr double tiny[n]{-.0004, .0004, -.0004, .0004, -.0004, .0004, -.0004, .0004};
        constexpr double slopes[g]{.6, 1.2, 0};
        for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < g; ++j) {
            const double x = j == 2 ? double(i + 1) : 1.0 + .7 * double(i) + .3 * double(j) + .02 * double(i * j);
            out[i * g + j] = j == 2 ? 2.0001 * x + tiny[i]
                                    : slopes[j] * x + .5 + .2 * double(j) + noise[i] * double(j + 1);
        }
        return out;
    }
    static std::array<float, n * g> quantize_for_baseline(const std::array<double, n * g>& x) {
        std::array<float, n * g> out{};
        for (std::size_t i = 0; i < x.size(); ++i)
            out[i] = static_cast<float>(std::round(x[i] * 100.0) / 100.0);
        return out;
    }
    static std::array<float, n * n> identity() {
        std::array<float, n * n> out{};
        for (std::size_t i = 0; i < n; ++i) out[i * n + i] = 1;
        return out;
    }

    std::array<double, n * g> full_s = make_spliced();
    std::array<double, n * g> full_u = make_unspliced();
    std::array<float, n * g> baseline_s = quantize_for_baseline(full_s);
    std::array<float, n * g> baseline_u = quantize_for_baseline(full_u);
    std::array<float, n * n> weights = identity();
    anndatapp::Array full_s_array = anndatapp::Array::borrow_dense(full_s.data(), {n, g}, {std::ptrdiff_t(g), 1});
    anndatapp::Array full_u_array = anndatapp::Array::borrow_dense(full_u.data(), {n, g}, {std::ptrdiff_t(g), 1});
    anndatapp::Array baseline_s_array = anndatapp::Array::borrow_dense(baseline_s.data(), {n, g}, {std::ptrdiff_t(g), 1});
    anndatapp::Array baseline_u_array = anndatapp::Array::borrow_dense(baseline_u.data(), {n, g}, {std::ptrdiff_t(g), 1});
    anndatapp::Array graph_array = anndatapp::Array::borrow_dense(weights.data(), {n, n}, {std::ptrdiff_t(n), 1});
    velocitypp::PreparedInput full_input{{full_s_array.view(), full_u_array.view(), graph_array.view(), anndatapp::Axis::gather({2, 0, 1})}};
    velocitypp::PreparedInput baseline_input{{baseline_s_array.view(), baseline_u_array.view(), graph_array.view(), anndatapp::Axis::gather({2, 0, 1})}};

    velocitypp::FitOptions fit_options() const {
        velocitypp::FitOptions o;
        o.trimming = velocitypp::Trimming::none;
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

    static bool acquire(void* context, std::size_t source_gene, velocitypp::precision::OriginalColumn& column) {
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

    velocitypp::precision::OriginalSource source(std::size_t staging = 0) {
        return {this, acquire, staging};
    }
};

velocitypp::RunOptions run32() {
    velocitypp::RunOptions o;
    o.workspace_bytes = 1u << 20;
    o.block_genes = 1;
    o.output_dtype = anndatapp::DType::float32;
    o.compute_precision = velocitypp::ComputePrecision::float32;
    return o;
}

velocitypp::StochasticRunOptions stochastic_run32() {
    velocitypp::StochasticRunOptions o;
    o.workspace_bytes = 1u << 20;
    o.block_genes = 1;
    o.output_dtype = anndatapp::DType::float32;
    o.compute_precision = velocitypp::ComputePrecision::float32;
    return o;
}

template<class T>
void compare_fit_at(const velocitypp::FitResult& actual, const velocitypp::FitResult& expected,
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
    velocitypp::Workspace workspace(data.baseline_input, options);
    auto fit = data.fit_options();
    velocitypp::ReferenceStages32 stages(data.baseline_input, workspace, fit);
    constexpr std::size_t staging = 64;
    const auto score = PromotionData::n * sizeof(float);
    const auto budget = workspace.scratch_bytes() + score + staging;
    velocitypp::precision::SensitivityWorkspace32 scratch(data.baseline_input, workspace, stages, budget, staging);
    expect(scratch.data() != nullptr && scratch.scratch_bytes() == score,
           "sensitivity workspace allocates only its row score buffer under the exact aggregate budget");
    expect_invalid([&] {
        velocitypp::precision::SensitivityWorkspace32 too_small(data.baseline_input, workspace, stages, budget - 1, staging);
    }, "sensitivity workspace rejects a budget one byte below stage plus score plus staging");

    velocitypp::PreparedInput other_input{{data.baseline_s_array.view(), data.baseline_u_array.view(),
        data.graph_array.view(), anndatapp::Axis::gather({2, 0, 1})}};
    expect_invalid([&] {
        velocitypp::precision::SensitivityWorkspace32 wrong_input(other_input, workspace, stages, budget, staging);
    }, "sensitivity workspace rejects stages bound to a different prepared input object");
    velocitypp::Workspace other_workspace(data.baseline_input, options);
    expect_invalid([&] {
        velocitypp::precision::SensitivityWorkspace32 wrong_workspace(data.baseline_input, other_workspace, stages, budget, staging);
    }, "sensitivity workspace rejects stages bound to another workspace");

    auto stochastic_options = stochastic_run32();
    velocitypp::StochasticWorkspace stochastic_workspace(data.baseline_input, stochastic_options);
    velocitypp::StochasticOptions so;
    so.initialization = fit;
    velocitypp::StochasticReferenceStages32 stochastic_stages(data.baseline_input, stochastic_workspace, so);
    const auto stochastic_budget = stochastic_workspace.scratch_bytes() + score + staging;
    velocitypp::precision::SensitivityWorkspace32 stochastic_scratch(
        data.baseline_input, stochastic_workspace, stochastic_stages, stochastic_budget, staging);
    expect(stochastic_scratch.data() != nullptr && stochastic_scratch.scratch_bytes() == score,
           "stochastic sensitivity workspace accepts its correctly bound stage within budget");
    expect_invalid([&] {
        velocitypp::precision::SensitivityWorkspace32 too_small(
            data.baseline_input, stochastic_workspace, stochastic_stages, stochastic_budget - 1, staging);
    }, "stochastic sensitivity workspace charges caller staging against the same aggregate budget");
}

void test_steady_selective_promotion() {
    PromotionData data;
    const auto run = run32();
    const auto fit = data.fit_options();
    velocitypp::Workspace workspace(data.baseline_input, run), all_workspace(data.full_input, run), baseline_workspace(data.baseline_input, run);
    velocitypp::ReferenceStages32 reference(data.baseline_input, workspace, fit);
    SourceContext context{&data};
    const std::uint8_t requested[PromotionData::g]{1, 0, 1};
    velocitypp::precision::PromotionOptions promotion;
    promotion.source_gene_mask = requested;
    promotion.source_gene_count = PromotionData::g;
    promotion.workspace_bytes = 1u << 20;
    auto source = context.source(2 * PromotionData::n * sizeof(double));
    velocitypp::precision::SelectiveStages32 selective(data.baseline_input, workspace, reference, source, promotion);
    const auto selected = velocitypp::steady_state_with(selective, data.baseline_input.n_genes(), workspace.block_genes());
    const auto all = velocitypp::reference::steady_state(data.full_input, all_workspace, fit);
    const auto baseline = velocitypp::reference::steady_state(data.baseline_input, baseline_workspace, fit);

    const auto status = selective.promotion_status();
    const bool statuses_ok = selective.promotion_count() == 1 && status &&
        status[0] == std::uint8_t(velocitypp::precision::PromotionStatus::promoted) &&
        status[1] == std::uint8_t(velocitypp::precision::PromotionStatus::unavailable) &&
        status[2] == std::uint8_t(velocitypp::precision::PromotionStatus::not_requested);
    expect(data.baseline_input.gene_index(0) == 2 && data.baseline_input.gene_index(1) == 0 &&
           data.baseline_input.gene_index(2) == 1 && statuses_ok,
           "source-gene requests map through the prepared permutation and statuses remain in prepared order");
    bool diagnostics_ok = true;
    compare_fit_at<float>(selected, all, 0, 1e-6, diagnostics_ok);
    compare_fit_at<float>(selected, baseline, 1, 1e-6, diagnostics_ok);
    compare_fit_at<float>(selected, baseline, 2, 1e-6, diagnostics_ok);
    expect(diagnostics_ok,
           "promoted gene matches FP64-source diagnostics while unavailable and unrequested columns retain quantized-FP32 baseline results");
    expect(context.calls[2] == 1 && context.calls[0] == 1 && context.calls[1] == 0,
           "steady-state acquisition calls the source only for requested genes");

    SourceContext array_source_context{&data};
    velocitypp::precision::OriginalArraySource array_source(data.full_s_array.view(), data.full_u_array.view());
    velocitypp::Workspace array_workspace(data.baseline_input, run);
    velocitypp::ReferenceStages32 array_reference(data.baseline_input, array_workspace, fit);
    velocitypp::precision::SelectiveStages32 array_selective(data.baseline_input, array_workspace,
        array_reference, array_source.source(), promotion);
    const auto array_result = velocitypp::steady_state_with(
        array_selective, data.baseline_input.n_genes(), array_workspace.block_genes());
    bool array_ok = true;
    compare_fit_at<float>(array_result, all, 0, 1e-6, array_ok);
    expect(array_ok && array_selective.promotion_count() == 2,
           "dense OriginalArraySource acquires the requested permuted columns without staging copies");

    auto tight = promotion;
    tight.workspace_bytes = workspace.scratch_bytes();
    auto staged_source = context.source(1);
    expect_invalid([&] {
        velocitypp::precision::SelectiveStages32 rejected(data.baseline_input, workspace, reference, staged_source, tight);
    }, "selective promotion rejects caller staging beyond the remaining numerical workspace budget");
    tight.workspace_bytes = workspace.scratch_bytes() - 1;
    auto no_staging_source = context.source();
    expect_invalid([&] {
        velocitypp::precision::SelectiveStages32 rejected(data.baseline_input, workspace, reference, no_staging_source, tight);
    }, "selective promotion rejects a budget below the reference stage workspace");
}

void test_stochastic_selective_promotion() {
    PromotionData data;
    const auto run = stochastic_run32();
    velocitypp::StochasticRunOptions all_run = run, baseline_run = run;
    velocitypp::StochasticOptions stochastic;
    stochastic.initialization = data.fit_options();

    double original_xy = 0, original_xx = 0;
    for (std::size_t i = 0; i < PromotionData::n; ++i) {
        const double x = data.full_s[i * PromotionData::g + 2];
        const double y = data.full_u[i * PromotionData::g + 2];
        original_xy += x * y;
        original_xx += x * x;
    }
    const double original_gamma = original_xy / original_xx;
    expect(original_gamma > 2.0, "FP64 original fixture retains a slope above its quantized-FP32 baseline");
    stochastic.initialization.min_ratio = (2.0 + original_gamma) / 2.0;

    velocitypp::StochasticWorkspace workspace(data.baseline_input, run),
        all_workspace(data.full_input, all_run), baseline_workspace(data.baseline_input, baseline_run);
    velocitypp::StochasticReferenceStages32 reference(data.baseline_input, workspace, stochastic);
    SourceContext context{&data};
    const std::uint8_t requested[PromotionData::g]{1, 0, 1};
    velocitypp::precision::PromotionOptions promotion;
    promotion.source_gene_mask = requested;
    promotion.source_gene_count = PromotionData::g;
    promotion.workspace_bytes = 1u << 20;
    auto source = context.source(2 * PromotionData::n * sizeof(double));
    velocitypp::precision::SelectiveStochasticStages32 selective(
        data.baseline_input, workspace, reference, source, promotion);
    const auto selected = velocitypp::stochastic_with(
        selective, data.baseline_input.n_genes(), workspace.block_genes());
    const auto all = velocitypp::reference::stochastic(data.full_input, all_workspace, stochastic);
    const auto baseline = velocitypp::reference::stochastic(data.baseline_input, baseline_workspace, stochastic);

    const auto promotion_status = selective.promotion_status();
    const bool mask_ok = selective.promotion_count() == 1 && promotion_status &&
        promotion_status[0] == std::uint8_t(velocitypp::precision::PromotionStatus::promoted) &&
        promotion_status[1] == std::uint8_t(velocitypp::precision::PromotionStatus::unavailable) &&
        promotion_status[2] == std::uint8_t(velocitypp::precision::PromotionStatus::not_requested);
    const auto selected_genes = boolean_values(selected.initialization.genes);
    const auto all_genes = boolean_values(all.initialization.genes);
    const auto baseline_genes = boolean_values(baseline.initialization.genes);
    const auto selected_status = byte_values(selected.status);
    bool results_ok = mask_ok && selected_genes[0] && all_genes[0] && !baseline_genes[0] &&
        selected_status[0] != std::uint8_t(velocitypp::StochasticStatus::not_selected);
    const auto sg = values<float>(selected.gamma), ag = values<float>(all.gamma), bg = values<float>(baseline.gamma);
    const auto so = values<float>(selected.offset), ao = values<float>(all.offset), bo = values<float>(baseline.offset);
    const auto so2 = values<float>(selected.offset2), ao2 = values<float>(all.offset2), bo2 = values<float>(baseline.offset2);
    const auto ss1 = values<float>(selected.first_scale), as1 = values<float>(all.first_scale), bs1 = values<float>(baseline.first_scale);
    const auto ss2 = values<float>(selected.second_scale), as2 = values<float>(all.second_scale), bs2 = values<float>(baseline.second_scale);
    const auto s_obj = values<float>(selected.objective), a_obj = values<float>(all.objective), b_obj = values<float>(baseline.objective);
    const auto s_status = byte_values(selected.status), a_status = byte_values(all.status), b_status = byte_values(baseline.status);
    for (std::size_t j = 0; j < PromotionData::g; ++j) {
        const bool promoted = j == 0;
        const auto& gamma = promoted ? ag : bg;
        const auto& offset = promoted ? ao : bo;
        const auto& offset2 = promoted ? ao2 : bo2;
        const auto& first = promoted ? as1 : bs1;
        const auto& second = promoted ? as2 : bs2;
        const auto& objective = promoted ? a_obj : b_obj;
        const auto& status = promoted ? a_status : b_status;
        const auto& genes = promoted ? all_genes : baseline_genes;
        results_ok &= close(sg[j], gamma[j], 2e-5) && close(so[j], offset[j], 2e-5) &&
            close(so2[j], offset2[j], 2e-5) && close(ss1[j], first[j], 2e-5) &&
            close(ss2[j], second[j], 2e-5) && close(s_obj[j], objective[j], 2e-5) &&
            s_status[j] == status[j] && selected_genes[j] == genes[j];
    }
    expect(results_ok,
           "stochastic promotion is applied before initialization selection and matches either the FP64-source or quantized-FP32 baseline");
    expect(context.calls[2] == 2 && context.calls[0] == 1 && context.calls[1] == 0,
           "stochastic source callbacks run for requested genes during initialization and again for promoted genes on refinement");

    SourceContext no_request_context{&data};
    velocitypp::precision::PromotionOptions no_request;
    no_request.workspace_bytes = 1u << 20;
    velocitypp::StochasticWorkspace no_request_workspace(data.baseline_input, run);
    velocitypp::StochasticReferenceStages32 no_request_reference(data.baseline_input, no_request_workspace, stochastic);
    velocitypp::precision::SelectiveStochasticStages32 no_request_stages(data.baseline_input, no_request_workspace,
        no_request_reference, no_request_context.source(), no_request);
    (void)velocitypp::stochastic_with(no_request_stages, data.baseline_input.n_genes(), no_request_workspace.block_genes());
    expect(no_request_stages.promotion_count() == 0 &&
           no_request_context.calls[0] + no_request_context.calls[1] + no_request_context.calls[2] == 0,
           "an absent source-gene request mask never invokes the original-source callback");

    SourceContext disappearing{&data};
    disappearing.fail_gene2_after_first = true;
    velocitypp::StochasticWorkspace disappearing_workspace(data.baseline_input, run);
    velocitypp::StochasticReferenceStages32 disappearing_reference(data.baseline_input, disappearing_workspace, stochastic);
    velocitypp::precision::SelectiveStochasticStages32 disappearing_stages(data.baseline_input, disappearing_workspace,
        disappearing_reference, disappearing.source(), promotion);
    expect_runtime([&] {
        (void)velocitypp::stochastic_with(disappearing_stages, data.baseline_input.n_genes(), disappearing_workspace.block_genes());
    }, "an original promoted gene disappearing before refinement raises instead of mixing precisions");
    expect(disappearing.calls[2] == 2,
           "the disappearing-source case reaches the second-pass reacquisition boundary");
}

void test_steady_no_request_does_not_acquire() {
    PromotionData data;
    const auto run = run32();
    velocitypp::Workspace workspace(data.baseline_input, run);
    velocitypp::ReferenceStages32 reference(data.baseline_input, workspace, data.fit_options());
    SourceContext context{&data};
    velocitypp::precision::PromotionOptions promotion;
    promotion.workspace_bytes = workspace.scratch_bytes();
    velocitypp::precision::SelectiveStages32 selective(data.baseline_input, workspace, reference,
        context.source(std::numeric_limits<std::size_t>::max()), promotion);
    (void)velocitypp::steady_state_with(selective, data.baseline_input.n_genes(), workspace.block_genes());
    expect(selective.promotion_count() == 0 &&
           context.calls[0] + context.calls[1] + context.calls[2] == 0,
           "no request bypasses source acquisition and its declared staging budget");

    velocitypp::Workspace zero_workspace(data.baseline_input, run);
    velocitypp::ReferenceStages32 zero_reference(data.baseline_input, zero_workspace, data.fit_options());
    const std::uint8_t zero_mask[PromotionData::g]{};
    promotion.source_gene_mask = zero_mask;
    promotion.source_gene_count = PromotionData::g;
    promotion.workspace_bytes = zero_workspace.scratch_bytes();
    velocitypp::precision::SelectiveStages32 zero_selective(data.baseline_input, zero_workspace, zero_reference,
        context.source(std::numeric_limits<std::size_t>::max()), promotion);
    (void)velocitypp::steady_state_with(zero_selective, data.baseline_input.n_genes(), zero_workspace.block_genes());
    expect(zero_selective.promotion_count() == 0 &&
           context.calls[0] + context.calls[1] + context.calls[2] == 0,
           "an all-zero mask bypasses source acquisition and its declared staging budget");

    velocitypp::Workspace missing_workspace(data.baseline_input, run);
    velocitypp::ReferenceStages32 missing_reference(data.baseline_input, missing_workspace, data.fit_options());
    const std::uint8_t requested[PromotionData::g]{0, 0, 1};
    promotion.source_gene_mask = requested;
    promotion.workspace_bytes = missing_workspace.scratch_bytes();
    velocitypp::precision::OriginalSource missing{nullptr, nullptr, std::numeric_limits<std::size_t>::max()};
    velocitypp::precision::SelectiveStages32 missing_selective(data.baseline_input, missing_workspace,
        missing_reference, missing, promotion);
    (void)velocitypp::steady_state_with(missing_selective, data.baseline_input.n_genes(), missing_workspace.block_genes());
    expect(missing_selective.promotion_count() == 0 &&
           missing_selective.promotion_status()[0] == std::uint8_t(velocitypp::precision::PromotionStatus::unavailable),
           "missing original provider keeps baseline results even if it advertises staging");
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
    if (!failures) std::cout << "precision tests passed\n";
    return failures ? 1 : 0;
}
