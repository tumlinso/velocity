#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

import velocitypp.precision;
#include "support.hh"

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
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
    expect(velocitypp::precision::validate_sensitivity(n, 1, x.data(), y.data(), scratch, n, fit, bad).code ==
               velocitypp::precision::PrecisionCode::invalid_argument,
           "negative sensitivity band is rejected by explicit validation");
    bad = {};
    bad.prediction_limit = std::numeric_limits<double>::quiet_NaN();
    expect(velocitypp::precision::validate_sensitivity(n, 1, x.data(), y.data(), scratch, n, fit, bad).code ==
               velocitypp::precision::PrecisionCode::invalid_argument,
           "nonfinite sensitivity limit is rejected by explicit validation");
    bad = {};
    expect(velocitypp::precision::validate_sensitivity(n, 0, x.data(), y.data(), scratch, n, fit, bad).code ==
               velocitypp::precision::PrecisionCode::invalid_argument,
           "zero input stride is rejected by explicit validation");
    expect(velocitypp::precision::validate_sensitivity(n, std::numeric_limits<std::ptrdiff_t>::max(),
               x.data(), y.data(), scratch, n, fit, bad).code ==
               velocitypp::precision::PrecisionCode::invalid_argument,
           "row-offset overflow from an oversized stride is rejected before dereference");
    auto bad_fit = fit;
    bad_fit.percentile_low = 80;
    bad_fit.percentile_high = 20;
    expect(velocitypp::precision::validate_sensitivity(n, 1, x.data(), y.data(), scratch, n, bad_fit).code ==
               velocitypp::precision::PrecisionCode::invalid_argument,
           "invalid fitting percentiles are rejected by explicit validation");
    expect(velocitypp::precision::validate_sensitivity(n, 1, x.data(), y.data(), scratch, n - 1, fit).code ==
               velocitypp::precision::PrecisionCode::insufficient_capacity,
           "trimmed sensitivity validation reports the required row-score capacity");
}

template<class T> struct FitBuffers {
    std::vector<T> gamma, offset, r2;
    std::unique_ptr<bool[]> selected;
    std::vector<std::uint8_t> status;
    velocitypp::BasicFitResult<T> result;

    explicit FitBuffers(std::size_t genes)
        : gamma(genes), offset(genes), r2(genes), selected(new bool[genes]{}), status(genes) {
        result = {gamma.data(), offset.data(), r2.data(), selected.get(), status.data(), genes, false};
    }
};

template<class T> struct StochasticBuffers {
    FitBuffers<T> initialization;
    std::vector<T> gamma, offset, offset2, first_scale, second_scale, objective;
    std::vector<std::uint8_t> status;
    velocitypp::BasicStochasticFitResult<T> result;

    explicit StochasticBuffers(std::size_t genes)
        : initialization(genes), gamma(genes), offset(genes), offset2(genes), first_scale(genes),
          second_scale(genes), objective(genes), status(genes) {
        result = {initialization.result, gamma.data(), offset.data(), offset2.data(),
                  first_scale.data(), second_scale.data(), objective.data(), status.data(), genes, false};
    }
};

template<class T> struct WorkspaceBuffers {
    std::vector<T> scalars;
    std::vector<std::uint8_t> mask;
    velocitypp::Workspace<T> workspace;

    bool bind(const velocitypp::PreparedInput& input, std::size_t block_genes, bool stochastic) {
        velocitypp::RunOptions options;
        options.workspace_bytes = 1u << 20;
        options.block_genes = block_genes;
        velocitypp::WorkspaceSize needed;
        if (!velocitypp::workspace_size<T>(input, options, stochastic, needed)) return false;
        scalars.resize(needed.scalars);
        mask.resize(needed.masks);
        return bool(velocitypp::bind_workspace(input, options, stochastic, scalars.data(), scalars.size(),
                                               mask.data(), mask.size(), workspace));
    }
};

bool prepare_dense(velocitypp::Input input, velocitypp::PreparedInput& prepared) {
    velocitypp::PreparationSize size;
    if (!velocitypp::preparation_size(input, size) || size.indices || size.entries) return false;
    return bool(velocitypp::prepare(input, {}, prepared));
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
    static std::array<float, n * g> quantize(const std::array<double, n * g>& x) {
        std::array<float, n * g> out{};
        for (std::size_t i = 0; i < x.size(); ++i)
            out[i] = static_cast<float>(std::round(x[i] * 100.0) / 100.0);
        return out;
    }
    static std::array<float, n * g> narrow(const std::array<double, n * g>& x) {
        std::array<float, n * g> out{};
        for (std::size_t i = 0; i < x.size(); ++i) out[i] = static_cast<float>(x[i]);
        return out;
    }
    static std::array<float, n * n> identity() {
        std::array<float, n * n> out{};
        for (std::size_t i = 0; i < n; ++i) out[i * n + i] = 1;
        return out;
    }

    std::array<double, n * g> full_s = make_spliced();
    std::array<double, n * g> full_u = make_unspliced();
    std::array<float, n * g> mixed_u = narrow(full_u);
    std::array<float, n * g> baseline_s = quantize(full_s);
    std::array<float, n * g> baseline_u = quantize(full_u);
    std::array<float, n * n> weights = identity();
    std::array<std::size_t, g> gene_order{2, 0, 1};
    anndatapp::Array full_s_array = anndatapp::Array::borrow_dense(full_s.data(), {n, g}, {std::ptrdiff_t(g), 1});
    anndatapp::Array full_u_array = anndatapp::Array::borrow_dense(full_u.data(), {n, g}, {std::ptrdiff_t(g), 1});
    anndatapp::Array baseline_s_array = anndatapp::Array::borrow_dense(baseline_s.data(), {n, g}, {std::ptrdiff_t(g), 1});
    anndatapp::Array baseline_u_array = anndatapp::Array::borrow_dense(baseline_u.data(), {n, g}, {std::ptrdiff_t(g), 1});
    anndatapp::Array graph_array = anndatapp::Array::borrow_dense(weights.data(), {n, n}, {std::ptrdiff_t(n), 1});
    Acquired full_s_acquired{full_s_array.view()}, full_u_acquired{full_u_array.view()};
    Acquired baseline_s_acquired{baseline_s_array.view()}, baseline_u_acquired{baseline_u_array.view()};
    Acquired graph_acquired{graph_array.view()};
    velocitypp::Input full_input{}, baseline_input{};
    velocitypp::PreparedInput full_prepared{}, baseline_prepared{};
    bool ready = false;

    PromotionData() {
        full_input = {full_s_acquired.matrix, full_u_acquired.matrix, graph_acquired.matrix, gene_order.data(), g};
        baseline_input = {baseline_s_acquired.matrix, baseline_u_acquired.matrix, graph_acquired.matrix, gene_order.data(), g};
        ready = prepare_dense(full_input, full_prepared) && prepare_dense(baseline_input, baseline_prepared);
    }

    velocitypp::FitOptions fit_options() const {
        velocitypp::FitOptions options;
        options.trimming = velocitypp::Trimming::none;
        options.fit_intercept = false;
        options.min_r2 = 0;
        options.min_ratio = 0;
        return options;
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
        column = {self.data->full_s.data() + source_gene, self.data->mixed_u.data() + source_gene,
                  std::ptrdiff_t(PromotionData::g * sizeof(double)),
                  std::ptrdiff_t(PromotionData::g * sizeof(float)),
                  velocitypp::precision::OriginalType::float64,
                  velocitypp::precision::OriginalType::float32, PromotionData::n};
        return true;
    }
};

template<class T> bool run_steady(const velocitypp::PreparedInput& input,
                                  velocitypp::FitOptions fit, FitBuffers<T>& result) {
    WorkspaceBuffers<T> buffers;
    if (!buffers.bind(input, 1, false)) return false;
    return bool(velocitypp::reference::steady_state(input, buffers.workspace, result.result, fit));
}

template<class T> bool run_stochastic(const velocitypp::PreparedInput& input,
    velocitypp::StochasticOptions options, StochasticBuffers<T>& result) {
    WorkspaceBuffers<T> buffers;
    if (!buffers.bind(input, 1, true)) return false;
    return bool(velocitypp::reference::stochastic(input, buffers.workspace, result.result, options));
}

template<class T> bool fit_matches_at(const FitBuffers<T>& a, const FitBuffers<T>& b,
                                      std::size_t gene, double tolerance) {
    return close(a.gamma[gene], b.gamma[gene], tolerance) &&
        close(a.offset[gene], b.offset[gene], tolerance) && close(a.r2[gene], b.r2[gene], tolerance) &&
        a.status[gene] == b.status[gene] && a.selected[gene] == b.selected[gene];
}

template<class T> bool promote_steady(PromotionData& data, const velocitypp::PreparedInput& input,
    FitBuffers<T>& output, WorkspaceBuffers<T>& workspace, velocitypp::precision::Promotion& promotion) {
    if (!workspace.bind(input, 1, false)) return false;
    velocitypp::BasicReferenceStages<T> stages{};
    if (!velocitypp::bind_stages(input, workspace.workspace, data.fit_options(), output.result, stages)) return false;
    if (velocitypp::precision::initialize_promotion(promotion).code != velocitypp::precision::PrecisionCode::ok) return false;
    for (std::size_t first = 0; first < input.genes;) {
        const std::size_t count = input.genes - first < workspace.workspace.block ? input.genes - first : workspace.workspace.block;
        auto moments = stages.moments(first, count);
        velocitypp::precision::ReferenceMoments<T> reference{&input, &moments, nullptr};
        promotion.replacement = {&reference, std::is_same_v<T, float> ?
            velocitypp::precision::replace_reference32 : velocitypp::precision::replace_reference64};
        const auto check = velocitypp::precision::promote_initial(promotion, first, count);
        if (check.code != velocitypp::precision::PrecisionCode::ok) return false;
        auto fit = stages.fit(moments);
        stages.emit(moments, fit);
        first += count;
    }
    output.result = stages.finish();
    return true;
}

template<class T> bool promote_stochastic(PromotionData& data, const velocitypp::PreparedInput& input,
    velocitypp::StochasticOptions options, StochasticBuffers<T>& output, WorkspaceBuffers<T>& workspace,
    velocitypp::precision::Promotion& promotion) {
    if (!workspace.bind(input, 1, true)) return false;
    velocitypp::BasicStochasticReferenceStages<T> stages{};
    if (!velocitypp::bind_stages(input, workspace.workspace, options, output.result, stages)) return false;
    if (velocitypp::precision::initialize_promotion(promotion).code != velocitypp::precision::PrecisionCode::ok) return false;
    for (std::size_t first = 0; first < input.genes;) {
        const std::size_t count = input.genes - first < workspace.workspace.block ? input.genes - first : workspace.workspace.block;
        auto moments = stages.initial_moments(first, count);
        velocitypp::precision::ReferenceMoments<T> reference{&input, &moments, nullptr};
        promotion.replacement = {&reference, std::is_same_v<T, float> ?
            velocitypp::precision::replace_reference32 : velocitypp::precision::replace_reference64};
        if (velocitypp::precision::promote_initial(promotion, first, count).code != velocitypp::precision::PrecisionCode::ok) return false;
        auto fit = stages.initial_fit(moments);
        stages.accept_initialization(moments, fit);
        first += count;
    }
    stages.resolve_selection();
    for (std::size_t first = 0; first < input.genes;) {
        const std::size_t count = input.genes - first < workspace.workspace.block ? input.genes - first : workspace.workspace.block;
        auto moments = stages.moments(first, count);
        velocitypp::precision::ReferenceMoments<T> reference{&input, nullptr, &moments};
        promotion.replacement.context = &reference;
        const auto check = velocitypp::precision::promote_second(promotion, first, count);
        if (check.code != velocitypp::precision::PrecisionCode::ok) return false;
        auto scales = stages.scales(moments);
        auto fit = stages.fit(moments, scales);
        stages.emit(moments, fit);
        first += count;
    }
    output.result = stages.finish();
    return true;
}

void test_sensitivity_validation_capacity() {
    constexpr std::size_t rows = 5;
    const double x[rows]{1, 2, 3, 4, 5}, y[rows]{2, 4, 6, 8, 10}, scratch[rows]{};
    velocitypp::FitOptions fit;
    const auto too_small = velocitypp::precision::validate_sensitivity(rows, 1, x, y, scratch, rows - 1, fit);
    expect(too_small.code == velocitypp::precision::PrecisionCode::insufficient_capacity && too_small.required == rows,
           "sensitivity validation reports exact caller-owned score capacity");
    const auto enough = velocitypp::precision::validate_sensitivity(rows, 1, x, y, scratch, rows, fit);
    expect(enough.code == velocitypp::precision::PrecisionCode::ok,
           "sensitivity validation accepts correctly sized caller-owned buffers");
}

void test_steady_selective_promotion() {
    PromotionData data;
    expect(data.ready, "caller-owned promotion fixture prepares dense source and baseline inputs");
    if (!data.ready) return;
    auto fit = data.fit_options();
    FitBuffers<float> full(PromotionData::g), baseline(PromotionData::g), selected(PromotionData::g);
    expect(run_steady(data.full_prepared, fit, full) && run_steady(data.baseline_prepared, fit, baseline),
           "reference runs produce full-precision-source and quantized baseline diagnostics");

    SourceContext context{&data};
    const std::uint8_t requested[PromotionData::g]{1, 0, 1};
    std::uint8_t status[PromotionData::g]{};
    velocitypp::precision::Promotion promotion{data.gene_order.data(), PromotionData::g, PromotionData::g,
        PromotionData::n, requested, PromotionData::g, status, PromotionData::g, 0,
        {&context, SourceContext::acquire}, {}};
    WorkspaceBuffers<float> workspace;
    expect(promote_steady(data, data.baseline_prepared, selected, workspace, promotion),
           "explicit first-pass promotion composes caller-bound reference moments and raw sources");
    expect(promotion.promoted == 1 &&
           status[0] == std::uint8_t(velocitypp::precision::PromotionStatus::promoted) &&
           status[1] == std::uint8_t(velocitypp::precision::PromotionStatus::unavailable) &&
           status[2] == std::uint8_t(velocitypp::precision::PromotionStatus::not_requested),
           "source-gene requests map through the prepared permutation and retain per-gene promotion states");
    bool diagnostics = fit_matches_at(selected, full, 0, 2e-6) &&
        fit_matches_at(selected, baseline, 1, 2e-6) && fit_matches_at(selected, baseline, 2, 2e-6);
    expect(diagnostics,
           "promoted mixed-dtype source matches full-input diagnostics while unavailable and unrequested genes retain baseline results");
    expect(context.calls[2] == 1 && context.calls[0] == 1 && context.calls[1] == 0,
           "steady promotion acquires only requested source genes");
}

void test_stochastic_selective_promotion() {
    PromotionData data;
    expect(data.ready, "caller-owned promotion fixture prepares dense source and baseline inputs");
    if (!data.ready) return;
    velocitypp::StochasticOptions options;
    options.initialization = data.fit_options();
    double original_xy = 0, original_xx = 0;
    for (std::size_t i = 0; i < PromotionData::n; ++i) {
        const double x = data.full_s[i * PromotionData::g + 2];
        const double y = data.full_u[i * PromotionData::g + 2];
        original_xy += x * y;
        original_xx += x * x;
    }
    const double original_gamma = original_xy / original_xx;
    expect(original_gamma > 2.0, "FP64 original fixture retains a slope above its quantized-FP32 baseline");
    options.initialization.min_ratio = (2.0 + original_gamma) / 2.0;

    StochasticBuffers<float> full(PromotionData::g), baseline(PromotionData::g), selected(PromotionData::g);
    expect(run_stochastic(data.full_prepared, options, full) &&
           run_stochastic(data.baseline_prepared, options, baseline),
           "stochastic reference runs establish promoted-source and baseline outcomes");

    SourceContext context{&data};
    const std::uint8_t requested[PromotionData::g]{1, 0, 1};
    std::uint8_t status[PromotionData::g]{};
    velocitypp::precision::Promotion promotion{data.gene_order.data(), PromotionData::g, PromotionData::g,
        PromotionData::n, requested, PromotionData::g, status, PromotionData::g, 0,
        {&context, SourceContext::acquire}, {}};
    WorkspaceBuffers<float> workspace;
    expect(promote_stochastic(data, data.baseline_prepared, options, selected, workspace, promotion),
           "stochastic promotion applies original columns before initialization selection and second moments");
    const bool mask_ok = promotion.promoted == 1 &&
        status[0] == std::uint8_t(velocitypp::precision::PromotionStatus::promoted) &&
        status[1] == std::uint8_t(velocitypp::precision::PromotionStatus::unavailable) &&
        status[2] == std::uint8_t(velocitypp::precision::PromotionStatus::not_requested);
    bool outcomes = mask_ok && selected.initialization.selected[0] && full.initialization.selected[0] &&
        !baseline.initialization.selected[0] && selected.status[0] != std::uint8_t(velocitypp::StochasticStatus::not_selected);
    for (std::size_t j = 0; j < PromotionData::g; ++j) {
        const auto& expected = j == 0 ? full : baseline;
        outcomes = outcomes && close(selected.gamma[j], expected.gamma[j], 2e-5) &&
            close(selected.offset[j], expected.offset[j], 2e-5) &&
            close(selected.offset2[j], expected.offset2[j], 2e-5) &&
            close(selected.first_scale[j], expected.first_scale[j], 2e-5) &&
            close(selected.second_scale[j], expected.second_scale[j], 2e-5) &&
            close(selected.objective[j], expected.objective[j], 2e-5) &&
            selected.status[j] == expected.status[j] &&
            selected.initialization.selected[j] == expected.initialization.selected[j] &&
            selected.initialization.status[j] == expected.initialization.status[j];
    }
    expect(outcomes,
           "stochastic promoted outputs match original-source refinement while remaining genes match quantized baseline");
    expect(context.calls[2] == 2 && context.calls[0] == 1 && context.calls[1] == 0,
           "stochastic promotion reacquires an original column for the second-moment pass");

    SourceContext disappearing{&data};
    disappearing.fail_gene2_after_first = true;
    StochasticBuffers<float> failed(PromotionData::g);
    WorkspaceBuffers<float> failed_workspace;
    std::uint8_t failed_status[PromotionData::g]{};
    velocitypp::precision::Promotion failing{data.gene_order.data(), PromotionData::g, PromotionData::g,
        PromotionData::n, requested, PromotionData::g, failed_status, PromotionData::g, 0,
        {&disappearing, SourceContext::acquire}, {}};
    const bool first_pass = [&] {
        if (!failed_workspace.bind(data.baseline_prepared, 1, true)) return false;
        velocitypp::BasicStochasticReferenceStages<float> stages{};
        if (!velocitypp::bind_stages(data.baseline_prepared, failed_workspace.workspace, options, failed.result, stages)) return false;
        if (velocitypp::precision::initialize_promotion(failing).code != velocitypp::precision::PrecisionCode::ok) return false;
        auto moments = stages.initial_moments(0, 1);
        velocitypp::precision::ReferenceMoments<float> reference{&data.baseline_prepared, &moments, nullptr};
        failing.replacement = {&reference, velocitypp::precision::replace_reference32};
        if (velocitypp::precision::promote_initial(failing, 0, 1).code != velocitypp::precision::PrecisionCode::ok) return false;
        auto fit = stages.initial_fit(moments);
        stages.accept_initialization(moments, fit);
        return failing.promoted == 1;
    }();
    velocitypp::precision::ReferenceMoments<float> second_reference{};
    velocitypp::BasicStochasticMomentBlock<float> second_block{};
    second_reference.input = &data.baseline_prepared;
    second_reference.second = &second_block;
    failing.replacement = {&second_reference, velocitypp::precision::replace_reference32};
    auto unavailable = velocitypp::precision::promote_second(failing, 0, 1);
    expect(first_pass && unavailable.code == velocitypp::precision::PrecisionCode::unavailable &&
           unavailable.location == 0 && disappearing.calls[2] == 2,
           "source disappearance before stochastic refinement reports an explicit unavailable status at its prepared position");
}

void test_promotion_without_requests_and_missing_source() {
    PromotionData data;
    expect(data.ready, "caller-owned promotion fixture prepares dense inputs");
    if (!data.ready) return;
    FitBuffers<float> result(PromotionData::g);
    WorkspaceBuffers<float> workspace;
    SourceContext no_request{&data};
    std::uint8_t status[PromotionData::g]{};
    velocitypp::precision::Promotion none{data.gene_order.data(), PromotionData::g, PromotionData::g,
        PromotionData::n, nullptr, 0, status, PromotionData::g, 0, {&no_request, SourceContext::acquire}, {}};
    expect(promote_steady(data, data.baseline_prepared, result, workspace, none) && none.promoted == 0 &&
           no_request.calls[0] + no_request.calls[1] + no_request.calls[2] == 0,
           "an absent source-gene mask executes the reference path without provider calls");

    const std::uint8_t zero[PromotionData::g]{};
    velocitypp::precision::Promotion all_zero{data.gene_order.data(), PromotionData::g, PromotionData::g,
        PromotionData::n, zero, PromotionData::g, status, PromotionData::g, 0, {&no_request, SourceContext::acquire}, {}};
    expect(promote_steady(data, data.baseline_prepared, result, workspace, all_zero) && all_zero.promoted == 0 &&
           no_request.calls[0] + no_request.calls[1] + no_request.calls[2] == 0,
           "an all-zero request mask bypasses original-column acquisition");

    const std::uint8_t requested[PromotionData::g]{0, 0, 1};
    velocitypp::precision::Promotion missing{data.gene_order.data(), PromotionData::g, PromotionData::g,
        PromotionData::n, requested, PromotionData::g, status, PromotionData::g, 0, {}, {}};
    expect(promote_steady(data, data.baseline_prepared, result, workspace, missing) && missing.promoted == 0 &&
           status[0] == std::uint8_t(velocitypp::precision::PromotionStatus::unavailable),
           "requested but missing original source retains the reference result with unavailable status");
}

}

int main() {
    test_analytic_screen<double>();
    test_analytic_screen<float>();
    test_constant_cutoff_and_invalid_inputs<double>();
    test_constant_cutoff_and_invalid_inputs<float>();
    test_sensitivity_validation_capacity();
    test_steady_selective_promotion();
    test_stochastic_selective_promotion();
    test_promotion_without_requests_and_missing_source();
    if (!failures) std::cout << "precision tests passed\n";
    return failures ? 1 : 0;
}
