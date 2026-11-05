#pragma once

// Diagnostic arithmetic for the portable CPU implementation.  The
// caller supplies one score buffer of at least `rows` elements; no storage is
// allocated here.  Values are evaluated in the selected compute precision.
#include "scientific.hh"

namespace velocity::detail::sensitivity {

enum class State : std::uint8_t { no_trimming, unscoreable, stable, sensitive };

template<class T> struct Options {
    T band = T(.5), prediction_limit = T(.1), r2_limit = T(.05);
};

template<class T> struct Report {
    State state = State::unscoreable;
    T low = 0, high = 0, delta = 0, prediction_change = 0, r2_change = 0;
    scientific::FitStatus expanded_status = scientific::FitStatus::insufficient_data;
    scientific::FitStatus contracted_status = scientific::FitStatus::insufficient_data;
    bool status_change = false, selection_change = false;
};

// `base` is the already computed initialization fit.  `scratch` and the input
// columns must be disjoint.  Prediction norms include all rows, like R²; the
// supplied fit mask applies to percentile selection and the regression itself.
template<class T, class Sorter> inline Report<T> report(
    const T* x, const T* y, std::size_t rows, std::ptrdiff_t stride,
    const scientific::Fit<T>& base, const scientific::Options<T>& fit,
    T* scratch, const std::uint8_t* mask, bool enabled,
    Options<T> opt, Sorter sort) {
    using namespace scientific;
    Report<T> out;
    out.expanded_status = out.contracted_status = base.status;
    if (fit.trimming == Trimming::none) { out.state = State::no_trimming; return out; }
    if (!rows || !x || !y || !stride || !scratch || base.status != FitStatus::ok ||
        !isfinite(base.gamma) || !isfinite(base.offset) || !isfinite(base.r2) ||
        !isfinite(opt.band) || !isfinite(opt.prediction_limit) || !isfinite(opt.r2_limit) ||
        opt.band < 0 || opt.prediction_limit < 0 || opt.r2_limit < 0) return out;

    const bool lower = fit.trimming == Trimming::tails && fit.fit_intercept;
    T max_x = -std::numeric_limits<T>::infinity(), max_y = max_x;
    std::size_t n = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        const auto k = std::ptrdiff_t(i)*stride;
        if (!isfinite(x[k]) || !isfinite(y[k])) return out;
        if (!included(i, mask)) continue;
        max_x = maximum(max_x, x[k]); max_y = maximum(max_y, y[k]); ++n;
    }
    if (!n) return out;
    std::size_t k = 0;
    for (std::size_t i = 0; i < rows; ++i) if (included(i, mask)) {
        const auto p = std::ptrdiff_t(i)*stride;
        scratch[k++] = score(x[p], y[p], max_x, max_y);
    }
    for (std::size_t i = 0; i < n; ++i) if (!isfinite(scratch[i])) return out;
    sort(scratch, n);
    out.low = lower ? percentile(scratch, n, fit.percentile_low) : T(0);
    out.high = percentile(scratch, n, fit.percentile_high);
    T magnitude = maximum(T(1), absolute(out.high));
    if (lower) magnitude = maximum(magnitude, absolute(out.low));
    out.delta = opt.band*T(1.0/2048.0)*magnitude;
    if (!isfinite(out.delta)) return out;

    const TrimBounds<T> expanded{out.low + out.delta, out.high - out.delta};
    const TrimBounds<T> contracted{out.low - out.delta, out.high + out.delta};
    const auto a = fit_gene(x, y, rows, stride, fit, scratch, mask, enabled, sort, &expanded);
    const auto b = fit_gene(x, y, rows, stride, fit, scratch, mask, enabled, sort, &contracted);
    out.expanded_status = a.status; out.contracted_status = b.status;
    out.status_change = a.status != base.status || b.status != base.status;
    out.selection_change = a.selected != base.selected || b.selected != base.selected;

    T residual2 = 0, response2 = 0, prediction2 = 0, change_a2 = 0, change_b2 = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        const auto p = std::ptrdiff_t(i)*stride;
        T prediction = base.gamma*x[p] + base.offset, residual = y[p] - prediction;
        residual2 += residual*residual; response2 += y[p]*y[p]; prediction2 += prediction*prediction;
        if (a.status == FitStatus::ok) { T d = a.gamma*x[p] + a.offset - prediction; change_a2 += d*d; }
        if (b.status == FitStatus::ok) { T d = b.gamma*x[p] + b.offset - prediction; change_b2 += d*d; }
    }
    if (!isfinite(residual2) || !isfinite(response2) || !isfinite(prediction2) ||
        !isfinite(change_a2) || !isfinite(change_b2)) return out;
    T denominator = maximum(square_root(residual2), T(32)*std::numeric_limits<T>::epsilon()*
        maximum(square_root(response2), square_root(prediction2)));
    if (!(denominator > 0) || !isfinite(denominator)) return out;
    out.prediction_change = square_root(maximum(change_a2, change_b2))/denominator;
    if (a.status == FitStatus::ok) out.r2_change = maximum(out.r2_change, absolute(a.r2 - base.r2));
    if (b.status == FitStatus::ok) out.r2_change = maximum(out.r2_change, absolute(b.r2 - base.r2));
    if (!isfinite(out.prediction_change) || !isfinite(out.r2_change)) return out;
    out.state = out.status_change || out.selection_change || out.prediction_change > opt.prediction_limit ||
        out.r2_change > opt.r2_limit ? State::sensitive : State::stable;
    return out;
}

}
