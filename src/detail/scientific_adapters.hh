#pragma once
// Include scientific.hh in the global fragment, then this header after importing
// velocity. Conversion occurs outside the element loops.
namespace velocity::detail {
template<class T> scientific::Options<T> numeric_options(const FitOptions& o) {
    return {scientific::Trimming(o.trimming), T(o.percentile_low), T(o.percentile_high),
            o.fit_intercept, o.positive_intercept, o.r2_adjusted,
            T(o.ratio_min), T(o.ratio_max), T(o.min_r2), T(o.min_ratio)};
}
template<class T> scientific::StochasticOptions<T> numeric_options(const StochasticOptions& o) {
    return {numeric_options<T>(o.initialization), o.fit_second_intercept};
}
template<class T> scientific::Fit<T> numeric_fit(const BasicGeneFit<T>& f) {
    return {f.gamma, f.offset, f.r2, scientific::FitStatus(f.status), f.selected};
}
template<class T> scientific::Scales<T> numeric_scales(const BasicStochasticScales<T>& f) {
    return {f.first, f.second, scientific::StochasticStatus(f.status)};
}
struct StandardSort {
    template<class T> void operator()(T* p, std::size_t n) const { std::sort(p, p + n); }
};
}
