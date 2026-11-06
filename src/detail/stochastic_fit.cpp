module;
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include "scientific.hh"

module velocitypp;
import anndatapp.core;
#include "stochastic_fit.hh"
#include "scientific_adapters.hh"

namespace velocitypp::detail {
namespace {
template<class T> BasicStochasticScales<T> scales_impl(const T* ms, const T* mu,
    const T* mss, const T* mus, std::size_t rows, std::ptrdiff_t stride,
    const BasicGeneFit<T>& initial, const StochasticOptions& options, const std::uint8_t* mask) {
    auto r = scientific::scales(ms, mu, mss, mus, rows, stride, numeric_fit(initial), numeric_options<T>(options), mask);
    return {r.first, r.second, StochasticStatus(r.status)};
}
template<class T> BasicStochasticGeneFit<T> fit_impl(const T* ms, const T* mu,
    const T* mss, const T* mus, std::size_t rows, std::ptrdiff_t stride,
    const BasicGeneFit<T>& initial, BasicStochasticScales<T> scales,
    const StochasticOptions& options, T* scratch, const std::uint8_t* mask) {
    auto r = scientific::joint_fit(ms, mu, mss, mus, rows, stride, numeric_fit(initial), numeric_scales(scales), numeric_options<T>(options), scratch, mask, StandardSort{});
    return {r.gamma, r.offset, r.offset2, {r.scales.first, r.scales.second, StochasticStatus(r.scales.status)}, StochasticStatus(r.status), r.objective};
}
}
#define VELOCITYPP_STOCHASTIC_WRAPPERS(T) \
BasicStochasticScales<T> stochastic_scales(const T* ms, const T* mu, const T* mss, const T* mus, std::size_t rows, std::ptrdiff_t stride, const BasicGeneFit<T>& initial, const StochasticOptions& options, const std::uint8_t* mask) { return scales_impl(ms,mu,mss,mus,rows,stride,initial,options,mask); } \
BasicStochasticGeneFit<T> stochastic_fit_gene(const T* ms, const T* mu, const T* mss, const T* mus, std::size_t rows, std::ptrdiff_t stride, const BasicGeneFit<T>& initial, BasicStochasticScales<T> scales, const StochasticOptions& options, T* scratch, const std::uint8_t* mask) { return fit_impl(ms,mu,mss,mus,rows,stride,initial,scales,options,scratch,mask); }
VELOCITYPP_STOCHASTIC_WRAPPERS(float)
VELOCITYPP_STOCHASTIC_WRAPPERS(double)
#undef VELOCITYPP_STOCHASTIC_WRAPPERS
}
