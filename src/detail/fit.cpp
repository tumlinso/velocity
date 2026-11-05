module;
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "scientific.hh"

module velocity;
import anndatapp.core;
#include "fit.hh"
#include "scientific_adapters.hh"

namespace velocity::detail {
namespace {
bool valid_options(const FitOptions& o) {
    if (std::isnan(o.ratio_min) || std::isnan(o.ratio_max) || o.ratio_min > o.ratio_max ||
        !std::isfinite(o.min_r2) || !std::isfinite(o.min_ratio) ||
        !std::isfinite(o.percentile_low) || !std::isfinite(o.percentile_high) ||
        o.percentile_low < 0 || o.percentile_high > 100 || o.percentile_low > o.percentile_high) return false;
    return o.trimming == Trimming::none || o.trimming == Trimming::upper || o.trimming == Trimming::tails;
}
template<class T> BasicGeneFit<T> fit_gene_impl(const T* x, const T* y,
    std::size_t rows, std::ptrdiff_t stride, const FitOptions& o, T* scratch,
    const std::uint8_t* mask, bool enabled) {
    if (!valid_options(o)) throw std::invalid_argument("invalid fit options");
    if (rows) {
        if (!x || !y || !stride) throw std::invalid_argument("invalid fit buffers or stride");
        if (o.trimming != Trimming::none && !scratch) throw std::invalid_argument("trimmed fit requires score scratch");
        auto magnitude = stride > 0 ? std::uintmax_t(stride) : std::uintmax_t(-(stride + 1)) + 1;
        if (rows > 1 && std::uintmax_t(rows - 1) > std::uintmax_t(std::numeric_limits<std::ptrdiff_t>::max())/magnitude)
            throw std::invalid_argument("fit stride overflows row offsets");
    }
    auto f = scientific::fit_gene(x, y, rows, stride, numeric_options<T>(o), scratch, mask, enabled, StandardSort{});
    return {f.gamma, f.offset, f.r2, FitStatus(f.status), f.selected};
}
}
GeneFit fit_gene(const double* x, const double* y, std::size_t n, std::ptrdiff_t stride,
    const FitOptions& o, double* scratch, const std::uint8_t* mask, bool enabled) {
    return fit_gene_impl(x, y, n, stride, o, scratch, mask, enabled);
}
GeneFit32 fit_gene(const float* x, const float* y, std::size_t n, std::ptrdiff_t stride,
    const FitOptions& o, float* scratch, const std::uint8_t* mask, bool enabled) {
    return fit_gene_impl(x, y, n, stride, o, scratch, mask, enabled);
}
}
namespace velocity::reference {
GeneFit fit_gene(const double* x, const double* y, std::size_t n, std::ptrdiff_t stride,
    const FitOptions& o, double* scratch, const std::uint8_t* mask, bool enabled) {
    return detail::fit_gene(x, y, n, stride, o, scratch, mask, enabled);
}
GeneFit32 fit_gene(const float* x, const float* y, std::size_t n, std::ptrdiff_t stride,
    const FitOptions& o, float* scratch, const std::uint8_t* mask, bool enabled) {
    return detail::fit_gene(x, y, n, stride, o, scratch, mask, enabled);
}
}
