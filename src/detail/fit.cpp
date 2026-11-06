module;
#include <stddef.h>
#include <stdint.h>
#include "scientific.hh"

module velocitypp;
import anndatapp.core;
#include "fit.hh"
#include "scientific_adapters.hh"

namespace velocitypp::detail {
namespace {
template<class T> BasicGeneFit<T> fit_gene_impl(const T* x, const T* y,
    size_t rows, ptrdiff_t stride, const FitOptions& o, T* scratch,
    const uint8_t* mask, bool enabled) {
    auto f = scientific::fit_gene(x, y, rows, stride, numeric_options<T>(o), scratch, mask, enabled,
                                  scientific::HeapSort{});
    return {f.gamma, f.offset, f.r2, FitStatus(f.status), f.selected};
}
}
GeneFit fit_gene(const double* x, const double* y, size_t n, ptrdiff_t stride,
    const FitOptions& o, double* scratch, const uint8_t* mask, bool enabled) {
    return fit_gene_impl(x, y, n, stride, o, scratch, mask, enabled);
}
GeneFit32 fit_gene(const float* x, const float* y, size_t n, ptrdiff_t stride,
    const FitOptions& o, float* scratch, const uint8_t* mask, bool enabled) {
    return fit_gene_impl(x, y, n, stride, o, scratch, mask, enabled);
}
}
namespace velocitypp::reference {
GeneFit fit_gene(const double* x, const double* y, size_t n, ptrdiff_t stride,
    const FitOptions& o, double* scratch, const uint8_t* mask, bool enabled) {
    return detail::fit_gene(x, y, n, stride, o, scratch, mask, enabled);
}
GeneFit32 fit_gene(const float* x, const float* y, size_t n, ptrdiff_t stride,
    const FitOptions& o, float* scratch, const uint8_t* mask, bool enabled) {
    return detail::fit_gene(x, y, n, stride, o, scratch, mask, enabled);
}
}
