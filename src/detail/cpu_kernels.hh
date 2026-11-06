#pragma once
#include <stddef.h>

// Compiled portable CPU implementation ABI. Public BMIs and scalar reference
// implementations never include implementation headers; column_sums retains
// ascending row order separately for each gene.
namespace velocitypp::detail::cpu {
void weighted_add(const float* source, float weight, float* output, size_t count) noexcept;
void weighted_add(const double* source, double weight, double* output, size_t count) noexcept;
void products(const float* s, const float* u, float* ss, float* su, float* uu, size_t count) noexcept;
void products(const double* s, const double* u, double* ss, double* su, double* uu, size_t count) noexcept;
void residuals(const float* s, const float* u, const float* gamma, const float* offset,
               float* output, size_t rows, size_t genes) noexcept;
void residuals(const double* s, const double* u, const double* gamma, const double* offset,
               double* output, size_t rows, size_t genes) noexcept;
void column_sums(const float* values, size_t rows, size_t genes,
                 ptrdiff_t row_stride, float* output) noexcept;
void column_sums(const double* values, size_t rows, size_t genes,
                 ptrdiff_t row_stride, double* output) noexcept;
}
