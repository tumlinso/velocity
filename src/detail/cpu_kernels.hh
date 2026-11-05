#pragma once
#include <cstddef>
#include <cstdint>

// Compiled portable CPU implementation ABI. Public BMIs and scalar reference
// implementations never include implementation headers; column_sums retains
// ascending row order separately for each gene.
namespace velocity::detail::cpu {
void half_to_float(const std::uint16_t* source, std::ptrdiff_t stride_bytes,
                   float* output, std::size_t count) noexcept;
void weighted_add(const float* source, float weight, float* output, std::size_t count) noexcept;
void weighted_add(const double* source, double weight, double* output, std::size_t count) noexcept;
void products(const float* s, const float* u, float* ss, float* su, float* uu, std::size_t count) noexcept;
void products(const double* s, const double* u, double* ss, double* su, double* uu, std::size_t count) noexcept;
void residuals(const float* s, const float* u, const float* gamma, const float* offset,
               float* output, std::size_t rows, std::size_t genes) noexcept;
void residuals(const double* s, const double* u, const double* gamma, const double* offset,
               double* output, std::size_t rows, std::size_t genes) noexcept;
void column_sums(const float* values, std::size_t rows, std::size_t genes,
                 std::ptrdiff_t row_stride, float* output) noexcept;
void column_sums(const double* values, std::size_t rows, std::size_t genes,
                 std::ptrdiff_t row_stride, double* output) noexcept;
}
