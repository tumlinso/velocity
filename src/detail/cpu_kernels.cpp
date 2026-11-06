#include "cpu_kernels.hh"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace velocitypp::detail::cpu {
namespace {

template<class T> void weighted_add_scalar(const T* source, T weight, T* output, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) output[i] = output[i] + weight*source[i];
}

template<bool SS, bool SU, bool UU, class T>
void products_scalar_run(const T* s, const T* u, T* ss, T* su, T* uu, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        if constexpr (SS) ss[i] = s[i]*s[i];
        if constexpr (SU) su[i] = s[i]*u[i];
        if constexpr (UU) uu[i] = u[i]*u[i];
    }
}

template<class T> void products_scalar(const T* s, const T* u, T* ss, T* su, T* uu,
                                        std::size_t count) noexcept {
    if (ss) {
        if (su) { if (uu) products_scalar_run<true,true,true>(s,u,ss,su,uu,count);
                  else products_scalar_run<true,true,false>(s,u,ss,su,uu,count); }
        else { if (uu) products_scalar_run<true,false,true>(s,u,ss,su,uu,count);
               else products_scalar_run<true,false,false>(s,u,ss,su,uu,count); }
    } else if (su) {
        if (uu) products_scalar_run<false,true,true>(s,u,ss,su,uu,count);
        else products_scalar_run<false,true,false>(s,u,ss,su,uu,count);
    } else if (uu) products_scalar_run<false,false,true>(s,u,ss,su,uu,count);
}

template<class T> void residuals_scalar(const T* s, const T* u, const T* gamma, const T* offset,
                                         T* output, std::size_t rows, std::size_t genes) noexcept {
    for (std::size_t i = 0; i < rows; ++i) for (std::size_t j = 0; j < genes; ++j)
        output[i*genes + j] = u[i*genes + j] - gamma[j]*s[i*genes + j] - offset[j];
}

template<class T> void column_sums_scalar(const T* values, std::size_t rows, std::size_t genes,
                                          std::ptrdiff_t row_stride, T* output) noexcept {
    for (std::size_t j = 0; j < genes; ++j) {
        T sum = 0;
        for (std::size_t i = 0; i < rows; ++i) sum += values[std::ptrdiff_t(i)*row_stride + std::ptrdiff_t(j)];
        output[j] = sum;
    }
}

}


void weighted_add(const float* source, float weight, float* output, std::size_t count) noexcept {
    weighted_add_scalar(source, weight, output, count);
}
void weighted_add(const double* source, double weight, double* output, std::size_t count) noexcept {
    weighted_add_scalar(source, weight, output, count);
}

void products(const float* s, const float* u, float* ss, float* su, float* uu, std::size_t count) noexcept {
    products_scalar(s, u, ss, su, uu, count);
}
void products(const double* s, const double* u, double* ss, double* su, double* uu, std::size_t count) noexcept {
    products_scalar(s, u, ss, su, uu, count);
}

void residuals(const float* s, const float* u, const float* gamma, const float* offset,
               float* output, std::size_t rows, std::size_t genes) noexcept {
    residuals_scalar(s, u, gamma, offset, output, rows, genes);
}
void residuals(const double* s, const double* u, const double* gamma, const double* offset,
               double* output, std::size_t rows, std::size_t genes) noexcept {
    residuals_scalar(s, u, gamma, offset, output, rows, genes);
}

void column_sums(const float* values, std::size_t rows, std::size_t genes,
                 std::ptrdiff_t row_stride, float* output) noexcept {
    column_sums_scalar(values, rows, genes, row_stride, output);
}
void column_sums(const double* values, std::size_t rows, std::size_t genes,
                 std::ptrdiff_t row_stride, double* output) noexcept {
    column_sums_scalar(values, rows, genes, row_stride, output);
}

}
