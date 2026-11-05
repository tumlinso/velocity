#include "../src/detail/cpu_kernels.hh"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<class T> std::uint64_t bits(T x) {
    if constexpr (std::is_same_v<T, float>) return std::bit_cast<std::uint32_t>(x);
    else return std::bit_cast<std::uint64_t>(x);
}

template<class T> void same(T actual, T expected, const char* message) {
    require(bits(actual) == bits(expected), message);
}

template<class T> T value(std::size_t i, unsigned salt = 0) {
    const int k = int((i*17 + salt*11) % 31) - 15;
    if constexpr (std::is_same_v<T, float>) return float(k)*0.125f + float(salt)*0.03125f;
    else return double(k)*0.125 + double(salt)*0.03125;
}

template<class T> void weighted_add_case() {
    const std::size_t lanes = std::is_same_v<T, float> ? 8 : 4;
    for (auto n : {std::size_t(0), std::size_t(1), lanes-1, lanes, lanes+1, 3*lanes+2}) {
        std::vector<T> src(n+3), dst(n+4), expected(n+4);
        T* x = src.data() + 1;
        T* y = dst.data() + 2;
        T* e = expected.data() + 2;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = i%2 ? T(1.00000011920928955078125) : T(1.0e10);
            y[i] = e[i] = i%2 ? T(-1.0e10) : T(0.33333333333333331);
        }
        const T w = T(0.33333333333333331);
        for (std::size_t i = 0; i < n; ++i) {
            volatile T product = w*x[i];
            e[i] = e[i] + product;
        }
        velocity::detail::cpu::weighted_add(x, w, y, n);
        for (std::size_t i = 0; i < n; ++i) same(y[i], e[i], "weighted_add multiply-then-add parity");
    }
    std::vector<T> x(lanes+1,T(1)-std::numeric_limits<T>::epsilon());
    std::vector<T> y(lanes+1,T(-1));
    const T w = T(1)+std::numeric_limits<T>::epsilon();
    T* xp = x.data()+1; T* yp = y.data()+1;
    for (std::size_t i = 0; i < lanes; ++i) {
        volatile T product = w*xp[i];
        same(yp[i] + product,T(0),"weighted_add FMA witness setup");
    }
    velocity::detail::cpu::weighted_add(xp,w,yp,lanes);
    for (std::size_t i = 0; i < lanes; ++i) same(yp[i],T(0),"weighted_add does not fuse multiply and add");
}

template<class T> void products_case() {
    const std::size_t lanes = std::is_same_v<T, float> ? 8 : 4;
    for (auto n : {std::size_t(0), std::size_t(1), lanes-1, lanes, lanes+1, 2*lanes+3}) {
        std::vector<T> sv(n+3), uv(n+4), ss(n+5), su(n+6), uu(n+7);
        T* s = sv.data() + 1; T* u = uv.data() + 2;
        T* a = ss.data() + 2; T* b = su.data() + 3; T* c = uu.data() + 4;
        for (std::size_t i = 0; i < n; ++i) { s[i] = value<T>(i, 1); u[i] = value<T>(i, 3); }
        velocity::detail::cpu::products(s, u, a, b, c, n);
        for (std::size_t i = 0; i < n; ++i) {
            same(a[i], s[i]*s[i], "products ss"); same(b[i], s[i]*u[i], "products su");
            same(c[i], u[i]*u[i], "products uu");
        }
        if (n) {
            velocity::detail::cpu::products(s, nullptr, a, nullptr, nullptr, n);
            velocity::detail::cpu::products(nullptr, u, nullptr, nullptr, c, n);
            velocity::detail::cpu::products(s, u, nullptr, b, nullptr, n);
            for (std::size_t i = 0; i < n; ++i) {
                same(a[i], s[i]*s[i], "products optional ss"); same(c[i], u[i]*u[i], "products optional uu");
                same(b[i], s[i]*u[i], "products optional su");
            }
        }
    }
}

template<class T> void residuals_case() {
    const std::size_t lanes = std::is_same_v<T, float> ? 8 : 4;
    for (auto genes : {std::size_t(0), std::size_t(1), lanes-1, lanes, lanes+1, std::size_t(13)}) {
        const std::size_t rows = 3, n = rows*genes;
        std::vector<T> sv(n+3), uv(n+4), out(n+5), expected(n+5), gv(genes+2), av(genes+3);
        const T* s = sv.data()+1; const T* u = uv.data()+2;
        T* v = out.data()+3; T* e = expected.data()+3;
        const T* g = gv.data()+1; const T* a = av.data()+2;
        for (std::size_t j = 0; j < genes; ++j) { gv[j+1]=value<T>(j,4); av[j+2]=value<T>(j,7); }
        for (std::size_t i = 0; i < n; ++i) { sv[i+1]=value<T>(i,2); uv[i+2]=value<T>(i,9); }
        for (std::size_t i = 0; i < rows; ++i) for (std::size_t j = 0; j < genes; ++j) {
            volatile T product = g[j]*s[i*genes+j];
            e[i*genes+j] = (u[i*genes+j]-product)-a[j];
        }
        velocity::detail::cpu::residuals(s,u,g,a,v,rows,genes);
        for (std::size_t i = 0; i < n; ++i) same(v[i],e[i],"residuals multiply-subtract order");
    }
    velocity::detail::cpu::residuals(static_cast<const T*>(nullptr),static_cast<const T*>(nullptr),
        static_cast<const T*>(nullptr),static_cast<const T*>(nullptr),static_cast<T*>(nullptr),7,0);
    std::vector<T> s(lanes+1,T(1)-std::numeric_limits<T>::epsilon());
    std::vector<T> u(lanes+1,T(0)), g(lanes+1,T(1)+std::numeric_limits<T>::epsilon());
    std::vector<T> a(lanes+1,T(-1)), out(lanes+1,T(1));
    velocity::detail::cpu::residuals(s.data()+1,u.data()+1,g.data()+1,a.data()+1,out.data()+1,1,lanes);
    for (std::size_t j = 0; j < lanes; ++j) same(out[j+1],T(0),"residuals do not fuse multiply and subtract");
}

template<class T> void column_sums_case() {
    const std::size_t lanes = std::is_same_v<T, float> ? 8 : 4;
    for (auto genes : {std::size_t(0), std::size_t(1), lanes-1, lanes, lanes+1, std::size_t(13)}) {
        constexpr std::size_t rows = 4;
        const std::size_t stride = genes + 3;
        for (bool reverse : {false,true}) {
            std::vector<T> data(rows*stride+2), output(genes+3), expected(genes+3);
            T* logical0 = data.data()+1+(reverse ? (rows-1)*stride : 0);
            const auto row_stride = reverse ? -std::ptrdiff_t(stride) : std::ptrdiff_t(stride);
            for (std::size_t i = 0; i < rows; ++i) for (std::size_t j = 0; j < genes; ++j) {
                T x;
                if (j == 0) x = i == 0 ? T(1.0e20) : i == 1 ? T(1) : i == 2 ? T(-1.0e20) : T(1);
                else x = value<T>(i*genes+j, unsigned(j));
                logical0[std::ptrdiff_t(i)*row_stride+std::ptrdiff_t(j)] = x;
            }
            T* out = output.data()+2; T* ref = expected.data()+2;
            for (std::size_t j = 0; j < genes; ++j) {
                T sum = 0;
                for (std::size_t i = 0; i < rows; ++i) sum += logical0[std::ptrdiff_t(i)*row_stride+std::ptrdiff_t(j)];
                ref[j] = sum;
            }
            velocity::detail::cpu::column_sums(logical0,rows,genes,row_stride,out);
            for (std::size_t j = 0; j < genes; ++j) same(out[j],ref[j],"column_sums ascending row order and signed stride");
        }
        std::vector<T> zeros(genes+1,T(1));
        velocity::detail::cpu::column_sums(static_cast<const T*>(nullptr),0,genes,
            std::ptrdiff_t(stride),zeros.data()+1);
        for (std::size_t j = 0; j < genes; ++j) same(zeros[j+1],T(0),"zero-row column_sums");
    }
}

float expected_half(std::uint16_t h) {
    const bool negative = (h & 0x8000) != 0;
    const unsigned exponent = (h >> 10) & 0x1f, fraction = h & 0x3ff;
    if (exponent == 31) {
        if (fraction) return std::bit_cast<float>(std::uint32_t(0x7fc00000));
        return std::bit_cast<float>((negative ? 0x80000000u : 0u) | 0x7f800000u);
    }
    float x = exponent ? std::ldexp(float(1024+fraction), int(exponent)-25)
                       : std::ldexp(float(fraction), -24);
    return std::copysign(x, negative ? -1.0f : 1.0f);
}

void half_case() {
    std::vector<std::uint16_t> h(65537);
    std::vector<float> out(65537);
    auto* source = h.data()+1; auto* result = out.data()+1;
    for (std::size_t i = 0; i < 65536; ++i) source[i] = std::uint16_t(i);
    velocity::detail::cpu::half_to_float(source,2,result,65536);
    for (std::size_t i = 0; i < 65536; ++i) same(result[i],expected_half(std::uint16_t(i)),"binary16 exhaustive conversion");

    constexpr std::size_t n = 19;
    std::vector<unsigned char> bytes(n*3+8);
    std::vector<float> strided(n+2);
    auto* p = bytes.data()+1;
    for (std::size_t i = 0; i < n; ++i) {
        auto x = std::uint16_t((i*4099u + 0x8001u) & 0xffffu);
        std::memcpy(p+3*i,&x,2);
    }
    velocity::detail::cpu::half_to_float(reinterpret_cast<const std::uint16_t*>(p),3,strided.data()+1,n);
    for (std::size_t i = 0; i < n; ++i) {
        std::uint16_t x; std::memcpy(&x,p+3*i,2);
        same(strided[i+1],expected_half(x),"binary16 positive byte stride");
    }
    velocity::detail::cpu::half_to_float(reinterpret_cast<const std::uint16_t*>(p+3*(n-1)),-3,strided.data()+1,n);
    for (std::size_t i = 0; i < n; ++i) {
        std::uint16_t x; std::memcpy(&x,p+3*(n-1-i),2);
        same(strided[i+1],expected_half(x),"binary16 negative byte stride");
    }
}

void run() {
    velocity::detail::cpu::weighted_add(static_cast<const float*>(nullptr),0.0f,static_cast<float*>(nullptr),0);
    velocity::detail::cpu::products(static_cast<const double*>(nullptr),static_cast<const double*>(nullptr),
        static_cast<double*>(nullptr),static_cast<double*>(nullptr),static_cast<double*>(nullptr),0);
    velocity::detail::cpu::half_to_float(nullptr,2,nullptr,0);
    half_case();
    weighted_add_case<float>(); weighted_add_case<double>();
    products_case<float>(); products_case<double>();
    residuals_case<float>(); residuals_case<double>();
    column_sums_case<float>(); column_sums_case<double>();
}
}

int main() {
    try { run(); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    std::cout << "CPU kernel checks passed\n";
}
