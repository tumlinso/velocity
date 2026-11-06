#pragma once
// Canonical portable CPU scientific arithmetic.
// This private header has no module, anndata, allocation, or synchronization dependency.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace velocitypp::detail::scientific {
enum class FitStatus : std::uint8_t { ok, insufficient_data, no_signal, singular, constant_response };
enum class Trimming { none, upper, tails };
enum class StochasticStatus : std::uint8_t { ok, not_selected, invalid_initialization, insufficient_data, zero_scale, singular, nonfinite };
template<class T> struct Options {
    Trimming trimming = Trimming::tails;
    T percentile_low = 5, percentile_high = 95;
    bool fit_intercept = false, positive_intercept = true, r2_adjusted = false;
    T ratio_min = -std::numeric_limits<T>::infinity(), ratio_max = std::numeric_limits<T>::infinity();
    T min_r2 = T(.01), min_ratio = T(.01);
};
template<class T> struct StochasticOptions { Options<T> initialization; bool fit_second_intercept = false; };
template<class T> struct TrimBounds { T low = 0, high = 0; };
template<class T> struct Fit { T gamma = 0, offset = 0, r2 = 0; FitStatus status = FitStatus::insufficient_data; bool selected = false; };
template<class T> struct Scales { T first = 0, second = 0; StochasticStatus status = StochasticStatus::not_selected; };
template<class T> struct JointFit { T gamma = 0, offset = 0, offset2 = 0; Scales<T> scales; StochasticStatus status = StochasticStatus::not_selected; T objective = 0; };
template<class T> inline T minimum(T a, T b) { return b < a ? b : a; }
template<class T> inline T maximum(T a, T b) { return a < b ? b : a; }
template<class T> inline T clamp(T x, T lo, T hi) { return x < lo ? lo : hi < x ? hi : x; }
template<class T> inline T absolute(T x) { return x < 0 ? -x : x; }
template<class T> inline bool isfinite(T x) {
    return std::isfinite(x);
}
template<class T> inline T square_root(T x) { if constexpr (sizeof(T) == sizeof(float)) return ::sqrtf(x); else return ::sqrt(x); }
template<class T> inline T floor_value(T x) { if constexpr (sizeof(T) == sizeof(float)) return ::floorf(x); else return ::floor(x); }
template<class T> constexpr T rank_tolerance() { return sizeof(T) == sizeof(float) ? T(1e-6) : T(1e-12); }
template<class T> constexpr T residual_floor(T term_scale) { return sizeof(T) == sizeof(float) ? T(8)*std::numeric_limits<T>::epsilon()*term_scale : T(0); }
// Sorter contract: operator()(T* values, size_t count), finite values only.
struct HeapSort {
    template<class T> void operator()(T* a, std::size_t n) const {
        for (std::size_t start = n/2; start > 0; --start) sift(a, start-1, n);
        for (std::size_t end = n; end > 1; --end) { T t=a[0]; a[0]=a[end-1]; a[end-1]=t; sift(a,0,end-1); }
    }
    template<class T> static void sift(T* a, std::size_t root, std::size_t n) {
        while (root < n/2) { auto child=2*root+1; if (child+1<n && a[child]<a[child+1]) ++child; if (!(a[root]<a[child])) return; T t=a[root];a[root]=a[child];a[child]=t;root=child; }
    }
};


template<class T> inline Fit<T> failed(FitStatus status) {
    return {0, 0, 0, status, false};
}


inline bool included(std::size_t row, const std::uint8_t* fit_mask) {
    return !fit_mask || fit_mask[row] != 0;
}

template<class T> inline T score(T x, T y, T max_x, T max_y) {
    return x / maximum(max_x, T(1e-3)) + y / maximum(max_y, T(1e-3));
}

template<class T> inline T percentile(const T* sorted, std::size_t count, T q) {
    const T position = T(count - 1) * (q / T(100.0));
    const auto lower = static_cast<std::size_t>(floor_value(position));
    const auto upper = minimum(lower + 1, count - 1);
    const T fraction = position - T(lower);
    return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

template<class T> struct Sums {
    std::size_t count = 0;
    T x = 0, y = 0, xx = 0, xy = 0;
    T centered_xx = 0, centered_xy = 0;
    T minimum_x = std::numeric_limits<T>::infinity(), maximum_x = -std::numeric_limits<T>::infinity();
};

template<class T> inline bool finite_sums(const Sums<T>& s) {
    return isfinite(s.x) && isfinite(s.y) &&
           isfinite(s.xx) && isfinite(s.xy) &&
           isfinite(s.centered_xx) && isfinite(s.centered_xy);
}

template<class T> struct Regression {
    T gamma = 0, offset = 0;
    bool valid = false;
};

template<class T> inline Regression<T> regress(const Sums<T>& sums, bool fit_intercept, bool positive_intercept) {
    if (!sums.count || !finite_sums(sums)) return {};

    T gamma, offset = 0;
    if (fit_intercept) {
        if (sums.minimum_x == sums.maximum_x) return {};
        const T n = T(sums.count);
        const T mean_x = sums.x / n;
        const T mean_y = sums.y / n;
        const T variance_x = sums.centered_xx / n;
        const T covariance = sums.centered_xy / n;
        if (!(variance_x > 0) || !isfinite(variance_x) ||
            !isfinite(covariance)) return {};
        gamma = covariance / variance_x;
        offset = mean_y - gamma * mean_x;
        if (!isfinite(gamma) || !isfinite(offset)) return {};

        if (positive_intercept && offset < 0) {
            if (!(sums.xx > 0)) return {};
            gamma = sums.xy / sums.xx;
            offset = 0;
        }
    } else {
        if (!(sums.xx > 0)) return {};
        gamma = sums.xy / sums.xx;
    }

    if (!isfinite(gamma) || !isfinite(offset)) return {};
    return {gamma, offset, true};
}

template<class T> inline void accumulate(Sums<T>& sums, T x, T y) {
    ++sums.count;
    sums.minimum_x = minimum(sums.minimum_x, x); sums.maximum_x = maximum(sums.maximum_x, x);
    sums.x += x; sums.y += y; sums.xx += x * x; sums.xy += x * y;
}

template<class T> inline Sums<T> collect_all(const T* spliced, const T* unspliced, std::size_t rows,
                 std::ptrdiff_t stride, const std::uint8_t* fit_mask) {
    Sums<T> sums;
    for (std::size_t i = 0; i < rows; ++i) {
        if (!included(i, fit_mask)) continue;
        const auto offset = std::ptrdiff_t(i) * stride;
        accumulate(sums, spliced[offset], unspliced[offset]);
    }
    return sums;
}

template<class T> inline Sums<T> collect_upper(const T* spliced, const T* unspliced, std::size_t rows,
                   std::ptrdiff_t stride, const std::uint8_t* fit_mask,
                   T max_x, T max_y, T high) {
    Sums<T> sums;
    for (std::size_t i = 0; i < rows; ++i) {
        if (!included(i, fit_mask)) continue;
        const auto offset = std::ptrdiff_t(i) * stride;
        const T x = spliced[offset], y = unspliced[offset];
        if (score(x, y, max_x, max_y) < high) continue;
        accumulate(sums, x, y);
    }
    return sums;
}

template<class T> inline Sums<T> collect_tails(const T* spliced, const T* unspliced, std::size_t rows,
                   std::ptrdiff_t stride, const std::uint8_t* fit_mask,
                   T max_x, T max_y, T low, T high) {
    Sums<T> sums;
    for (std::size_t i = 0; i < rows; ++i) {
        if (!included(i, fit_mask)) continue;
        const auto offset = std::ptrdiff_t(i) * stride;
        const T x = spliced[offset], y = unspliced[offset];
        const T value = score(x, y, max_x, max_y);
        if (value > low && value < high) continue;
        accumulate(sums, x, y);
    }
    return sums;
}

template<class T, bool Trimmed, bool BothTails>
inline void center(Sums<T>& sums, const T* spliced, const T* unspliced,
            std::size_t rows, std::ptrdiff_t stride, const std::uint8_t* fit_mask,
            T max_x, T max_y, T low, T high) {
    // A second pass avoids losing small variance to E[x*x] - E[x]*E[x].
    const T mean_x = sums.x / T(sums.count);
    const T mean_y = sums.y / T(sums.count);
    for (std::size_t i = 0; i < rows; ++i) {
        if (!included(i, fit_mask)) continue;
        const auto offset = std::ptrdiff_t(i) * stride;
        const T x = spliced[offset], y = unspliced[offset];
        if constexpr (Trimmed) {
            const T value = score(x, y, max_x, max_y);
            if constexpr (BothTails) {
                if (value > low && value < high) continue;
            } else if (value < high) continue;
        }
        const T dx = x - mean_x, dy = y - mean_y;
        sums.centered_xx += dx * dx;
        sums.centered_xy += dx * dy;
    }
}

template<class T> inline FitStatus r_squared(const T* spliced, const T* unspliced,
                    std::size_t rows, std::ptrdiff_t stride,
                    const Regression<T>& model, T& result) {
    T mean_y = 0, min_y = unspliced[0], max_y = min_y;
    for (std::size_t i = 0; i < rows; ++i) {
        T y = unspliced[std::ptrdiff_t(i) * stride];
        mean_y += y; min_y = minimum(min_y, y); max_y = maximum(max_y, y);
    }
    if (min_y == max_y) return FitStatus::constant_response;
    mean_y /= T(rows);

    T residual_sum = 0, total_sum = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        const auto offset = std::ptrdiff_t(i) * stride;
        const T residual = unspliced[offset] - model.gamma * spliced[offset] - model.offset;
        const T total = unspliced[offset] - mean_y;
        residual_sum += residual * residual;
        total_sum += total * total;
    }
    if (!isfinite(residual_sum) || !isfinite(total_sum)) return FitStatus::singular;
    if (!(total_sum > 0)) return FitStatus::constant_response;
    result = T(1.0) - residual_sum / total_sum;
    return isfinite(result) ? FitStatus::ok : FitStatus::singular;
}


template<class T, class Sorter> inline Fit<T> fit_gene(const T* spliced, const T* unspliced,
                 std::size_t rows, std::ptrdiff_t stride,
                 const Options<T>& options, T* score_scratch,
                 const std::uint8_t* fit_mask, bool gene_enabled, Sorter sort, const TrimBounds<T>* forced = nullptr) {
    if (!rows) return failed<T>(FitStatus::insufficient_data);
    T all_max_x = -std::numeric_limits<T>::infinity();
    T all_max_y = -std::numeric_limits<T>::infinity();
    for (std::size_t i = 0; i < rows; ++i) {
        const auto offset = std::ptrdiff_t(i) * stride;
        const T x = spliced[offset], y = unspliced[offset];
        if (!isfinite(x) || !isfinite(y)) return failed<T>(FitStatus::singular);
        all_max_x = maximum(all_max_x, x);
        all_max_y = maximum(all_max_y, y);
    }
    if (!(all_max_x > 0) || !(all_max_y > 0)) return failed<T>(FitStatus::no_signal);

    std::size_t fitting_rows = 0;
    T fit_max_x = -std::numeric_limits<T>::infinity();
    T fit_max_y = -std::numeric_limits<T>::infinity();
    for (std::size_t i = 0; i < rows; ++i) {
        if (!included(i, fit_mask)) continue;
        const auto offset = std::ptrdiff_t(i) * stride;
        ++fitting_rows;
        fit_max_x = maximum(fit_max_x, spliced[offset]);
        fit_max_y = maximum(fit_max_y, unspliced[offset]);
    }
    if (!fitting_rows) return failed<T>(FitStatus::insufficient_data);

    const bool trim = options.trimming != Trimming::none;
    const bool lower_tail = options.trimming == Trimming::tails && options.fit_intercept;
    const bool upper_tail = options.trimming == Trimming::upper ||
                            options.trimming == Trimming::tails;
    T low = 0, high = 0;
    if (trim && forced) { low = forced->low; high = forced->high; }
    else if (trim) {
        std::size_t position = 0;
        for (std::size_t i = 0; i < rows; ++i) {
            if (!included(i, fit_mask)) continue;
            const auto offset = std::ptrdiff_t(i) * stride;
            score_scratch[position++] = score(spliced[offset], unspliced[offset],
                                               fit_max_x, fit_max_y);
        }
        for (std::size_t i = 0; i < fitting_rows; ++i)
            if (!isfinite(score_scratch[i])) return failed<T>(FitStatus::singular);
        sort(score_scratch, fitting_rows);
        if (lower_tail) low = percentile(score_scratch, fitting_rows, options.percentile_low);
        if (upper_tail) high = percentile(score_scratch, fitting_rows, options.percentile_high);
    }

    auto sums = !trim ? collect_all(spliced, unspliced, rows, stride, fit_mask)
        : lower_tail ? collect_tails(spliced, unspliced, rows, stride, fit_mask,
                                     fit_max_x, fit_max_y, low, high)
                     : collect_upper(spliced, unspliced, rows, stride, fit_mask,
                                     fit_max_x, fit_max_y, high);
    if (!sums.count) return failed<T>(FitStatus::insufficient_data);
    if (options.fit_intercept) {
        if (!trim) center<T, false, false>(sums, spliced, unspliced, rows, stride,
                                        fit_mask, fit_max_x, fit_max_y, 0, 0);
        else if (lower_tail) center<T, true, true>(sums, spliced, unspliced, rows, stride,
                                                fit_mask, fit_max_x, fit_max_y, low, high);
        else center<T, true, false>(sums, spliced, unspliced, rows, stride,
                                 fit_mask, fit_max_x, fit_max_y, 0, high);
    }
    const auto fitted = regress(sums, options.fit_intercept, options.positive_intercept);
    if (!fitted.valid) return failed<T>(FitStatus::singular);

    const T gamma = clamp(fitted.gamma, options.ratio_min, options.ratio_max);
    if (!isfinite(gamma)) return failed<T>(FitStatus::singular);
    const Regression<T> reported{gamma, fitted.offset, true};

    Regression<T> score_model = reported;
    if (options.r2_adjusted) {
        auto untrimmed = collect_all(spliced, unspliced, rows, stride, fit_mask);
        if (options.fit_intercept)
            center<T, false, false>(untrimmed, spliced, unspliced, rows, stride,
                                 fit_mask, fit_max_x, fit_max_y, 0, 0);
        score_model = regress(untrimmed, options.fit_intercept, options.positive_intercept);
        if (!score_model.valid) return failed<T>(FitStatus::singular);
    }

    T r2 = 0;
    const auto r2_status = r_squared(spliced, unspliced, rows, stride, score_model, r2);
    if (r2_status != FitStatus::ok) return failed<T>(r2_status);
    const bool selected = gene_enabled && gamma > options.min_ratio && r2 > options.min_r2;
    return {gamma, fitted.offset, r2, FitStatus::ok, selected};
}


template<class T> struct Interval { T low = 0, high = 0; };

template<class T, class Sorter> inline Interval<T> interval(T* x, std::size_t n, const Options<T>& o, Sorter sort) {
    sort(x, n);
    const bool tails = o.trimming == Trimming::tails && o.fit_intercept;
    return {tails ? percentile(x, n, o.percentile_low) : -std::numeric_limits<T>::infinity(),
            percentile(x, n, o.percentile_high)};
}

template<class T> inline bool extreme(T x, Interval<T> b) { return x <= b.low || x >= b.high; }

template<class T> inline JointFit<T> failed_joint(const Fit<T>& initial, Scales<T> scales, StochasticStatus status) {
    JointFit<T> out;
    if (!isfinite(scales.first)) scales.first = 0;
    if (!isfinite(scales.second)) scales.second = 0;
    out.gamma = initial.gamma; out.offset = initial.offset; out.scales = scales; out.status = status;
    return out;
}


template<class T> inline Scales<T> scales(const T* ms, const T* mu,
    const T* mss, const T* mus, std::size_t rows, std::ptrdiff_t stride,
    const Fit<T>& initial, const StochasticOptions<T>& options, const std::uint8_t* mask) {
    Scales<T> out;
    if (initial.status != FitStatus::ok) { out.status = StochasticStatus::invalid_initialization; return out; }
    std::size_t n = 0;
    T mx = 0, my = 0, xx = 0, xy = 0;
    T min_a = std::numeric_limits<T>::infinity(), max_a = -min_a;
    for (std::size_t i = 0; i < rows; ++i) {
        auto p = std::ptrdiff_t(i)*stride;
        T s = ms[p], u = mu[p], a = 2*mss[p] - s, b = 2*mus[p] + u;
        if (!isfinite(s) || !isfinite(u) || !isfinite(a) || !isfinite(b)) {
            out.status = StochasticStatus::nonfinite; return out;
        }
        if (!included(i, mask)) continue;
        min_a = minimum(min_a, a); max_a = maximum(max_a, a);
        ++n; mx += (a - mx)/T(n); my += (b - my)/T(n);
        xx += a*a; xy += a*b;
    }
    if (n < 2) { out.status = StochasticStatus::insufficient_data; return out; }
    if (!isfinite(mx) || !isfinite(my) || !isfinite(xx) || !isfinite(xy)) {
        out.status = StochasticStatus::nonfinite; return out;
    }

    T gamma2 = 0, offset2 = 0;
    if (options.fit_second_intercept) {
        if (min_a == max_a) { out.status = StochasticStatus::singular; return out; }
        T centered_xx = 0, centered_xy = 0;
        for (std::size_t i = 0; i < rows; ++i) if (included(i, mask)) {
            auto p = std::ptrdiff_t(i)*stride;
            T a = 2*mss[p] - ms[p] - mx, b = 2*mus[p] + mu[p] - my;
            centered_xx += a*a; centered_xy += a*b;
        }
        if (!isfinite(centered_xx) || !isfinite(centered_xy)) {
            out.status = StochasticStatus::nonfinite; return out;
        }
        if (!(centered_xx > 0)) { out.status = StochasticStatus::singular; return out; }
        gamma2 = centered_xy/centered_xx; offset2 = my - gamma2*mx;
        if (options.initialization.positive_intercept && offset2 < 0) {
            if (!(xx > 0)) { out.status = StochasticStatus::singular; return out; }
            gamma2 = xy/xx; offset2 = 0;
        }
    } else {
        if (!(xx > 0)) { out.status = StochasticStatus::singular; return out; }
        gamma2 = xy/xx;
    }
    if (!isfinite(gamma2) || !isfinite(offset2)) { out.status = StochasticStatus::nonfinite; return out; }

    T mean1 = 0, mean2 = 0, term1 = 0, term2 = 0;
    n = 0;
    for (std::size_t i = 0; i < rows; ++i) if (included(i, mask)) {
        auto p = std::ptrdiff_t(i)*stride;
        T r1 = mu[p] - initial.gamma*ms[p] - initial.offset;
        T r2 = 2*mus[p] + mu[p] - gamma2*(2*mss[p] - ms[p]) - offset2;
        term1 = maximum(term1, maximum(absolute(mu[p]), maximum(absolute(initial.gamma*ms[p]), absolute(initial.offset))));
        term2 = maximum(term2, maximum(absolute(2*mus[p] + mu[p]), maximum(absolute(gamma2*(2*mss[p] - ms[p])), absolute(offset2))));
        ++n; mean1 += (r1 - mean1)/T(n); mean2 += (r2 - mean2)/T(n);
    }
    T sum1 = 0, sum2 = 0;
    for (std::size_t i = 0; i < rows; ++i) if (included(i, mask)) {
        auto p = std::ptrdiff_t(i)*stride;
        T r1 = mu[p] - initial.gamma*ms[p] - initial.offset - mean1;
        T r2 = 2*mus[p] + mu[p] - gamma2*(2*mss[p] - ms[p]) - offset2 - mean2;
        sum1 += r1*r1; sum2 += r2*r2;
    }
    out.first = square_root(sum1/T(n)); out.second = square_root(sum2/T(n));
    if (!isfinite(out.first) || !isfinite(out.second)) {
        if (!isfinite(out.first)) out.first = 0;
        if (!isfinite(out.second)) out.second = 0;
        out.status = StochasticStatus::nonfinite;
    } else if (!(out.first > residual_floor<T>(term1)) || !(out.second > residual_floor<T>(term2))) out.status = StochasticStatus::zero_scale;
    else out.status = StochasticStatus::ok;
    return out;
}

template<class T, class Sorter> inline JointFit<T> joint_fit(const T* ms, const T* mu,
    const T* mss, const T* mus, std::size_t rows, std::ptrdiff_t stride,
    const Fit<T>& initial, Scales<T> scales, const StochasticOptions<T>& options,
    T* scratch, const std::uint8_t* mask, Sorter sort) {
    if (scales.status != StochasticStatus::ok) return failed_joint(initial, scales, scales.status);
    if (!(scales.first > 0) || !(scales.second > 0) ||
        !isfinite(scales.first) || !isfinite(scales.second))
        return failed_joint(initial, scales, StochasticStatus::zero_scale);

    const auto& opt = options.initialization;
    const bool trim = opt.trimming != Trimming::none;
    std::size_t n = 0;
    T max_s = -std::numeric_limits<T>::infinity(), max_u = max_s;
    for (std::size_t i = 0; i < rows; ++i) {
        auto p = std::ptrdiff_t(i)*stride;
        if (!isfinite(ms[p]) || !isfinite(mu[p]) ||
            !isfinite(2*mss[p] - ms[p]) || !isfinite(2*mus[p] + mu[p]))
            return failed_joint(initial, scales, StochasticStatus::nonfinite);
        if (!included(i, mask)) continue;
        ++n; max_s = maximum(max_s, ms[p]); max_u = maximum(max_u, mu[p]);
    }
    if (n < 2) return failed_joint(initial, scales, StochasticStatus::insufficient_data);
    max_s = maximum(max_s, T(1e-3)); max_u = maximum(max_u, T(1e-3));
    Interval<T> joint, spliced;
    if (trim) {
        std::size_t k = 0;
        for (std::size_t i = 0; i < rows; ++i) if (included(i, mask)) {
            auto p = std::ptrdiff_t(i)*stride;
            scratch[k++] = ms[p]/max_s + mu[p]/max_u;
        }
        joint = interval(scratch, n, opt, sort); k = 0;
        for (std::size_t i = 0; i < rows; ++i) if (included(i, mask)) scratch[k++] = ms[std::ptrdiff_t(i)*stride];
        spliced = interval(scratch, n, opt, sort);
    }
    auto keep = [&](T s, T u) {
        return !trim || extreme(s/max_s + u/max_u, joint) || extreme(s, spliced);
    };

    // The intercept columns have disjoint support. Project each block onto its
    // constant column, then solve the slope from the orthogonal residuals.
    // This is QR without allocating the 2N by 3 design or forming X'X.
    T x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    std::size_t k = 0;
    for (std::size_t i = 0; i < rows; ++i) if (included(i, mask)) {
        auto p = std::ptrdiff_t(i)*stride;
        bool retain = keep(ms[p], mu[p]);
        T a = retain ? ms[p] : 0, b = retain ? mu[p] : 0;
        T c = 2*mss[p] - ms[p], d = 2*mus[p] + mu[p];
        ++k; x1 += (a - x1)/T(k); y1 += (b - y1)/T(k);
        x2 += (c - x2)/T(k); y2 += (d - y2)/T(k);
    }
    T total = 0, diagonal = 0, rhs = 0;
    for (std::size_t i = 0; i < rows; ++i) if (included(i, mask)) {
        auto p = std::ptrdiff_t(i)*stride;
        bool retain = keep(ms[p], mu[p]);
        T a = retain ? ms[p] : 0, b = retain ? mu[p] : 0;
        T c = 2*mss[p] - ms[p], d = 2*mus[p] + mu[p];
        total += (a/scales.first)*(a/scales.first) + (c/scales.second)*(c/scales.second);
        if (opt.fit_intercept) { a -= x1; b -= y1; }
        if (options.fit_second_intercept) { c -= x2; d -= y2; }
        a /= scales.first; b /= scales.first; c /= scales.second; d /= scales.second;
        diagonal += a*a + c*c; rhs += a*b + c*d;
    }
    if (!isfinite(total) || !isfinite(diagonal) || !isfinite(rhs))
        return failed_joint(initial, scales, StochasticStatus::nonfinite);
    if (!(total > 0) || !(diagonal > 0)) return failed_joint(initial, scales, StochasticStatus::singular);
    if (opt.fit_intercept || options.fit_second_intercept) {
        // The normalized triangular R has unit diagonal in each intercept
        // column and a unit-norm slope column. Its extreme singular-value
        // ratio is d/(1 + sqrt(1-d*d)), with d the residual slope norm.
        T d = minimum(T(1.), square_root(diagonal/total));
        T ratio = d/(1 + square_root(maximum(T(0.), 1 - d*d)));
        if (!(ratio > rank_tolerance<T>())) return failed_joint(initial, scales, StochasticStatus::singular);
    }
    JointFit<T> out;
    out.gamma = clamp(rhs/diagonal, opt.ratio_min, opt.ratio_max);
    out.offset = opt.fit_intercept ? y1 - out.gamma*x1 : 0;
    out.offset2 = options.fit_second_intercept ? y2 - out.gamma*x2 : 0;
    out.scales = scales; out.status = StochasticStatus::ok;
    if (!isfinite(out.gamma) || !isfinite(out.offset) || !isfinite(out.offset2))
        return failed_joint(initial, scales, StochasticStatus::nonfinite);
    T objective = 0;
    for (std::size_t i = 0; i < rows; ++i) if (included(i, mask)) {
        auto p = std::ptrdiff_t(i)*stride;
        bool retain = keep(ms[p], mu[p]);
        T a = retain ? ms[p] : 0, b = retain ? mu[p] : 0;
        T r1 = (b - out.gamma*a - out.offset)/scales.first;
        T r2 = (2*mus[p] + mu[p] - out.gamma*(2*mss[p] - ms[p]) - out.offset2)/scales.second;
        objective += r1*r1 + r2*r2;
    }
    out.objective = objective/(2*T(n));
    if (!isfinite(out.objective)) return failed_joint(initial, scales, StochasticStatus::nonfinite);
    return out;
}

}
