module;
#include <stddef.h>
#include <stdint.h>
#include "detail/sensitivity.hh"

module velocitypp.precision;
import velocitypp;

namespace velocitypp::precision {
namespace {

template<class T> velocitypp::detail::sensitivity::Options<T> numeric(SensitivityOptions o) {
    return {T(o.band), T(o.prediction_limit), T(o.r2_limit)};
}
template<class T> velocitypp::detail::scientific::Options<T> numeric(const FitOptions& o) {
    using namespace velocitypp::detail;
    return {scientific::Trimming(o.trimming), T(o.percentile_low), T(o.percentile_high),
            o.fit_intercept, o.positive_intercept, o.r2_adjusted,
            T(o.ratio_min), T(o.ratio_max), T(o.min_r2), T(o.min_ratio)};
}
template<class T> BasicSensitivityReport<T> report_gene(const T* x, const T* y,
    size_t rows, ptrdiff_t stride, const BasicGeneFit<T>& base, const FitOptions& fit,
    T* scratch, const uint8_t* mask, bool enabled, SensitivityOptions o) noexcept {
    using namespace velocitypp::detail;
    auto r = sensitivity::report(x, y, rows, stride,
        scientific::Fit<T>{base.gamma, base.offset, base.r2, scientific::FitStatus(base.status), base.selected},
        numeric<T>(fit), scratch, mask, enabled, numeric<T>(o), scientific::HeapSort{});
    return {SensitivityState(r.state), r.low, r.high, r.delta, r.prediction_change, r.r2_change,
            FitStatus(r.expanded_status), FitStatus(r.contracted_status), r.status_change, r.selection_change};
}
size_t gene(const Promotion& p, size_t i) { return p.genes ? p.genes[i] : i; }
bool valid_stride(size_t rows, ptrdiff_t stride) {
    if (rows < 2) return true;
    uintmax_t steps = uintmax_t(rows-1);
    if (steps > uintmax_t(PTRDIFF_MAX)) return false;
    if (!stride) return true;
    uintmax_t magnitude = stride < 0 ? uintmax_t(-(stride+1))+1 : uintmax_t(stride);
    uintmax_t limit = uintmax_t(PTRDIFF_MAX) + uintmax_t(stride < 0);
    return steps <= limit/magnitude;
}
bool valid_column(const OriginalColumn& c, size_t rows) {
    if (c.rows != rows || (rows && (!c.spliced || !c.unspliced)) ||
        !valid_stride(rows,c.spliced_stride_bytes) || !valid_stride(rows,c.unspliced_stride_bytes)) return false;
    return (c.spliced_type == OriginalType::float32 || c.spliced_type == OriginalType::float64) &&
           (c.unspliced_type == OriginalType::float32 || c.unspliced_type == OriginalType::float64);
}

} // namespace

PrecisionCheck validate_sensitivity(size_t rows, ptrdiff_t stride, const void* x,
    const void* y, const void* scratch, size_t scratch_count,
    const FitOptions& f, SensitivityOptions o) noexcept {
    if (!__builtin_isfinite(o.band) || !__builtin_isfinite(o.prediction_limit) ||
        !__builtin_isfinite(o.r2_limit) || o.band < 0 || o.prediction_limit < 0 || o.r2_limit < 0 ||
        __builtin_isnan(f.ratio_min) || __builtin_isnan(f.ratio_max) || f.ratio_min > f.ratio_max ||
        !__builtin_isfinite(f.min_r2) || !__builtin_isfinite(f.min_ratio) ||
        !__builtin_isfinite(f.percentile_low) || !__builtin_isfinite(f.percentile_high) ||
        f.percentile_low < 0 || f.percentile_high > 100 || f.percentile_low > f.percentile_high ||
        (f.trimming != Trimming::none && f.trimming != Trimming::upper && f.trimming != Trimming::tails))
        return {PrecisionCode::invalid_argument, 0, 0};
    if (rows && (!x || !y || !stride)) return {PrecisionCode::invalid_argument, 0, 0};
    if (rows && f.trimming != Trimming::none && (!scratch || scratch_count < rows))
        return {PrecisionCode::insufficient_capacity, 0, rows};
    if (rows > 1) {
        uintmax_t step = stride > 0 ? uintmax_t(stride) : uintmax_t(-(stride+1))+1;
        if (uintmax_t(rows-1) > uintmax_t(PTRDIFF_MAX)/step) return {PrecisionCode::invalid_argument, 0, 0};
    }
    return {};
}
SensitivityReport sensitivity_gene(const double* x, const double* y, size_t rows,
    ptrdiff_t stride, const GeneFit& base, const FitOptions& fit, double* scratch,
    const uint8_t* mask, bool enabled, SensitivityOptions o) noexcept {
    return report_gene(x,y,rows,stride,base,fit,scratch,mask,enabled,o);
}
SensitivityReport32 sensitivity_gene(const float* x, const float* y, size_t rows,
    ptrdiff_t stride, const GeneFit32& base, const FitOptions& fit, float* scratch,
    const uint8_t* mask, bool enabled, SensitivityOptions o) noexcept {
    return report_gene(x,y,rows,stride,base,fit,scratch,mask,enabled,o);
}

namespace {
template<class T> PrecisionCode replace_reference(ReferenceMoments<T>& r, size_t position, size_t offset,
                                                   const OriginalColumn& c, bool second) noexcept {
    if (!r.input || !valid_column(c,r.input->rows)) return PrecisionCode::invalid_argument;
    auto st = c.spliced_type, ut = c.unspliced_type;
    if (!second) {
        auto* b = r.initial;
        if (!b || offset >= b->count || position < b->first || position-b->first != offset || b->rows != c.rows ||
            b->stride <= 0 || (c.rows && (!b->spliced || !b->unspliced))) return PrecisionCode::invalid_argument;
        reference::moments_from(*r.input,c.spliced,c.spliced_stride_bytes,st,
            c.unspliced,c.unspliced_stride_bytes,ut,b->spliced ? b->spliced+offset : nullptr,b->unspliced ? b->unspliced+offset : nullptr,size_t(b->stride));
    } else {
        auto* b = r.second;
        if (!b || offset >= b->count || position < b->first || position-b->first != offset || b->rows != c.rows || b->stride <= 0 ||
            (c.rows && (!b->spliced || !b->unspliced || !b->spliced_second || !b->cross))) return PrecisionCode::invalid_argument;
        reference::second_moments_from(*r.input,c.spliced,c.spliced_stride_bytes,st,
            c.unspliced,c.unspliced_stride_bytes,ut,b->spliced ? b->spliced+offset : nullptr,b->unspliced ? b->unspliced+offset : nullptr,
            b->spliced_second ? b->spliced_second+offset : nullptr,b->cross ? b->cross+offset : nullptr,
            b->unspliced_second ? b->unspliced_second+offset : nullptr,size_t(b->stride));
    }
    return PrecisionCode::ok;
}
}
PrecisionCode replace_reference64(void* p, size_t position, size_t offset,
                                  const OriginalColumn& c, bool second) noexcept {
    return p ? replace_reference(*static_cast<ReferenceMoments<double>*>(p),position,offset,c,second) : PrecisionCode::invalid_argument;
}
PrecisionCode replace_reference32(void* p, size_t position, size_t offset,
                                  const OriginalColumn& c, bool second) noexcept {
    return p ? replace_reference(*static_cast<ReferenceMoments<float>*>(p),position,offset,c,second) : PrecisionCode::invalid_argument;
}

PrecisionCheck validate_promotion(const Promotion& p) noexcept {
    if (p.mask && p.mask_count != p.source_gene_count) return {PrecisionCode::invalid_argument, 0, p.source_gene_count};
    if (p.mask && p.gene_count && (!p.status || p.status_capacity < p.gene_count))
        return {PrecisionCode::insufficient_capacity, 0, p.gene_count};
    if (!p.genes && p.gene_count > p.source_gene_count) return {PrecisionCode::invalid_argument, p.source_gene_count, 0};
    for (size_t i = 0; p.genes && i < p.gene_count; ++i)
        if (p.genes[i] >= p.source_gene_count) return {PrecisionCode::invalid_argument, i, 0};
    return {};
}
PrecisionCheck initialize_promotion(Promotion& p) noexcept {
    auto check = validate_promotion(p);
    if (check.code != PrecisionCode::ok) return check;
    p.promoted = 0;
    if (p.mask) for (size_t i=0;i<p.gene_count;++i) p.status[i] = uint8_t(PromotionStatus::not_requested);
    return {};
}
PrecisionCheck promote_initial(Promotion& p, size_t first, size_t count) {
    if (first > p.gene_count || count > p.gene_count-first) return {PrecisionCode::invalid_argument, first, 0};
    if (!p.mask) return {};
    for (size_t j=0;j<count;++j) {
        size_t i=first+j, g=gene(p,i);
        if (!p.mask[g]) continue;
        OriginalColumn c;
        if (!p.source.acquire || !p.source.acquire(p.source.context,g,c)) {
            p.status[i]=uint8_t(PromotionStatus::unavailable); continue;
        }
        if (!valid_column(c,p.rows) || !p.replacement.replace) return {PrecisionCode::invalid_argument,i,0};
        auto code=p.replacement.replace(p.replacement.context,i,j,c,false);
        if (code != PrecisionCode::ok) return {code,i,0};
        p.status[i]=uint8_t(PromotionStatus::promoted); ++p.promoted;
    }
    return {};
}
PrecisionCheck promote_second(Promotion& p, size_t first, size_t count) {
    if (first > p.gene_count || count > p.gene_count-first) return {PrecisionCode::invalid_argument, first, 0};
    if (!p.mask) return {};
    for (size_t j=0;j<count;++j) {
        size_t i=first+j;
        if (p.status[i] != uint8_t(PromotionStatus::promoted)) continue;
        OriginalColumn c;
        if (!p.source.acquire || !p.source.acquire(p.source.context,gene(p,i),c))
            return {PrecisionCode::unavailable,i,0};
        if (!valid_column(c,p.rows) || !p.replacement.replace) return {PrecisionCode::invalid_argument,i,0};
        auto code=p.replacement.replace(p.replacement.context,i,j,c,true);
        if (code != PrecisionCode::ok) return {code,i,0};
    }
    return {};
}

} // namespace velocitypp::precision
