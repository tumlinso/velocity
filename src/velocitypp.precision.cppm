module;
#include <stddef.h>
#include <stdint.h>

export module velocitypp.precision;
export import velocitypp;

export namespace velocitypp::precision {

struct SensitivityOptions { double band = .5, prediction_limit = .1, r2_limit = .05; };
enum class SensitivityState : uint8_t { no_trimming, unscoreable, stable, sensitive };
template<class T> struct BasicSensitivityReport {
    SensitivityState state = SensitivityState::unscoreable;
    T low = 0, high = 0, delta = 0, prediction_change = 0, r2_change = 0;
    FitStatus expanded_status = FitStatus::insufficient_data;
    FitStatus contracted_status = FitStatus::insufficient_data;
    bool status_change = false, selection_change = false;
};
using SensitivityReport = BasicSensitivityReport<double>;
using SensitivityReport32 = BasicSensitivityReport<float>;

enum class PrecisionCode : uint8_t { ok, invalid_argument, insufficient_capacity, unavailable, replacement_failed };
struct PrecisionCheck { PrecisionCode code = PrecisionCode::ok; size_t location = 0, required = 0; };

// scratch holds at least rows elements when trimming is enabled; the buffers
// are disjoint and have a common signed element stride. The unchecked routines
// require valid options, extents and pointers. The report is advisory only.
PrecisionCheck validate_sensitivity(size_t rows, ptrdiff_t stride, const void* spliced,
    const void* unspliced, const void* scratch, size_t scratch_count,
    const FitOptions&, SensitivityOptions = {}) noexcept;
SensitivityReport sensitivity_gene(const double* spliced, const double* unspliced,
    size_t rows, ptrdiff_t stride, const GeneFit& baseline, const FitOptions&,
    double* scratch, const uint8_t* fit_mask = nullptr, bool gene_enabled = true,
    SensitivityOptions = {}) noexcept;
SensitivityReport32 sensitivity_gene(const float* spliced, const float* unspliced,
    size_t rows, ptrdiff_t stride, const GeneFit32& baseline, const FitOptions&,
    float* scratch, const uint8_t* fit_mask = nullptr, bool gene_enabled = true,
    SensitivityOptions = {}) noexcept;

// A provider supplies already acquired original expression columns. Pointers
// address the first logical observation; strides are signed bytes. Values are
// FP32 or FP64. The full signed byte-stride extent must fit ptrdiff_t.
// The provider owns storage until its replacement call returns.
using OriginalType = anndatapp::DType;
struct OriginalColumn {
    const void* spliced = nullptr;
    const void* unspliced = nullptr;
    ptrdiff_t spliced_stride_bytes = 0, unspliced_stride_bytes = 0;
    OriginalType spliced_type = OriginalType::float32, unspliced_type = OriginalType::float32;
    size_t rows = 0;
};
struct OriginalSource {
    void* context = nullptr;
    bool (*acquire)(void*, size_t source_gene, OriginalColumn&) = nullptr;
};
enum class PromotionStatus : uint8_t { not_requested, promoted, unavailable };

// A replacement callback consumes the original column immediately. position
// is the prepared-gene index, offset is its position within the active block;
// second is false for initial moments and true for second-order moments.
struct MomentReplacement {
    void* context = nullptr;
    PrecisionCode (*replace)(void*, size_t position, size_t offset,
                             const OriginalColumn&, bool second) = nullptr;
};

// Reference replacement binds the active reference blocks and graph. Rebind
// initial/second before each corresponding block. All objects and storage are
// caller owned; output block stride is in elements and must be positive.
template<class T> struct ReferenceMoments {
    const PreparedInput* input = nullptr;
    BasicMomentBlock<T>* initial = nullptr;
    BasicStochasticMomentBlock<T>* second = nullptr;
};
PrecisionCode replace_reference64(void*, size_t position, size_t offset,
                                  const OriginalColumn&, bool second) noexcept;
PrecisionCode replace_reference32(void*, size_t position, size_t offset,
                                  const OriginalColumn&, bool second) noexcept;

// All pointers are borrowed. genes maps prepared to source order; null means
// identity. mask is in source order. status is in prepared order. Keep genes
// and mask immutable until all passes finish. A successful initial promotion
// is fetched again during the second pass. Callback changes
// and state reuse must be synchronized by the caller. Initialize before these
// trusted block calls; cover each prepared gene once per pass. On failure,
// locations before the reported position are complete; the failing callback
// may have written partial results. There is no implicit rollback.
struct Promotion {
    const size_t* genes = nullptr;
    size_t gene_count = 0, source_gene_count = 0, rows = 0;
    const uint8_t* mask = nullptr;
    size_t mask_count = 0;
    uint8_t* status = nullptr;
    size_t status_capacity = 0, promoted = 0;
    OriginalSource source;
    MomentReplacement replacement;
};
PrecisionCheck validate_promotion(const Promotion&) noexcept;
PrecisionCheck initialize_promotion(Promotion&) noexcept;
PrecisionCheck promote_initial(Promotion&, size_t first, size_t count);
PrecisionCheck promote_second(Promotion&, size_t first, size_t count);

} // namespace velocitypp::precision
