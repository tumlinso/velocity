module;
#include <cstddef>
#include <cstdint>
#include <memory>

export module velocitypp.precision;
export import velocitypp;

export namespace velocitypp::precision {

struct SensitivityOptions {
    double band = .5, prediction_limit = .1, r2_limit = .05;
};
enum class SensitivityState : std::uint8_t { no_trimming, unscoreable, stable, sensitive };

template<class T> struct BasicSensitivityReport {
    SensitivityState state = SensitivityState::unscoreable;
    T low = 0, high = 0, delta = 0, prediction_change = 0, r2_change = 0;
    FitStatus expanded_status = FitStatus::insufficient_data;
    FitStatus contracted_status = FitStatus::insufficient_data;
    bool status_change = false, selection_change = false;
};
using SensitivityReport = BasicSensitivityReport<double>;
using SensitivityReport32 = BasicSensitivityReport<float>;

// Optional CPU scratch owner. It checks the existing stage workspace and
// caller-declared promotion staging against the total numeric budget. Results
// remain caller-owned; only the rows-element score buffer is retained.
template<class T> class BasicSensitivityWorkspace {
public:
    BasicSensitivityWorkspace(const PreparedInput&, const Workspace&, const BasicReferenceStages<T>&,
        std::size_t budget_bytes = 64 * 1024 * 1024, std::size_t staging_bytes = 0);
    BasicSensitivityWorkspace(const PreparedInput&, const StochasticWorkspace&, const BasicStochasticReferenceStages<T>&,
        std::size_t budget_bytes = 64 * 1024 * 1024, std::size_t staging_bytes = 0);
    ~BasicSensitivityWorkspace();
    BasicSensitivityWorkspace(BasicSensitivityWorkspace&&) noexcept;
    BasicSensitivityWorkspace& operator=(BasicSensitivityWorkspace&&) noexcept;
    BasicSensitivityWorkspace(const BasicSensitivityWorkspace&) = delete;
    BasicSensitivityWorkspace& operator=(const BasicSensitivityWorkspace&) = delete;
    T* data() noexcept;
    std::size_t scratch_bytes() const noexcept;
private:
    std::unique_ptr<T[]> data_;
    std::size_t bytes_ = 0;
};
using SensitivityWorkspace = BasicSensitivityWorkspace<double>;
using SensitivityWorkspace32 = BasicSensitivityWorkspace<float>;

// The supplied baseline is the existing initialization fit. scratch has at
// least rows elements and is disjoint from observations. With trimming off it
// is unused. Account manually supplied scratch against the run's workspace
// budget. The baseline's selection must precede global low-signal recovery.
// The report is advisory and does not request promotion.
SensitivityReport sensitivity_gene(const double* spliced, const double* unspliced,
    std::size_t rows, std::ptrdiff_t stride, const GeneFit& baseline,
    const FitOptions&, double* scratch, const std::uint8_t* fit_mask = nullptr,
    bool gene_enabled = true, SensitivityOptions = {});
SensitivityReport32 sensitivity_gene(const float* spliced, const float* unspliced,
    std::size_t rows, std::ptrdiff_t stride, const GeneFit32& baseline,
    const FitOptions&, float* scratch, const std::uint8_t* fit_mask = nullptr,
    bool gene_enabled = true, SensitivityOptions = {});

enum class PromotionStatus : std::uint8_t { not_requested, promoted, unavailable };
struct OriginalColumn { anndatapp::ArrayView spliced, unspliced; };
struct OriginalSource {
    void* context = nullptr;
    // Called only for explicitly requested source genes. Return false when the
    // original pair is unavailable. Each returned view is dense N by 1.
    // Promotion requires both columns in FP32/FP64.
    // Sum sparse duplicates while materializing a column. Owners and values
    // remain valid until consumed; later acquisitions must reproduce them.
    bool (*acquire)(void*, std::size_t source_gene, OriginalColumn&) = nullptr;
    // Caller-owned staging capacity is charged against the numeric budget.
    std::size_t staging_bytes = 0;
};

// Retains dense array views and returns zero-copy single-gene views on demand.
// Borrowed raw storage still requires its owner. Keep this object alive while
// source() is in use.
class OriginalArraySource {
public:
    OriginalArraySource(anndatapp::ArrayView spliced, anndatapp::ArrayView unspliced);
    OriginalArraySource(const OriginalArraySource&) = delete;
    OriginalArraySource& operator=(const OriginalArraySource&) = delete;
    OriginalArraySource(OriginalArraySource&&) = delete;
    OriginalArraySource& operator=(OriginalArraySource&&) = delete;
    OriginalSource source() noexcept;
private:
    static bool acquire(void*, std::size_t, OriginalColumn&);
    anndatapp::ArrayView spliced_, unspliced_;
};
struct PromotionOptions {
    // Host mask, indexed by source gene; keep immutable throughout the run.
    const std::uint8_t* source_gene_mask = nullptr;
    std::size_t source_gene_count = 0;
    std::size_t workspace_bytes = 64 * 1024 * 1024;
};

namespace detail { template<class T> struct SelectionState; }

// These wrappers are explicit opt-in stage compositions.  They never replace
// the reference path. Their status pointer is in prepared-gene order and is
// valid until destruction; it is null when no mask or no genes were supplied. Unavailable
// originals retain baseline storage. promotion_count counts prepared-gene
// occurrences, including duplicate selections.
template<class T> class BasicSelectiveStages {
public:
    BasicSelectiveStages(const PreparedInput&, Workspace&, BasicReferenceStages<T>&,
                         OriginalSource, PromotionOptions);
    ~BasicSelectiveStages();
    BasicSelectiveStages(BasicSelectiveStages&&) noexcept;
    BasicSelectiveStages& operator=(BasicSelectiveStages&&) noexcept;
    BasicSelectiveStages(const BasicSelectiveStages&) = delete;
    BasicSelectiveStages& operator=(const BasicSelectiveStages&) = delete;
    BasicMomentBlock<T> moments(std::size_t first, std::size_t count);
    BasicFitBlock<T> fit(BasicMomentBlock<T>);
    void emit(BasicMomentBlock<T>, BasicFitBlock<T>);
    FitResult finish();
    const std::uint8_t* promotion_status() const noexcept;
    std::size_t promotion_count() const noexcept;
private:
    std::unique_ptr<detail::SelectionState<T>> state_;
};
using SelectiveStages = BasicSelectiveStages<double>;
using SelectiveStages32 = BasicSelectiveStages<float>;

template<class T> class BasicSelectiveStochasticStages {
public:
    BasicSelectiveStochasticStages(const PreparedInput&, StochasticWorkspace&,
        BasicStochasticReferenceStages<T>&, OriginalSource, PromotionOptions);
    ~BasicSelectiveStochasticStages();
    BasicSelectiveStochasticStages(BasicSelectiveStochasticStages&&) noexcept;
    BasicSelectiveStochasticStages& operator=(BasicSelectiveStochasticStages&&) noexcept;
    BasicSelectiveStochasticStages(const BasicSelectiveStochasticStages&) = delete;
    BasicSelectiveStochasticStages& operator=(const BasicSelectiveStochasticStages&) = delete;
    BasicMomentBlock<T> initial_moments(std::size_t first, std::size_t count);
    BasicFitBlock<T> initial_fit(BasicMomentBlock<T>);
    void accept_initialization(BasicMomentBlock<T>, BasicFitBlock<T>);
    void resolve_selection();
    BasicStochasticMomentBlock<T> moments(std::size_t first, std::size_t count);
    BasicStochasticScaleBlock<T> scales(BasicStochasticMomentBlock<T>);
    BasicStochasticFitBlock<T> fit(BasicStochasticMomentBlock<T>, BasicStochasticScaleBlock<T>);
    void emit(BasicStochasticMomentBlock<T>, BasicStochasticFitBlock<T>);
    StochasticFitResult finish();
    const std::uint8_t* promotion_status() const noexcept;
    std::size_t promotion_count() const noexcept;
private:
    std::unique_ptr<detail::SelectionState<T>> state_;
};
using SelectiveStochasticStages = BasicSelectiveStochasticStages<double>;
using SelectiveStochasticStages32 = BasicSelectiveStochasticStages<float>;

}
