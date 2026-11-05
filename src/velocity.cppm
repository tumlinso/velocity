module;
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

export module velocity;
export import anndatapp.core;
export import :numeric;

export namespace velocity {

enum class ComputePrecision : std::uint8_t { float32, float64, automatic };

enum class FitStatus : std::uint8_t { ok, insufficient_data, no_signal, singular, constant_response };
enum class LowSignal { keep_filters, scvelo_percentile };
enum class Trimming { none, upper, tails };

struct FitOptions {
    Trimming trimming = Trimming::tails;
    double percentile_low = 5, percentile_high = 95;
    bool fit_intercept = false, positive_intercept = true, r2_adjusted = false;
    double ratio_min = -std::numeric_limits<double>::infinity();
    double ratio_max = std::numeric_limits<double>::infinity();
    double min_r2 = .01, min_ratio = .01;
    LowSignal low_signal = LowSignal::keep_filters;
};

struct RunOptions {
    std::size_t workspace_bytes = 64 * 1024 * 1024, block_genes = 64;
    std::optional<StorageType> output_dtype;
    bool velocity = true, spliced_moments = false, unspliced_moments = false;
    ComputePrecision compute_precision = ComputePrecision::automatic;
};

struct Input {
    NumericView spliced, unspliced, weights;
    anndatapp::Axis genes = anndatapp::Axis::all();
    std::optional<anndatapp::ArrayView> fit_mask, gene_mask;
};

Input from_anndata(const anndatapp::AnnData&, std::string_view spliced = "spliced",
                  std::string_view unspliced = "unspliced",
                  std::string_view weights = "connectivities");

namespace detail { struct PreparedState; struct WorkspaceState; template<class T> struct BasicStagesState; struct StochasticWorkspaceState; template<class T> struct BasicStochasticStagesState; struct Access; }

class PreparedInput {
public:
    explicit PreparedInput(Input, anndatapp::Validation = anndatapp::Validation::full);
    ~PreparedInput();
    PreparedInput(PreparedInput&&) noexcept;
    PreparedInput& operator=(PreparedInput&&) noexcept;
    PreparedInput(const PreparedInput&) = delete;
    PreparedInput& operator=(const PreparedInput&) = delete;
    std::size_t n_cells() const noexcept;
    std::size_t n_genes() const noexcept;
    std::size_t source_genes() const noexcept;
    std::size_t gene_index(std::size_t position) const;
    std::size_t storage_bytes() const noexcept;
    StorageType output_dtype() const noexcept;
    void validate_values() const;
    NumericView spliced() const;
    NumericView unspliced() const;
    NumericView weights() const;
    std::optional<anndatapp::ArrayView> fit_mask() const;
    std::optional<anndatapp::ArrayView> gene_mask() const;
private:
    std::unique_ptr<detail::PreparedState> state_;
    friend struct detail::Access;
};

class Workspace {
public:
    explicit Workspace(const PreparedInput&, RunOptions = {});
    ~Workspace();
    Workspace(Workspace&&) noexcept;
    Workspace& operator=(Workspace&&) noexcept;
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    std::size_t block_genes() const noexcept;
    std::size_t scratch_bytes() const noexcept;
    ComputePrecision compute_precision() const noexcept;
private:
    std::unique_ptr<detail::WorkspaceState> state_;
    friend struct detail::Access;
};

template<class T> struct BasicGeneFit {
    T gamma = 0, offset = 0, r2 = 0;
    FitStatus status = FitStatus::insufficient_data;
    bool selected = false;
};
using GeneFit = BasicGeneFit<double>;
using GeneFit32 = BasicGeneFit<float>;

struct FitResult {
    anndatapp::Array gamma, offset, r2, genes, status;
    bool low_signal = false;
};

// Numerical block pointers are borrowed and row-major with the stated stride.
template<class T> struct BasicMomentBlock {
    std::size_t first = 0, count = 0, rows = 0;
    std::ptrdiff_t stride = 0;
    T* spliced = nullptr;
    T* unspliced = nullptr;
};
using MomentBlock = BasicMomentBlock<double>;
using MomentBlock32 = BasicMomentBlock<float>;

template<class T> struct BasicFitBlock {
    std::size_t first = 0, count = 0;
    T* gamma = nullptr;
    T* offset = nullptr;
    T* r2 = nullptr;
    bool* selected = nullptr;
    std::uint8_t* status = nullptr;
};
using FitBlock = BasicFitBlock<double>;
using FitBlock32 = BasicFitBlock<float>;

// These views alias reusable workspace; a sink must finish reading before return.
struct OutputBlock {
    std::size_t first = 0, count = 0;
    NumericView velocity, spliced, unspliced;
};
struct BlockSink {
    void* context = nullptr;
    void (*write)(void*, const OutputBlock&) = nullptr;
};

template<class T> class BasicReferenceStages {
public:
    BasicReferenceStages(const PreparedInput&, Workspace&, FitOptions = {}, BlockSink = {});
    ~BasicReferenceStages();
    BasicReferenceStages(BasicReferenceStages&&) noexcept;
    BasicReferenceStages& operator=(BasicReferenceStages&&) noexcept;
    BasicReferenceStages(const BasicReferenceStages&) = delete;
    BasicReferenceStages& operator=(const BasicReferenceStages&) = delete;
    BasicMomentBlock<T> moments(std::size_t first, std::size_t count);
    BasicFitBlock<T> fit(BasicMomentBlock<T>);
    // Alternate sources must be dense N by count batches; caller owns staging.
    void replace_moments(BasicMomentBlock<T>&, std::size_t offset, std::size_t count, NumericView s, NumericView u);
    void emit(BasicMomentBlock<T>, BasicFitBlock<T>);
    FitResult finish();
    bool bound_to(const PreparedInput&, const Workspace&) const noexcept;
private:
    std::unique_ptr<detail::BasicStagesState<T>> state_;
    friend struct detail::Access;
};
using ReferenceStages = BasicReferenceStages<double>;
using ReferenceStages32 = BasicReferenceStages<float>;

// Custom stage results can be opaque device state. Only the reference stages
// prescribe host arrays. No dispatch occurs within an element loop.
template<class Operations>
auto steady_state_with(Operations& operations, std::size_t genes, std::size_t block_genes = 64) {
    if (!block_genes) throw std::invalid_argument("block_genes must be positive");
    if constexpr (requires { operations.begin(); }) operations.begin();
    for (std::size_t first = 0; first < genes;) {
        const auto count = genes - first < block_genes ? genes - first : block_genes;
        if constexpr (requires { operations.process_block(first, count); })
            operations.process_block(first, count);
        else {
            auto moments = operations.moments(first, count);
            auto fit = operations.fit(moments);
            operations.emit(moments, fit);
        }
        first += count;
    }
    return operations.finish();
}

namespace reference {
anndatapp::Array prepare_neighborhood(NumericView connectivities);
void moments(const PreparedInput&, std::size_t first, std::size_t count,
             double* spliced, double* unspliced);
GeneFit fit_gene(const double* spliced, const double* unspliced, std::size_t rows,
                 std::ptrdiff_t stride, const FitOptions&, double* score_scratch,
                 const std::uint8_t* fit_mask = nullptr, bool gene_enabled = true);
void residuals(const MomentBlock&, const FitBlock&, NumericEdit output);
FitResult steady_state(const PreparedInput&, Workspace&, FitOptions = {}, BlockSink = {});
FitResult steady_state_into(const PreparedInput&, Workspace&, NumericEdit velocity,
                            FitOptions = {});

void moments(const PreparedInput&, std::size_t first, std::size_t count,
             float* spliced, float* unspliced);
GeneFit32 fit_gene(const float* spliced, const float* unspliced, std::size_t rows,
                 std::ptrdiff_t stride, const FitOptions&, float* score_scratch,
                 const std::uint8_t* fit_mask = nullptr, bool gene_enabled = true);
void residuals(const MomentBlock32&, const FitBlock32&, NumericEdit output);
}

// Stochastic refinement preserves the deterministic selection/diagnostics.
enum class StochasticStatus : std::uint8_t {
    ok, not_selected, invalid_initialization, insufficient_data, zero_scale, singular, nonfinite
};
struct StochasticOptions {
    FitOptions initialization;
    bool fit_second_intercept = false, refit_all_if_none = false;
};
struct StochasticRunOptions {
    std::size_t workspace_bytes = 64 * 1024 * 1024, block_genes = 64;
    std::optional<StorageType> output_dtype;
    bool velocity = true, second_residual = false;
    bool spliced_moments = false, unspliced_moments = false;
    bool spliced_second = false, cross = false, unspliced_second = false;
    ComputePrecision compute_precision = ComputePrecision::automatic;
};
template<class T> struct BasicStochasticScales {
    T first = 0, second = 0;
    StochasticStatus status = StochasticStatus::not_selected;
};
using StochasticScales = BasicStochasticScales<double>;
using StochasticScales32 = BasicStochasticScales<float>;
template<class T> struct BasicStochasticGeneFit {
    T gamma = 0, offset = 0, offset2 = 0;
    BasicStochasticScales<T> scales;
    StochasticStatus status = StochasticStatus::not_selected;
    T objective = 0;
};
using StochasticGeneFit = BasicStochasticGeneFit<double>;
using StochasticGeneFit32 = BasicStochasticGeneFit<float>;
struct StochasticFitResult {
    FitResult initialization;
    anndatapp::Array gamma, offset, offset2, first_scale, second_scale, status, objective;
    bool refit_all = false;
};
class StochasticWorkspace {
public:
    explicit StochasticWorkspace(const PreparedInput&, StochasticRunOptions = {});
    ~StochasticWorkspace();
    StochasticWorkspace(StochasticWorkspace&&) noexcept;
    StochasticWorkspace& operator=(StochasticWorkspace&&) noexcept;
    StochasticWorkspace(const StochasticWorkspace&) = delete;
    StochasticWorkspace& operator=(const StochasticWorkspace&) = delete;
    std::size_t block_genes() const noexcept;
    std::size_t scratch_bytes() const noexcept;
    ComputePrecision compute_precision() const noexcept;
private:
    std::unique_ptr<detail::StochasticWorkspaceState> state_;
    friend struct detail::Access;
};
template<class T> struct BasicStochasticMomentBlock {
    std::size_t first = 0, count = 0, rows = 0;
    std::ptrdiff_t stride = 0;
    T* spliced = nullptr;
    T* unspliced = nullptr;
    T* spliced_second = nullptr;
    T* cross = nullptr;
    T* unspliced_second = nullptr;
};
using StochasticMomentBlock = BasicStochasticMomentBlock<double>;
using StochasticMomentBlock32 = BasicStochasticMomentBlock<float>;
template<class T> struct BasicStochasticScaleBlock {
    std::size_t first = 0, count = 0;
    T* first_scale = nullptr;
    T* second_scale = nullptr;
    std::uint8_t* status = nullptr;
};
using StochasticScaleBlock = BasicStochasticScaleBlock<double>;
using StochasticScaleBlock32 = BasicStochasticScaleBlock<float>;
template<class T> struct BasicStochasticFitBlock {
    std::size_t first = 0, count = 0;
    T* gamma = nullptr;
    T* offset = nullptr;
    T* offset2 = nullptr;
    T* first_scale = nullptr;
    T* second_scale = nullptr;
    std::uint8_t* status = nullptr;
    T* objective = nullptr;
};
using StochasticFitBlock = BasicStochasticFitBlock<double>;
using StochasticFitBlock32 = BasicStochasticFitBlock<float>;
struct StochasticOutputBlock {
    std::size_t first = 0, count = 0;
    NumericView velocity, second_residual, spliced, unspliced, spliced_second, cross, unspliced_second;
};
struct StochasticBlockSink {
    void* context = nullptr;
    void (*write)(void*, const StochasticOutputBlock&) = nullptr;
};
template<class T> class BasicStochasticReferenceStages {
public:
    BasicStochasticReferenceStages(const PreparedInput&, StochasticWorkspace&, StochasticOptions = {}, StochasticBlockSink = {});
    ~BasicStochasticReferenceStages();
    BasicStochasticReferenceStages(BasicStochasticReferenceStages&&) noexcept;
    BasicStochasticReferenceStages& operator=(BasicStochasticReferenceStages&&) noexcept;
    BasicStochasticReferenceStages(const BasicStochasticReferenceStages&) = delete;
    BasicStochasticReferenceStages& operator=(const BasicStochasticReferenceStages&) = delete;
    BasicMomentBlock<T> initial_moments(std::size_t first, std::size_t count);
    BasicFitBlock<T> initial_fit(BasicMomentBlock<T>);
    void accept_initialization(BasicMomentBlock<T>, BasicFitBlock<T>);
    // Alternate sources must be dense N by count batches; caller owns staging.
    void replace_moments(BasicMomentBlock<T>&, std::size_t offset, std::size_t count, NumericView s, NumericView u);
    // Alternate sources must be dense N by count batches; caller owns staging.
    void replace_moments(BasicStochasticMomentBlock<T>&, std::size_t offset, std::size_t count, NumericView s, NumericView u);
    void resolve_selection();
    BasicStochasticMomentBlock<T> moments(std::size_t first, std::size_t count);
    BasicStochasticScaleBlock<T> scales(BasicStochasticMomentBlock<T>);
    BasicStochasticFitBlock<T> fit(BasicStochasticMomentBlock<T>, BasicStochasticScaleBlock<T>);
    void emit(BasicStochasticMomentBlock<T>, BasicStochasticFitBlock<T>);
    StochasticFitResult finish();
    bool bound_to(const PreparedInput&, const StochasticWorkspace&) const noexcept;
private:
    std::unique_ptr<detail::BasicStochasticStagesState<T>> state_;
    friend struct detail::Access;
};
using StochasticReferenceStages = BasicStochasticReferenceStages<double>;
using StochasticReferenceStages32 = BasicStochasticReferenceStages<float>;

// Both passes accept opaque custom state; only explicit reference calls use host buffers.
template<class Operations>
auto stochastic_with(Operations& operations, std::size_t genes, std::size_t block_genes = 64) {
    if (!block_genes) throw std::invalid_argument("block_genes must be positive");
    if constexpr (requires { operations.begin(); }) operations.begin();
    for (std::size_t first = 0; first < genes;) {
        auto count = genes - first < block_genes ? genes - first : block_genes;
        auto moments = operations.initial_moments(first, count);
        auto fit = operations.initial_fit(moments);
        operations.accept_initialization(moments, fit);
        first += count;
    }
    operations.resolve_selection();
    for (std::size_t first = 0; first < genes;) {
        auto count = genes - first < block_genes ? genes - first : block_genes;
        if constexpr (requires { operations.process_block(first, count); }) operations.process_block(first, count);
        else {
            auto moments = operations.moments(first, count);
            auto scales = operations.scales(moments);
            auto fit = operations.fit(moments, scales);
            operations.emit(moments, fit);
        }
        first += count;
    }
    return operations.finish();
}
namespace reference {
// All moment outputs are optional; source_s/source_u each hold n_cells() doubles.
void second_moments(const PreparedInput&, std::size_t first, std::size_t count,
                    double* spliced, double* unspliced, double* spliced_second, double* cross,
                    double* unspliced_second, double* source_s, double* source_u);
// Per-gene pointers address rows elements at the stated signed stride. The
// caller supplies disjoint score_scratch[rows] for trimmed fitting; pointer
// capacity and scratch nonaliasing are caller preconditions. fit_mask, if set,
// holds rows bytes. Initial diagnostics and supplied scales must be finite.
StochasticScales stochastic_scales(const double* spliced, const double* unspliced,
    const double* spliced_second, const double* cross, std::size_t rows, std::ptrdiff_t stride,
    const GeneFit&, const StochasticOptions&, double* score_scratch, const std::uint8_t* fit_mask = nullptr);
StochasticGeneFit stochastic_fit_gene(const double* spliced, const double* unspliced,
    const double* spliced_second, const double* cross, std::size_t rows, std::ptrdiff_t stride,
    const GeneFit&, StochasticScales, const StochasticOptions&, double* score_scratch,
    const std::uint8_t* fit_mask = nullptr);
void stochastic_residuals(const StochasticMomentBlock&, const StochasticFitBlock&,
                          NumericEdit velocity, std::optional<NumericEdit> second_residual = {});
StochasticFitResult stochastic(const PreparedInput&, StochasticWorkspace&, StochasticOptions = {}, StochasticBlockSink = {});
StochasticFitResult stochastic_into(const PreparedInput&, StochasticWorkspace&, NumericEdit velocity,
    StochasticOptions = {}, std::optional<NumericEdit> second_residual = {});

void second_moments(const PreparedInput&, std::size_t first, std::size_t count,
                    float* spliced, float* unspliced, float* spliced_second, float* cross,
                    float* unspliced_second, float* source_s, float* source_u);
StochasticScales32 stochastic_scales(const float* spliced, const float* unspliced,
    const float* spliced_second, const float* cross, std::size_t rows, std::ptrdiff_t stride,
    const GeneFit32&, const StochasticOptions&, float* score_scratch, const std::uint8_t* fit_mask = nullptr);
StochasticGeneFit32 stochastic_fit_gene(const float* spliced, const float* unspliced,
    const float* spliced_second, const float* cross, std::size_t rows, std::ptrdiff_t stride,
    const GeneFit32&, StochasticScales32, const StochasticOptions&, float* score_scratch,
    const std::uint8_t* fit_mask = nullptr);
void stochastic_residuals(const StochasticMomentBlock32&, const StochasticFitBlock32&,
                          NumericEdit velocity, std::optional<NumericEdit> second_residual = {});
// Alternate sources are N by K batches in caller-supplied column order. The
// prepared graph is reused; sparse source indexing is cold caller-owned
// acquisition outside a Workspace budget. Stage replacement requires dense input.
template<class T> void moments_from(const PreparedInput&, NumericView s, NumericView u,
    T* ms, T* mu, std::size_t row_stride);
template<class T> void second_moments_from(const PreparedInput&, NumericView s, NumericView u,
    T* ms, T* mu, T* mss, T* mus, T* muu, T* source_s, T* source_u, std::size_t row_stride);

}

}

export namespace velocity::cpu {
// Compiled portable CPU helpers; explicitly callable when other paths are selected.
FitResult steady_state(const PreparedInput&, Workspace&, FitOptions = {}, BlockSink = {});
StochasticFitResult stochastic(const PreparedInput&, StochasticWorkspace&, StochasticOptions = {}, StochasticBlockSink = {});
}
