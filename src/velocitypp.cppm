module;
#include <stddef.h>
#include <stdint.h>

export module velocitypp;
export import anndatapp.core;

export namespace velocitypp {

enum class Code : uint8_t { ok, argument, shape, dtype, layout, capacity, overflow, nonfinite, bounds, unfinished };
struct Status {
    Code code = Code::ok;
    size_t location = 0, required = 0;
    explicit operator bool() const noexcept { return code == Code::ok; }
};
enum class ComputePrecision : uint8_t { float32, float64, automatic };
enum class FitStatus : uint8_t { ok, insufficient_data, no_signal, singular, constant_response };
enum class LowSignal { keep_filters, scvelo_percentile };
enum class Trimming { none, upper, tails };
struct FitOptions {
    Trimming trimming = Trimming::tails;
    double percentile_low = 5, percentile_high = 95;
    bool fit_intercept = false, positive_intercept = true, r2_adjusted = false;
    double ratio_min = -__builtin_huge_val(), ratio_max = __builtin_huge_val();
    double min_r2 = .01, min_ratio = .01;
    LowSignal low_signal = LowSignal::keep_filters;
};
struct StochasticOptions {
    FitOptions initialization;
    bool fit_second_intercept = false, refit_all_if_none = false;
};
struct RunOptions { size_t workspace_bytes = 64*1024*1024, block_genes = 64; bool unspliced_second = false; };
using StochasticRunOptions = RunOptions;

// The caller acquires and retains these AnnData++ descriptors before a call.
// This binding stores only existing descriptor addresses and their dtype tags.
using DenseInput = decltype(((anndatapp::ArrayView*)nullptr)->bytes());
using SparseInput = decltype(((anndatapp::ArrayView*)nullptr)->sparse_bytes());
struct AcquiredMatrix {
    const DenseInput* dense = nullptr;
    const SparseInput* sparse = nullptr;
    anndatapp::DType dtype = anndatapp::DType::float32;
    anndatapp::DType index_type = anndatapp::DType::int32, offset_type = anndatapp::DType::int64;
    anndatapp::Layout layout = anndatapp::Layout::dense;
};
// Masks/selection are borrowed; null genes means all source genes.
// A nonnull empty selection is valid.
struct Input {
    AcquiredMatrix spliced, unspliced, weights;
    const size_t* genes = nullptr;
    size_t gene_count = 0;
    const uint8_t* fit_mask = nullptr;
    const uint8_t* gene_mask = nullptr;
    ptrdiff_t fit_mask_stride = 1, gene_mask_stride = 1;
};
struct Entry { size_t index = 0, position = 0; };
struct ColumnIndex { size_t* offsets = nullptr; Entry* entries = nullptr; };
struct GraphIndex { size_t* rows = nullptr; size_t* columns = nullptr; Entry* row_edges = nullptr; Entry* column_edges = nullptr; size_t edges = 0; };
struct PreparationSize { size_t indices = 0, entries = 0; };
struct PreparationBuffers {
    size_t* indices = nullptr; size_t index_capacity = 0;
    Entry* entries = nullptr; size_t entry_capacity = 0;
};
struct PreparedInput {
    Input input;
    size_t rows = 0, columns = 0, genes = 0;
    ColumnIndex spliced, unspliced;
    GraphIndex graph;
};
Status validate(const Input&);
Status preparation_size(const Input&, PreparationSize&);
Status prepare(const Input&, const PreparationBuffers&, PreparedInput&);
// Trusted preparation omits scans for values/ranges, but still reports capacity.
Status prepare_unchecked(const Input&, const PreparationBuffers&, PreparedInput&);

struct WorkspaceSize { size_t scalars = 0, masks = 0, block = 0, bytes = 0; };
template<class T> struct Workspace {
    T* ms = nullptr; T* mu = nullptr; T* score = nullptr;
    T* mss = nullptr; T* mus = nullptr; T* muu = nullptr;
    T* source_s = nullptr; T* source_u = nullptr;
    uint8_t* mask = nullptr;
    size_t rows = 0, block = 0, scratch_bytes = 0;
    bool stochastic = false;
};
using Workspace64 = Workspace<double>;
using Workspace32 = Workspace<float>;
template<class T> Status workspace_size(const PreparedInput&, RunOptions, bool stochastic, WorkspaceSize&);
template<class T> Status bind_workspace(const PreparedInput&, RunOptions, bool stochastic,
    T* storage, size_t scalar_capacity, uint8_t* mask, size_t mask_capacity, Workspace<T>&);

template<class T> struct BasicGeneFit { T gamma = 0, offset = 0, r2 = 0; FitStatus status = FitStatus::insufficient_data; bool selected = false; };
using GeneFit = BasicGeneFit<double>;
using GeneFit32 = BasicGeneFit<float>;
// Result arrays are pairwise disjoint and disjoint from input, topology,
// workspace and numerical outputs. Binding checks capacities, not provenance.
template<class T> struct BasicFitResult {
    T* gamma = nullptr; T* offset = nullptr; T* r2 = nullptr;
    bool* genes = nullptr; uint8_t* status = nullptr;
    size_t capacity = 0;
    bool low_signal = false;
};
using FitResult = BasicFitResult<double>;
using FitResult32 = BasicFitResult<float>;
template<class T> struct BasicMomentBlock {
    size_t first = 0, count = 0, rows = 0;
    ptrdiff_t stride = 0;
    T* spliced = nullptr; T* unspliced = nullptr;
};
using MomentBlock = BasicMomentBlock<double>;
using MomentBlock32 = BasicMomentBlock<float>;
template<class T> struct BasicFitBlock {
    size_t first = 0, count = 0;
    T* gamma = nullptr; T* offset = nullptr; T* r2 = nullptr;
    bool* selected = nullptr; uint8_t* status = nullptr;
};
using FitBlock = BasicFitBlock<double>;
using FitBlock32 = BasicFitBlock<float>;
template<class T> struct BasicBlockSink { void* context = nullptr; void (*write)(void*, const BasicMomentBlock<T>&, const BasicFitBlock<T>&) = nullptr; };
using BlockSink = BasicBlockSink<double>;
using BlockSink32 = BasicBlockSink<float>;
template<class T> struct BasicReferenceStages {
    const PreparedInput* input = nullptr;
    Workspace<T>* workspace = nullptr;
    FitOptions options;
    BasicFitResult<T> result;
    BasicBlockSink<T> sink;
    size_t next = 0;
    bool finished = false;
    BasicMomentBlock<T> moments(size_t first, size_t count);
    BasicFitBlock<T> fit(BasicMomentBlock<T>);
    void emit(BasicMomentBlock<T>, BasicFitBlock<T>);
    BasicFitResult<T> finish();
};
using ReferenceStages = BasicReferenceStages<double>;
using ReferenceStages32 = BasicReferenceStages<float>;
template<class T> Status bind_stages(const PreparedInput&, Workspace<T>&, FitOptions,
    BasicFitResult<T>, BasicReferenceStages<T>&, BasicBlockSink<T> = {});

struct RuntimeOperations;
// Whole-region replacement precedes reference scheduling. Concrete intermediates
// and results belong to Operations; the reference's host layout is optional.
// Trusted schedule precondition: block > 0 unless Operations::run bypasses it.
template<class Operations> requires (!__is_same(Operations,RuntimeOperations) && !__is_same(Operations,const RuntimeOperations))
auto steady_state_with(Operations& o, size_t genes, size_t block = 64) {
    if constexpr (requires { o.run(); }) return o.run();
    else {
        if constexpr (requires { o.begin(); }) o.begin();
        for (size_t first = 0; first < genes;) {
            size_t count = genes-first < block ? genes-first : block;
            if constexpr (requires { o.process_block(first,count); }) o.process_block(first,count);
            else { auto m=o.moments(first,count); auto f=o.fit(m); o.emit(m,f); }
            first += count;
        }
        return o.finish();
    }
}
// Preconditions: block > 0; callbacks used by the selected region are nonnull.
// State/results remain entirely caller-owned, including novel device layouts.
struct RuntimeOperations {
    void* context = nullptr;
    Status (*run)(void*) = nullptr;
    Status (*begin)(void*) = nullptr;
    Status (*initialize)(void*) = nullptr;
    Status (*initial_block)(void*,size_t,size_t) = nullptr;
    Status (*resolve)(void*) = nullptr;
    Status (*moments)(void*,size_t,size_t) = nullptr;
    Status (*scales)(void*,size_t,size_t) = nullptr;
    Status (*fit)(void*,size_t,size_t) = nullptr;
    Status (*emit)(void*,size_t,size_t) = nullptr;
    Status (*process_block)(void*,size_t,size_t) = nullptr;
    Status (*finish)(void*) = nullptr;
};
Status steady_state_with(const RuntimeOperations&, size_t genes, size_t block = 64);

enum class StochasticStatus : uint8_t { ok, not_selected, invalid_initialization, insufficient_data, zero_scale, singular, nonfinite };
template<class T> struct BasicStochasticScales { T first = 0, second = 0; StochasticStatus status = StochasticStatus::not_selected; };
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
template<class T> struct BasicStochasticFitResult {
    BasicFitResult<T> initialization;
    T* gamma = nullptr; T* offset = nullptr; T* offset2 = nullptr;
    T* first_scale = nullptr; T* second_scale = nullptr; T* objective = nullptr;
    uint8_t* status = nullptr;
    size_t capacity = 0;
    bool refit_all = false;
};
using StochasticFitResult = BasicStochasticFitResult<double>;
using StochasticFitResult32 = BasicStochasticFitResult<float>;
template<class T> struct BasicStochasticMomentBlock {
    size_t first = 0, count = 0, rows = 0;
    ptrdiff_t stride = 0;
    T* spliced = nullptr; T* unspliced = nullptr; T* spliced_second = nullptr;
    T* cross = nullptr; T* unspliced_second = nullptr;
};
using StochasticMomentBlock = BasicStochasticMomentBlock<double>;
using StochasticMomentBlock32 = BasicStochasticMomentBlock<float>;
template<class T> struct BasicStochasticScaleBlock { size_t first=0,count=0; T* first_scale=nullptr; T* second_scale=nullptr; uint8_t* status=nullptr; };
template<class T> struct BasicStochasticFitBlock {
    size_t first=0,count=0;
    T* gamma=nullptr; T* offset=nullptr; T* offset2=nullptr;
    T* first_scale=nullptr; T* second_scale=nullptr; uint8_t* status=nullptr; T* objective=nullptr;
};
using StochasticScaleBlock=BasicStochasticScaleBlock<double>;
using StochasticScaleBlock32=BasicStochasticScaleBlock<float>;
using StochasticFitBlock=BasicStochasticFitBlock<double>;
using StochasticFitBlock32=BasicStochasticFitBlock<float>;
template<class T> struct BasicStochasticBlockSink { void* context=nullptr; void (*write)(void*,const BasicStochasticMomentBlock<T>&,const BasicStochasticFitBlock<T>&)=nullptr; };
using StochasticBlockSink=BasicStochasticBlockSink<double>;
template<class T> struct BasicStochasticReferenceStages {
    const PreparedInput* input=nullptr;
    Workspace<T>* workspace=nullptr;
    StochasticOptions options;
    BasicStochasticFitResult<T> result;
    BasicStochasticBlockSink<T> sink;
    size_t next=0;
    bool resolved=false,finished=false;
    BasicMomentBlock<T> initial_moments(size_t,size_t);
    BasicFitBlock<T> initial_fit(BasicMomentBlock<T>);
    void accept_initialization(BasicMomentBlock<T>,BasicFitBlock<T>);
    void resolve_selection();
    BasicStochasticMomentBlock<T> moments(size_t,size_t);
    BasicStochasticScaleBlock<T> scales(BasicStochasticMomentBlock<T>);
    BasicStochasticFitBlock<T> fit(BasicStochasticMomentBlock<T>,BasicStochasticScaleBlock<T>);
    void emit(BasicStochasticMomentBlock<T>,BasicStochasticFitBlock<T>);
    BasicStochasticFitResult<T> finish();
};
using StochasticReferenceStages=BasicStochasticReferenceStages<double>;
using StochasticReferenceStages32=BasicStochasticReferenceStages<float>;
template<class T> Status bind_stages(const PreparedInput&,Workspace<T>&,StochasticOptions,
    BasicStochasticFitResult<T>,BasicStochasticReferenceStages<T>&,BasicStochasticBlockSink<T> = {});
// Trusted schedule precondition: block > 0 unless Operations::run bypasses it.
template<class Operations> requires (!__is_same(Operations,RuntimeOperations) && !__is_same(Operations,const RuntimeOperations))
auto stochastic_with(Operations& o,size_t genes,size_t block=64) {
    if constexpr (requires { o.run(); }) return o.run();
    else {
        if constexpr (requires { o.begin(); }) o.begin();
        if constexpr (requires { o.initialize(); }) o.initialize();
        else {
            for (size_t first=0;first<genes;) { size_t count=genes-first<block?genes-first:block;
                auto m=o.initial_moments(first,count);auto f=o.initial_fit(m);o.accept_initialization(m,f);first+=count; }
            o.resolve_selection();
        }
        for(size_t first=0;first<genes;) {size_t count=genes-first<block?genes-first:block;
            if constexpr(requires{o.process_block(first,count);})o.process_block(first,count);
            else {auto m=o.moments(first,count);auto s=o.scales(m);auto f=o.fit(m,s);o.emit(m,f);}first+=count;}
        return o.finish();
    }
}
Status stochastic_with(const RuntimeOperations&,size_t genes,size_t block=64);

namespace reference {
Status validate(const FitOptions&);
Status validate(const StochasticOptions&);
// Raw per-gene routines are trusted: signed strides, score_scratch[rows],
// masks[rows], nonoverlap, finite options and enabled inputs are caller contracts.
GeneFit fit_gene(const double*,const double*,size_t,ptrdiff_t,const FitOptions&,double*,const uint8_t* = nullptr,bool = true);
GeneFit32 fit_gene(const float*,const float*,size_t,ptrdiff_t,const FitOptions&,float*,const uint8_t* = nullptr,bool = true);
StochasticScales stochastic_scales(const double*,const double*,const double*,const double*,size_t,ptrdiff_t,const GeneFit&,const StochasticOptions&,double*,const uint8_t* = nullptr);
StochasticScales32 stochastic_scales(const float*,const float*,const float*,const float*,size_t,ptrdiff_t,const GeneFit32&,const StochasticOptions&,float*,const uint8_t* = nullptr);
StochasticGeneFit stochastic_fit_gene(const double*,const double*,const double*,const double*,size_t,ptrdiff_t,const GeneFit&,StochasticScales,const StochasticOptions&,double*,const uint8_t* = nullptr);
StochasticGeneFit32 stochastic_fit_gene(const float*,const float*,const float*,const float*,size_t,ptrdiff_t,const GeneFit32&,StochasticScales32,const StochasticOptions&,float*,const uint8_t* = nullptr);
template<class T> void moments(const PreparedInput&,size_t,size_t,T*,T*);
template<class T> void second_moments(const PreparedInput&,size_t,size_t,T*,T*,T*,T*,T*,T*,T*);
template<class T> void moments_from(const PreparedInput&,const PreparedInput&,size_t,size_t,T*,T*);
template<class T> void moments_from(const PreparedInput&,const void*,ptrdiff_t,anndatapp::DType,const void*,ptrdiff_t,anndatapp::DType,T*,T*,size_t);
template<class T> void second_moments_from(const PreparedInput&,const void*,ptrdiff_t,anndatapp::DType,const void*,ptrdiff_t,anndatapp::DType,T*,T*,T*,T*,T*,size_t);
template<class T,class O> void residuals(const BasicMomentBlock<T>&,const BasicFitBlock<T>&,O*,ptrdiff_t,ptrdiff_t=1);
template<class T,class O> void stochastic_residuals(const BasicStochasticMomentBlock<T>&,const BasicStochasticFitBlock<T>&,O*,O*,ptrdiff_t,ptrdiff_t=1);
template<class T> Status steady_state(const PreparedInput&,Workspace<T>&,BasicFitResult<T>&,FitOptions = {},BasicBlockSink<T> = {});
// Output capacities are element spans including signed-stride gaps. The pointer
// addresses logical row zero. Output regions are disjoint from all other buffers.
template<class T,class O> Status steady_state_into(const PreparedInput&,Workspace<T>&,BasicFitResult<T>&,O*,size_t,ptrdiff_t,FitOptions = {});
template<class T> Status stochastic(const PreparedInput&,Workspace<T>&,BasicStochasticFitResult<T>&,StochasticOptions = {},BasicStochasticBlockSink<T> = {});
template<class T,class O> Status stochastic_into(const PreparedInput&,Workspace<T>&,BasicStochasticFitResult<T>&,O*,size_t,O*,size_t,ptrdiff_t,StochasticOptions = {});
struct NeighborhoodSize { size_t entries=0, indices=0, scratch_entries=0; };
struct NeighborhoodBuffers { size_t* indices=nullptr;size_t index_capacity=0;Entry* scratch=nullptr;size_t scratch_capacity=0;float* values=nullptr;size_t* columns=nullptr;size_t* offsets=nullptr;size_t capacity=0,offset_capacity=0; };
Status neighborhood_size(const AcquiredMatrix&,NeighborhoodSize&);
Status prepare_neighborhood(const AcquiredMatrix&,const NeighborhoodBuffers&,size_t& used);
}
namespace cpu {
template<class T> Status steady_state(const PreparedInput&,Workspace<T>&,BasicFitResult<T>&,FitOptions = {},BasicBlockSink<T> = {});
template<class T> Status stochastic(const PreparedInput&,Workspace<T>&,BasicStochasticFitResult<T>&,StochasticOptions = {},BasicStochasticBlockSink<T> = {});
}
}
