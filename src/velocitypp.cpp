module;
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>
#include <type_traits>
#include "detail/cpu_kernels.hh"

module velocitypp;
import anndatapp.core;
#include "detail/prepared.hh"
#include "detail/stochastic_fit.hh"

namespace velocitypp {
namespace detail {

std::size_t add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) throw std::invalid_argument("size overflow");
    return a + b;
}
std::size_t mul(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max()/a) throw std::invalid_argument("size overflow");
    return a*b;
}

template<class O,class T> O output_cast(T value) {
    return O(value);
}
template<class F> void output_dispatch(anndatapp::DType type,F&& f) {
    if(type==anndatapp::DType::float32) f.template operator()<float>();
    else f.template operator()<double>();
}
struct Mask {
    std::optional<anndatapp::ArrayView> view;
    const std::byte* data = nullptr;
    std::ptrdiff_t stride = 0;
    anndatapp::DType dtype = anndatapp::DType::uint8;
    Mask() = default;
    Mask(std::optional<anndatapp::ArrayView> v, std::size_t size) : view(std::move(v)) {
        if (!view) return;
        view->validate();
        if (view->rank() != 1 || view->extent(0) != size || view->layout() != anndatapp::Layout::dense ||
            (view->dtype() != anndatapp::DType::boolean && view->dtype() != anndatapp::DType::uint8))
            throw std::invalid_argument("mask must be a dense boolean or uint8 vector of the matching length");
        auto b = view->bytes(); data = b.data(); stride = b.stride_bytes(0); dtype = view->dtype();
    }
    bool get(std::size_t i) const {
        if (!view) return true;
        auto p = data + std::ptrdiff_t(i)*stride;
        if (dtype == anndatapp::DType::boolean) { bool x; std::memcpy(&x, p, 1); return x; }
        return std::to_integer<unsigned char>(*p) != 0;
    }
};

struct PreparedState {
    PreparedInputs input;
    anndatapp::ArrayView weights, spliced, unspliced;
    std::vector<std::size_t> genes;
    Mask fit_mask, gene_mask;
    anndatapp::DType output_type;
    bool checked;
    PreparedState(Input in, anndatapp::Validation validation) : input(in.weights, in.spliced, in.unspliced),
        weights(in.weights), spliced(in.spliced), unspliced(in.unspliced),
        fit_mask(std::move(in.fit_mask), input.n_obs()), gene_mask(std::move(in.gene_mask), input.n_vars()),
        output_type(in.spliced.dtype() == anndatapp::DType::float64 || in.unspliced.dtype() == anndatapp::DType::float64
                    ? anndatapp::DType::float64 : anndatapp::DType::float32), checked(validation == anndatapp::Validation::full) {
        // Selection validates Axis arithmetic and bounds before unchecked index().
        auto selected = in.spliced.select({anndatapp::Axis::all(), in.genes});
        auto n = selected.extent(1);
        genes.resize(n);
        for (std::size_t i = 0; i < n; ++i) genes[i] = in.genes.index(i);
        if (checked) input.check_values();
    }
};

template<class T> struct BasicWorkspaceState {
    const PreparedState* owner;
    RunOptions options;
    anndatapp::DType output_type;
    std::size_t n, genes, block, scratch, width, slots;
    std::vector<T> ms, mu, score;
    std::vector<double> out64;
    bool compiled = false;
    std::vector<float> out32;
    std::vector<std::uint8_t> mask;
    std::vector<anndatapp::ArrayView> views;

    BasicWorkspaceState(const PreparedInput& p, const PreparedState& s, RunOptions o)
        : owner(&s), options(o), output_type(o.output_dtype.value_or(std::is_same_v<T,float> ? anndatapp::DType::float32 : s.output_type)),
          n(s.input.n_obs()), genes(s.genes.size()), block(0), scratch(0), width(0), slots(0) {
        (void)p;
        if (output_type != anndatapp::DType::float32 && output_type != anndatapp::DType::float64)
            throw std::invalid_argument("output dtype must be float32 or float64");
        if (!o.block_genes) throw std::invalid_argument("block_genes must be positive");
        width = output_type == anndatapp::DType::float32 ? sizeof(float) : sizeof(double);
        slots = std::size_t(o.velocity) + std::size_t(o.spliced_moments) + std::size_t(o.unspliced_moments);
        auto fixed = add(mul(std::max(n, genes), sizeof(T)), n);
        auto per_gene = add(mul(n, add(2*sizeof(T), mul(slots, width))), sizeof(std::size_t));
        if (fixed > o.workspace_bytes || (genes && per_gene > o.workspace_bytes - fixed))
            throw std::invalid_argument("workspace budget cannot hold one gene");
        auto limit = std::min<std::size_t>({o.block_genes, 64, std::max<std::size_t>(genes, 1)});
        block = per_gene ? std::min(limit, (o.workspace_bytes - fixed)/per_gene) : limit;
        if (!block) throw std::invalid_argument("workspace budget cannot hold one gene");
        scratch = add(fixed, mul(per_gene, block));
        auto elements = mul(n, block);
        ms.resize(elements); mu.resize(elements); score.resize(std::max(n, genes)); mask.resize(n);
        if (width == sizeof(float)) out32.resize(mul(elements, slots));
        else out64.resize(mul(elements, slots));
        scratch = add(add(mul(add(ms.capacity(), add(mu.capacity(), score.capacity())), sizeof(T)), mask.capacity()),
                      add(mul(out32.capacity(), sizeof(float)), mul(out64.capacity(), sizeof(double))));
        if (scratch > o.workspace_bytes) throw std::invalid_argument("allocated workspace exceeds budget");
        views.reserve(mul(slots, block + 1));
        for (std::size_t slot = 0; slot < slots; ++slot) {
            views.emplace_back();
            for (std::size_t count = 1; count <= block; ++count) {
                auto off = mul(slot, elements);
                if (width == sizeof(float)) views.push_back(anndatapp::Array::borrow_dense(n ? out32.data() + off : nullptr, {n, count}, {std::ptrdiff_t(count), 1}).view());
                else views.push_back(anndatapp::Array::borrow_dense(n ? out64.data() + off : nullptr, {n, count}, {std::ptrdiff_t(count), 1}).view());
            }
        }
    }
    anndatapp::ArrayView view(std::size_t slot, std::size_t count) const { return views[slot*(block + 1) + count]; }
};

template<class T> struct BasicStagesState {
    const PreparedState* p;
    BasicWorkspaceState<T>* w;
    FitOptions options;
    BlockSink sink;
    FitResult result;
    T *gamma, *offset, *r2;
    bool* selected;
    std::uint8_t* status;
    std::size_t next = 0;
    bool finished = false;
    BasicStagesState(const PreparedState& ps, BasicWorkspaceState<T>& ws, FitOptions o, BlockSink b)
        : p(&ps), w(&ws), options(o), sink(b),
          result{anndatapp::Array::empty<T>({ps.genes.size()}), anndatapp::Array::empty<T>({ps.genes.size()}),
                 anndatapp::Array::empty<T>({ps.genes.size()}), anndatapp::Array::empty<bool>({ps.genes.size()}),
                 anndatapp::Array::empty<std::uint8_t>({ps.genes.size()})} {
        gamma = result.gamma.edit().dense<T>().data(); offset = result.offset.edit().dense<T>().data();
        r2 = result.r2.edit().dense<T>().data(); selected = result.genes.edit().dense<bool>().data();
        status = result.status.edit().dense<std::uint8_t>().data();
        T validation_scratch=0;
        reference::fit_gene(static_cast<const T*>(nullptr),static_cast<const T*>(nullptr),0,1,o,&validation_scratch);
        if (!ps.genes.empty()) {
            std::fill_n(gamma, ps.genes.size(), 0); std::fill_n(offset, ps.genes.size(), 0);
            std::fill_n(r2, ps.genes.size(), 0); std::fill_n(selected, ps.genes.size(), false);
            std::fill_n(status, ps.genes.size(), std::uint8_t(FitStatus::insufficient_data));
        }
        w->compiled = false;
        if (p->checked) p->input.check_values();
        for (std::size_t i = 0; i < w->n; ++i) w->mask[i] = std::uint8_t(p->fit_mask.get(i));
    }
};

template<class T> struct BasicStochasticWorkspaceState {
    const PreparedState* owner;
    StochasticRunOptions options;
    anndatapp::DType dtype;
    std::size_t n, block, bytes, slots, elements;
    std::vector<T> ms, mu, mss, mus, muu, source_s, source_u, score;
    std::vector<double> out64;
    bool compiled = false;
    std::vector<float> out32;
    std::vector<std::uint8_t> mask;
    std::vector<anndatapp::ArrayView> views;
    BasicStochasticWorkspaceState(const PreparedState& p, StochasticRunOptions o)
        : owner(&p), options(o), dtype(o.output_dtype.value_or(std::is_same_v<T,float> ? anndatapp::DType::float32 : p.output_type)), n(p.input.n_obs()) {
        if (!o.block_genes || (dtype != anndatapp::DType::float32 && dtype != anndatapp::DType::float64))
            throw std::invalid_argument("invalid stochastic workspace options");
        slots = std::size_t(o.velocity) + o.second_residual + o.spliced_moments + o.unspliced_moments
              + o.spliced_second + o.cross + o.unspliced_second;
        auto width = dtype == anndatapp::DType::float32 ? sizeof(float) : sizeof(double);
        auto fixed = add(mul(n, 3*sizeof(T)), n);
        auto per_gene = mul(n, add(mul(4 + std::size_t(o.unspliced_second), sizeof(T)), mul(slots, width)));
        if (fixed > o.workspace_bytes || (!p.genes.empty() && per_gene > o.workspace_bytes - fixed))
            throw std::invalid_argument("stochastic workspace budget cannot hold one gene");
        block = std::min<std::size_t>({o.block_genes, 64, std::max<std::size_t>(1, p.genes.size())});
        if (per_gene) block = std::min(block, (o.workspace_bytes - fixed)/per_gene);
        if (!block) throw std::invalid_argument("stochastic workspace budget cannot hold one gene");
        elements = mul(n, block);
        ms.resize(elements); mu.resize(elements); mss.resize(elements); mus.resize(elements);
        if (o.unspliced_second) muu.resize(elements);
        source_s.resize(n); source_u.resize(n); score.resize(n); mask.resize(n);
        if (width == sizeof(float)) out32.resize(mul(elements, slots)); else out64.resize(mul(elements, slots));
        bytes = add(mask.capacity(), add(mul(out32.capacity(), sizeof(float)), mul(out64.capacity(),sizeof(double))));
        for (auto* v : {&ms, &mu, &mss, &mus, &muu, &source_s, &source_u, &score}) bytes = add(bytes, mul(v->capacity(), sizeof(T)));
        if (bytes > o.workspace_bytes) throw std::invalid_argument("allocated stochastic workspace exceeds budget");
        views.reserve(mul(slots, block + 1));
        for (std::size_t slot = 0; slot < slots; ++slot) {
            views.emplace_back();
            for (std::size_t count = 1; count <= block; ++count) {
                auto off = slot*elements;
                if (width == sizeof(float)) views.push_back(anndatapp::Array::borrow_dense(n ? out32.data() + off : nullptr, {n,count}, {std::ptrdiff_t(count),1}).view());
                else views.push_back(anndatapp::Array::borrow_dense(n ? out64.data() + off : nullptr, {n,count}, {std::ptrdiff_t(count),1}).view());
            }
        }
    }
    anndatapp::ArrayView view(std::size_t slot, std::size_t count) const { return views[slot*(block + 1) + count]; }
};
template<class T> struct BasicStochasticStagesState {
    const PreparedState* p;
    BasicStochasticWorkspaceState<T>* w;
    StochasticOptions options;
    StochasticBlockSink sink;
    StochasticFitResult result;
    T *initial_gamma, *initial_offset, *r2, *gamma, *offset, *offset2, *scale1, *scale2, *objective;
    bool* selected;
    std::uint8_t *initial_status, *status;
    std::size_t next = 0;
    bool resolved = false, finished = false;
    BasicStochasticStagesState(const PreparedState& ps, BasicStochasticWorkspaceState<T>& ws, StochasticOptions o, StochasticBlockSink b)
        : p(&ps), w(&ws), options(o), sink(b) {
        auto g = p->genes.size();
        auto doubles = [g] { return anndatapp::Array::zeros<T>({g}); };
        result.initialization = {doubles(), doubles(), doubles(), anndatapp::Array::zeros<bool>({g}), anndatapp::Array::zeros<std::uint8_t>({g})};
        result.gamma = doubles(); result.offset = doubles(); result.offset2 = doubles(); result.first_scale = doubles();
        result.second_scale = doubles(); result.objective = doubles(); result.status = anndatapp::Array::zeros<std::uint8_t>({g});
        auto ptr = [](anndatapp::Array& a) { return a.edit().dense<T>().data(); };
        initial_gamma = ptr(result.initialization.gamma); initial_offset = ptr(result.initialization.offset); r2 = ptr(result.initialization.r2);
        gamma = ptr(result.gamma); offset = ptr(result.offset); offset2 = ptr(result.offset2);
        scale1 = ptr(result.first_scale); scale2 = ptr(result.second_scale); objective = ptr(result.objective);
        selected = result.initialization.genes.edit().dense<bool>().data();
        initial_status = result.initialization.status.edit().dense<std::uint8_t>().data();
        status = result.status.edit().dense<std::uint8_t>().data();
        std::fill_n(initial_status, g, std::uint8_t(FitStatus::insufficient_data));
        std::fill_n(status, g, std::uint8_t(StochasticStatus::not_selected));
        // Validate option contracts even for empty inputs, before any numerical work.
        T dummy = 0;
        reference::fit_gene(nullptr, nullptr, 0, 1, o.initialization, &dummy);
        w->compiled = false;
        if (p->checked) p->input.check_values();
        for (std::size_t i = 0; i < w->n; ++i) w->mask[i] = std::uint8_t(p->fit_mask.get(i));
    }
    BasicGeneFit<T> initial(std::size_t k) const { return {initial_gamma[k], initial_offset[k], r2[k], FitStatus(initial_status[k]), selected[k]}; }
    bool refine(std::size_t k) const {
        return (selected[k] || result.refit_all) && p->gene_mask.get(p->genes[k]) && initial_status[k] == std::uint8_t(FitStatus::ok);
    }
};
ComputePrecision resolve_precision(const PreparedState& p,ComputePrecision requested) {
    if(requested!=ComputePrecision::automatic)return requested;
    (void)p;
    return ComputePrecision::float64;
}
template<class T> constexpr ComputePrecision precision_of = std::is_same_v<T,float> ? ComputePrecision::float32 : ComputePrecision::float64;
struct WorkspaceState {
    const PreparedState* owner;
    ComputePrecision precision;
    std::size_t block=0,scratch=0;
    std::unique_ptr<BasicWorkspaceState<float>> f32;
    std::unique_ptr<BasicWorkspaceState<double>> f64;
    WorkspaceState(const PreparedInput& p,const PreparedState& s,RunOptions o):owner(&s),precision(resolve_precision(s,o.compute_precision)) {
        if(precision==ComputePrecision::float32){f32=std::make_unique<BasicWorkspaceState<float>>(p,s,o);block=f32->block;scratch=f32->scratch;}
        else if(precision==ComputePrecision::float64){f64=std::make_unique<BasicWorkspaceState<double>>(p,s,o);block=f64->block;scratch=f64->scratch;}
        else throw std::invalid_argument("invalid compute precision");
    }
    template<class T> BasicWorkspaceState<T>& get() {
        if constexpr(std::is_same_v<T,float>){if(f32)return *f32;}else{if(f64)return *f64;}
        throw std::invalid_argument("stage precision does not match workspace");
    }
};
struct StochasticWorkspaceState {
    const PreparedState* owner;
    ComputePrecision precision;
    std::size_t block=0,bytes=0;
    std::unique_ptr<BasicStochasticWorkspaceState<float>> f32;
    std::unique_ptr<BasicStochasticWorkspaceState<double>> f64;
    StochasticWorkspaceState(const PreparedState& p,StochasticRunOptions o):owner(&p),precision(resolve_precision(p,o.compute_precision)) {
        if(precision==ComputePrecision::float32){f32=std::make_unique<BasicStochasticWorkspaceState<float>>(p,o);block=f32->block;bytes=f32->bytes;}
        else if(precision==ComputePrecision::float64){f64=std::make_unique<BasicStochasticWorkspaceState<double>>(p,o);block=f64->block;bytes=f64->bytes;}
        else throw std::invalid_argument("invalid compute precision");
    }
    template<class T> BasicStochasticWorkspaceState<T>& get() {
        if constexpr(std::is_same_v<T,float>){if(f32)return *f32;}else{if(f64)return *f64;}
        throw std::invalid_argument("stage precision does not match workspace");
    }
};
struct Access {
    static const PreparedState* pointer(const PreparedInput& p){return p.state_.get();}
    static const WorkspaceState* pointer(const Workspace& w){return w.state_.get();}
    static const StochasticWorkspaceState* pointer(const StochasticWorkspace& w){return w.state_.get();}
    static const PreparedState& get(const PreparedInput& p){return *p.state_;}
    static WorkspaceState& get(Workspace& w){return *w.state_;}
    static StochasticWorkspaceState& get(StochasticWorkspace& w){return *w.state_;}
    template<class T> static BasicStagesState<T>& get(BasicReferenceStages<T>& s){return *s.state_;}
    template<class T> static BasicStochasticStagesState<T>& get(BasicStochasticReferenceStages<T>& s){return *s.state_;}
};
struct Span { std::uintptr_t begin = 0, end = 0; };
bool overlaps(Span a, Span b) { return a.begin < b.end && b.begin < a.end; }
std::size_t magnitude(std::ptrdiff_t x) {
    return x < 0 ? std::size_t(-(x + 1)) + 1 : std::size_t(x);
}
Span span(const std::byte* data, std::size_t rows, std::size_t cols,
          std::ptrdiff_t row_stride, std::ptrdiff_t col_stride, std::size_t width) {
    if (!rows || !cols) return {};
    std::size_t lo = 0, hi = 0;
    auto axis = [&](std::size_t n, std::ptrdiff_t stride) {
        auto d = mul(n - 1, magnitude(stride));
        if (stride < 0) lo = add(lo, d); else hi = add(hi, d);
    };
    axis(rows, row_stride); axis(cols, col_stride); hi = add(hi, width);
    auto base = reinterpret_cast<std::uintptr_t>(data);
    if (base < lo || hi > std::numeric_limits<std::uintptr_t>::max() - base)
        throw std::invalid_argument("buffer address range overflow");
    return {base - lo, base + hi};
}
Span contiguous_span(const std::byte* p, std::size_t n) {
    if (!n) return {};
    auto begin = reinterpret_cast<std::uintptr_t>(p);
    if (n > std::numeric_limits<std::uintptr_t>::max() - begin) throw std::invalid_argument("buffer address range overflow");
    return {begin, begin + n};
}
template<class View> bool aliases(View v, Span target) {
    if (v.layout() == anndatapp::Layout::dense) {
        auto b = v.bytes();
        auto n = v.extent(0), m = v.rank() == 1 ? 1 : v.extent(1);
        return overlaps(target, span(b.data(), n, m, b.stride_bytes(0), v.rank() == 1 ? 0 : b.stride_bytes(1), b.element_size()));
    }
    auto b = v.sparse_bytes();
    auto major = v.layout() == anndatapp::Layout::csr ? b.rows() : b.columns();
    return overlaps(target, contiguous_span(b.values(), mul(b.nnz(), b.value_size()))) ||
           overlaps(target, contiguous_span(b.indices(), mul(b.nnz(), b.index_size()))) ||
           overlaps(target, contiguous_span(b.offsets(), mul(add(major, 1), b.offset_size())));
}

void check_output_shape(anndatapp::ArrayView v, std::size_t rows, std::size_t cols) {
    v.validate();
    if (v.layout() != anndatapp::Layout::dense || v.rank() != 2 || v.extent(0) != rows || v.extent(1) != cols ||
        (v.dtype() != anndatapp::DType::float32 && v.dtype() != anndatapp::DType::float64))
        throw std::invalid_argument("output must be a matching dense float32 or float64 array");
    auto b = v.bytes(); auto rs = magnitude(b.stride_bytes(0)), cs = magnitude(b.stride_bytes(1));
    if ((rows > 1 && !rs) || (cols > 1 && !cs)) throw std::invalid_argument("output elements overlap");
    if (rows > 1 && cols > 1) {
        auto g = std::gcd(rs, cs);
        if (cs/g < rows && rs/g < cols) throw std::invalid_argument("output elements overlap");
    }
}

void check_input_alias(const PreparedState& p, Span target) {
    if (aliases(p.weights, target) || aliases(p.spliced, target) || aliases(p.unspliced, target) ||
        (p.fit_mask.view && aliases(*p.fit_mask.view, target)) || (p.gene_mask.view && aliases(*p.gene_mask.view, target)))
        throw std::invalid_argument("output aliases an input buffer");
}

template<class T> void check_block(const BasicStagesState<T>& s, const BasicMomentBlock<T>& m) {
    if (s.finished || m.first != s.next || !m.count || m.count > s.w->block ||
        m.count > s.p->genes.size() - s.next || m.rows != s.w->n ||
        (m.rows && (!m.spliced || !m.unspliced || !m.stride)))
        throw std::invalid_argument("invalid or out-of-order moment block");
    if (m.rows) {
        auto limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
        auto step = magnitude(m.stride);
        if (m.count - 1 > limit || (m.rows > 1 && step > (limit - (m.count - 1))/(m.rows - 1)))
            throw std::invalid_argument("moment block stride overflows");
    }
}

template<class T> void check_stochastic_range(const BasicStochasticStagesState<T>& s, std::size_t first, std::size_t count, bool resolved) {
    if (s.finished || s.resolved != resolved || first != s.next || !count || count > s.w->block ||
        first > s.p->genes.size() || count > s.p->genes.size() - first)
        throw std::invalid_argument("invalid or out-of-order stochastic block");
}
template<class T> void check_stochastic_block(const BasicStochasticStagesState<T>& s, const BasicStochasticMomentBlock<T>& m) {
    check_stochastic_range(s, m.first, m.count, true);
    if (m.rows != s.w->n || (m.rows && (!m.spliced || !m.unspliced || !m.spliced_second || !m.cross || !m.stride)))
        throw std::invalid_argument("invalid stochastic moment block");
    auto limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if (m.count - 1 > limit || (m.rows > 1 && magnitude(m.stride) > (limit - (m.count - 1))/(m.rows - 1)))
        throw std::invalid_argument("stochastic stride overflows");
}
}

Input from_anndata(const anndatapp::AnnData& data, std::string_view s, std::string_view u, std::string_view w) {
    return {data.layer(s), data.layer(u), data.obsp(w), anndatapp::Axis::all(), std::nullopt, std::nullopt};
}

PreparedInput::PreparedInput(Input in, anndatapp::Validation validation) : state_(std::make_unique<detail::PreparedState>(std::move(in), validation)) {}
PreparedInput::~PreparedInput() = default;
PreparedInput::PreparedInput(PreparedInput&&) noexcept = default;
PreparedInput& PreparedInput::operator=(PreparedInput&&) noexcept = default;
std::size_t PreparedInput::n_cells() const noexcept { return state_->input.n_obs(); }
std::size_t PreparedInput::n_genes() const noexcept { return state_->genes.size(); }
std::size_t PreparedInput::source_genes() const noexcept { return state_->input.n_vars(); }
std::size_t PreparedInput::gene_index(std::size_t i) const {
    if (i >= n_genes()) throw std::out_of_range("gene index out of range");
    return state_->genes[i];
}
std::size_t PreparedInput::storage_bytes() const noexcept { return state_->input.prepared_bytes() + state_->genes.capacity()*sizeof(std::size_t); }
anndatapp::DType PreparedInput::output_dtype() const noexcept { return state_->output_type; }
void PreparedInput::validate_values() const { state_->input.check_values(); }

Workspace::Workspace(const PreparedInput& p, RunOptions o) : state_(std::make_unique<detail::WorkspaceState>(p, detail::Access::get(p), o)) {}
Workspace::~Workspace() = default;
Workspace::Workspace(Workspace&&) noexcept = default;
Workspace& Workspace::operator=(Workspace&&) noexcept = default;
std::size_t Workspace::block_genes() const noexcept { return state_->block; }
std::size_t Workspace::scratch_bytes() const noexcept { return state_->scratch; }

StochasticWorkspace::StochasticWorkspace(const PreparedInput& p, StochasticRunOptions o)
    : state_(std::make_unique<detail::StochasticWorkspaceState>(detail::Access::get(p), o)) {}
StochasticWorkspace::~StochasticWorkspace() = default;
StochasticWorkspace::StochasticWorkspace(StochasticWorkspace&&) noexcept = default;
StochasticWorkspace& StochasticWorkspace::operator=(StochasticWorkspace&&) noexcept = default;
std::size_t StochasticWorkspace::block_genes() const noexcept { return state_->block; }
std::size_t StochasticWorkspace::scratch_bytes() const noexcept { return state_->bytes; }

ComputePrecision Workspace::compute_precision() const noexcept {return state_->precision;}
ComputePrecision StochasticWorkspace::compute_precision() const noexcept {return state_->precision;}
anndatapp::ArrayView PreparedInput::spliced() const {return state_->spliced;}
anndatapp::ArrayView PreparedInput::unspliced() const {return state_->unspliced;}
anndatapp::ArrayView PreparedInput::weights() const {return state_->weights;}
std::optional<anndatapp::ArrayView> PreparedInput::fit_mask() const {return state_->fit_mask.view;}
std::optional<anndatapp::ArrayView> PreparedInput::gene_mask() const {return state_->gene_mask.view;}
template<class T> BasicReferenceStages<T>::BasicReferenceStages(const PreparedInput& p, Workspace& w, FitOptions o, BlockSink b) {
    auto& ws = detail::Access::get(w).get<T>();
    if (ws.owner != &detail::Access::get(p)) throw std::invalid_argument("workspace belongs to another prepared input");
    if (b.write == nullptr && b.context != nullptr) throw std::invalid_argument("sink callback is null");
    state_ = std::make_unique<detail::BasicStagesState<T>>(detail::Access::get(p), ws, o, b);
}
template<class T> BasicReferenceStages<T>::~BasicReferenceStages() = default;
template<class T> BasicReferenceStages<T>::BasicReferenceStages(BasicReferenceStages<T>&&) noexcept = default;
template<class T> BasicReferenceStages<T>& BasicReferenceStages<T>::operator=(BasicReferenceStages<T>&&) noexcept = default;

template<class T> BasicMomentBlock<T> BasicReferenceStages<T>::moments(std::size_t first, std::size_t count) {
    auto& s = *state_; auto& w = *s.w;
    if (s.finished || first != s.next || !count || count > w.block || first > s.p->genes.size() || count > s.p->genes.size() - first)
        throw std::invalid_argument("invalid or out-of-order moment block");
    s.p->input.moments_selected(s.p->genes.data() + first, count, w.ms.data(), w.mu.data(), w.compiled);
    return {first, count, w.n, std::ptrdiff_t(count), w.ms.data(), w.mu.data()};
}

template<class T> BasicFitBlock<T> BasicReferenceStages<T>::fit(BasicMomentBlock<T> m) {
    auto& s = *state_; auto& w = *s.w;
    detail::check_block(s, m);
    for (std::size_t j = 0; j < m.count; ++j) {
        auto k = m.first + j;
        auto fit = reference::fit_gene(m.spliced ? m.spliced + j : nullptr, m.unspliced ? m.unspliced + j : nullptr, m.rows, m.stride, s.options,
                                       w.score.data(), w.mask.data(), s.p->gene_mask.get(s.p->genes[k]));
        s.gamma[k] = fit.gamma; s.offset[k] = fit.offset; s.r2[k] = fit.r2;
        s.selected[k] = fit.selected; s.status[k] = std::uint8_t(fit.status);
    }
    return {m.first, m.count, s.gamma + m.first, s.offset + m.first, s.r2 + m.first, s.selected + m.first, s.status + m.first};
}

template<class T> void BasicReferenceStages<T>::emit(BasicMomentBlock<T> m, BasicFitBlock<T> fit) {
    auto& s = *state_; auto& w = *s.w;
    detail::check_block(s, m);
    if (fit.first != m.first || fit.count != m.count || !fit.gamma || !fit.offset || !fit.r2 || !fit.selected || !fit.status)
        throw std::invalid_argument("invalid fit block");
    for (std::size_t j = 0; j < m.count; ++j)
        if (!std::isfinite(fit.gamma[j]) || !std::isfinite(fit.offset[j]) || !std::isfinite(fit.r2[j]) ||
            fit.status[j] > std::uint8_t(FitStatus::constant_response))
            throw std::invalid_argument("fit block contains invalid diagnostics");
    for (std::size_t j = 0; j < m.count; ++j) {
        auto k = m.first + j;
        auto gamma = fit.gamma[j], offset = fit.offset[j], r2 = fit.r2[j];
        auto status = fit.status[j]; bool selected = fit.selected[j];
        s.gamma[k] = gamma; s.offset[k] = offset; s.r2[k] = r2; s.status[k] = status;
        s.selected[k] = selected && status == std::uint8_t(FitStatus::ok) && s.p->gene_mask.get(s.p->genes[k]);
    }
    if (s.sink.write) {
        OutputBlock out; out.first = m.first; out.count = m.count;
        std::size_t slot = 0;
        auto copy = [&]<bool Velocity>(const T* src) {
            detail::output_dispatch(w.output_type,[&]<class O>() {
                O* dest;
                if constexpr(std::is_same_v<O,float>) dest=w.out32.data()+slot*w.n*w.block;
                else dest=w.out64.data()+slot*w.n*w.block;
                if constexpr(Velocity && std::is_same_v<O,T>) {
                    if(w.compiled && m.stride==std::ptrdiff_t(m.count)) {
                        detail::cpu::residuals(m.spliced,m.unspliced,s.gamma+m.first,s.offset+m.first,dest,m.rows,m.count);
                        return;
                    }
                }
                for(std::size_t i=0;i<m.rows;++i) for(std::size_t j=0;j<m.count;++j) {
                    auto q=std::ptrdiff_t(i)*m.stride+std::ptrdiff_t(j);
                    T value=Velocity ? m.unspliced[q]-s.gamma[m.first+j]*m.spliced[q]-s.offset[m.first+j] : src[q];
                    dest[i*m.count+j]=detail::output_cast<O>(value);
                }
            });
            return w.view(slot++,m.count);
        };
        if (w.options.velocity) out.velocity = copy.template operator()<true>(nullptr);
        if (w.options.spliced_moments) out.spliced = copy.template operator()<false>(m.spliced);
        if (w.options.unspliced_moments) out.unspliced = copy.template operator()<false>(m.unspliced);
        s.sink.write(s.sink.context, out);
    }
    s.next += m.count;
}

template<class T> FitResult BasicReferenceStages<T>::finish() {
    auto& s = *state_;
    if (s.finished || s.next != s.p->genes.size()) throw std::invalid_argument("reference stages have unfinished blocks");
    s.finished = true;
    auto count = s.p->genes.empty() ? 0 : std::count(s.selected, s.selected + s.p->genes.size(), true);
    if (s.options.low_signal == LowSignal::scvelo_percentile && count < 2) {
        std::size_t n = 0;
        for (std::size_t i = 0; i < s.p->genes.size(); ++i)
            if (s.status[i] == std::uint8_t(FitStatus::ok) && s.p->gene_mask.get(s.p->genes[i])) s.w->score[n++] = s.r2[i];
        if (n) {
            std::sort(s.w->score.begin(), s.w->score.begin() + n);
            auto x = T(.8)*T(n - 1); auto lo = std::size_t(x), hi = std::min(lo + 1, n - 1);
            auto threshold = s.w->score[lo] + (x - T(lo))*(s.w->score[hi] - s.w->score[lo]);
            for (std::size_t i = 0; i < s.p->genes.size(); ++i)
                s.selected[i] = s.status[i] == std::uint8_t(FitStatus::ok) && s.p->gene_mask.get(s.p->genes[i]) && s.r2[i] > threshold;
        }
        s.result.low_signal = true;
    }
    return std::move(s.result);
}

template<class T> BasicStochasticReferenceStages<T>::BasicStochasticReferenceStages(const PreparedInput& p, StochasticWorkspace& w, StochasticOptions o, StochasticBlockSink b) {
    auto& ws = detail::Access::get(w).get<T>();
    if (ws.owner != &detail::Access::get(p)) throw std::invalid_argument("stochastic workspace belongs to another input");
    if (!b.write && b.context) throw std::invalid_argument("stochastic sink callback is null");
    state_ = std::make_unique<detail::BasicStochasticStagesState<T>>(detail::Access::get(p), ws, o, b);
}
template<class T> BasicStochasticReferenceStages<T>::~BasicStochasticReferenceStages() = default;
template<class T> BasicStochasticReferenceStages<T>::BasicStochasticReferenceStages(BasicStochasticReferenceStages<T>&&) noexcept = default;
template<class T> BasicStochasticReferenceStages<T>& BasicStochasticReferenceStages<T>::operator=(BasicStochasticReferenceStages<T>&&) noexcept = default;
template<class T> BasicMomentBlock<T> BasicStochasticReferenceStages<T>::initial_moments(std::size_t first, std::size_t count) {
    auto& s = *state_; auto& w = *s.w;
    detail::check_stochastic_range(s, first, count, false);
    s.p->input.moments_selected(s.p->genes.data() + first, count, w.ms.data(), w.mu.data(), w.compiled);
    return {first, count, w.n, std::ptrdiff_t(count), w.ms.data(), w.mu.data()};
}
template<class T> BasicFitBlock<T> BasicStochasticReferenceStages<T>::initial_fit(BasicMomentBlock<T> m) {
    auto& s = *state_;
    detail::check_stochastic_range(s, m.first, m.count, false);
    if (m.rows != s.w->n || (m.rows && (!m.spliced || !m.unspliced || !m.stride))) throw std::invalid_argument("invalid initialization moments");
    auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if (m.count-1>limit || (m.rows>1 && detail::magnitude(m.stride)>(limit-(m.count-1))/(m.rows-1)))
        throw std::invalid_argument("initialization stride overflows");
    for (std::size_t j = 0; j < m.count; ++j) {
        auto k = m.first + j;
        auto f = reference::fit_gene(m.spliced ? m.spliced+j : nullptr, m.unspliced ? m.unspliced+j : nullptr,
            m.rows, m.stride, s.options.initialization, s.w->score.data(), s.w->mask.data(), s.p->gene_mask.get(s.p->genes[k]));
        s.initial_gamma[k] = f.gamma; s.initial_offset[k] = f.offset; s.r2[k] = f.r2;
        s.selected[k] = f.selected; s.initial_status[k] = std::uint8_t(f.status);
    }
    return {m.first, m.count, s.initial_gamma+m.first, s.initial_offset+m.first, s.r2+m.first, s.selected+m.first, s.initial_status+m.first};
}
template<class T> void BasicStochasticReferenceStages<T>::accept_initialization(BasicMomentBlock<T> m, BasicFitBlock<T> f) {
    auto& s = *state_;
    detail::check_stochastic_range(s, m.first, m.count, false);
    if (f.first != m.first || f.count != m.count || !f.gamma || !f.offset || !f.r2 || !f.selected || !f.status)
        throw std::invalid_argument("invalid stochastic initialization fit");
    for (std::size_t j = 0; j < m.count; ++j) {
        if (!std::isfinite(f.gamma[j]) || !std::isfinite(f.offset[j]) || !std::isfinite(f.r2[j]) || f.status[j] > std::uint8_t(FitStatus::constant_response))
            throw std::invalid_argument("invalid stochastic initialization diagnostics");
        auto k = m.first+j;
        s.initial_gamma[k] = f.gamma[j]; s.initial_offset[k] = f.offset[j]; s.r2[k] = f.r2[j]; s.initial_status[k] = f.status[j];
        s.selected[k] = f.selected[j] && f.status[j] == std::uint8_t(FitStatus::ok) && s.p->gene_mask.get(s.p->genes[k]);
    }
    s.next += m.count;
}
template<class T> void BasicStochasticReferenceStages<T>::resolve_selection() {
    auto& s = *state_; auto g = s.p->genes.size();
    if (s.resolved || s.finished || s.next != g) throw std::invalid_argument("unfinished stochastic initialization");
    auto count = g ? std::count(s.selected, s.selected+g, true) : 0;
    if (s.options.initialization.low_signal == LowSignal::scvelo_percentile && count < 2) {
        // Reuse an output diagnostic array, not G-dependent numerical workspace.
        std::size_t n = 0;
        for (std::size_t k = 0; k < g; ++k)
            if (s.initial_status[k] == std::uint8_t(FitStatus::ok) && s.p->gene_mask.get(s.p->genes[k])) s.objective[n++] = s.r2[k];
        if (n) {
            std::sort(s.objective, s.objective+n);
            T q = T(.8)*T(n-1); auto lo = std::size_t(q), hi = std::min(lo+1,n-1);
            T bound = s.objective[lo]+(q-T(lo))*(s.objective[hi]-s.objective[lo]);
            for (std::size_t k = 0; k < g; ++k)
                s.selected[k] = s.initial_status[k] == std::uint8_t(FitStatus::ok) && s.p->gene_mask.get(s.p->genes[k]) && s.r2[k] > bound;
        }
        s.result.initialization.low_signal = true;
        count = g ? std::count(s.selected,s.selected+g,true) : 0;
    }
    s.result.refit_all = s.options.refit_all_if_none && count == 0;
    for (std::size_t k = 0; k < g; ++k) {
        s.gamma[k] = s.initial_gamma[k]; s.offset[k] = s.initial_offset[k]; s.objective[k] = 0;
        s.status[k] = std::uint8_t(!s.p->gene_mask.get(s.p->genes[k]) || s.initial_status[k] == std::uint8_t(FitStatus::ok) ? StochasticStatus::not_selected : StochasticStatus::invalid_initialization);
    }
    s.resolved = true; s.next = 0;
}
template<class T> BasicStochasticMomentBlock<T> BasicStochasticReferenceStages<T>::moments(std::size_t first, std::size_t count) {
    auto& s = *state_; auto& w = *s.w;
    detail::check_stochastic_range(s, first, count, true);
    // A standalone stage promises genuine moments even when custom fitting
    // chooses a different gene population from the built-in refinement policy.
    s.p->input.second_moments_selected(s.p->genes.data()+first,count,w.ms.data(),w.mu.data(),
        w.mss.data(),w.mus.data(),w.options.unspliced_second ? w.muu.data() : nullptr,
        w.source_s.data(),w.source_u.data(),w.compiled);
    return {first,count,w.n,std::ptrdiff_t(count),w.ms.data(),w.mu.data(),w.mss.data(),w.mus.data(),
            w.options.unspliced_second ? w.muu.data() : nullptr};
}
template<class T> BasicStochasticScaleBlock<T> BasicStochasticReferenceStages<T>::scales(BasicStochasticMomentBlock<T> m) {
    auto& s = *state_; detail::check_stochastic_block(s,m);
    for (std::size_t j=0;j<m.count;++j) {
        auto k=m.first+j;
        if (!s.refine(k)) continue;
        auto r=reference::stochastic_scales(m.spliced ? m.spliced+j : nullptr,m.unspliced ? m.unspliced+j : nullptr,
            m.spliced_second ? m.spliced_second+j : nullptr,m.cross ? m.cross+j : nullptr,m.rows,m.stride,
            s.initial(k),s.options,s.w->score.data(),s.w->mask.data());
        s.scale1[k]=r.first; s.scale2[k]=r.second; s.status[k]=std::uint8_t(r.status);
    }
    return {m.first,m.count,s.scale1+m.first,s.scale2+m.first,s.status+m.first};
}
template<class T> BasicStochasticFitBlock<T> BasicStochasticReferenceStages<T>::fit(BasicStochasticMomentBlock<T> m,BasicStochasticScaleBlock<T> scales) {
    auto& s=*state_; detail::check_stochastic_block(s,m);
    if (scales.first!=m.first || scales.count!=m.count || !scales.first_scale || !scales.second_scale || !scales.status)
        throw std::invalid_argument("invalid stochastic scale block");
    for (std::size_t j=0;j<m.count;++j) {
        auto k=m.first+j;
        if (!s.refine(k)) continue;
        if (scales.status[j] > std::uint8_t(StochasticStatus::nonfinite) || !std::isfinite(scales.first_scale[j]) || !std::isfinite(scales.second_scale[j]))
            throw std::invalid_argument("invalid stochastic scales");
        auto f=reference::stochastic_fit_gene(m.spliced ? m.spliced+j : nullptr,m.unspliced ? m.unspliced+j : nullptr,
            m.spliced_second ? m.spliced_second+j : nullptr,m.cross ? m.cross+j : nullptr,m.rows,m.stride,s.initial(k),
            {scales.first_scale[j],scales.second_scale[j],StochasticStatus(scales.status[j])},s.options,s.w->score.data(),s.w->mask.data());
        s.gamma[k]=f.gamma; s.offset[k]=f.offset; s.offset2[k]=f.offset2; s.scale1[k]=f.scales.first; s.scale2[k]=f.scales.second;
        s.status[k]=std::uint8_t(f.status); s.objective[k]=f.objective;
    }
    return {m.first,m.count,s.gamma+m.first,s.offset+m.first,s.offset2+m.first,s.scale1+m.first,s.scale2+m.first,s.status+m.first,s.objective+m.first};
}
template<class T> void BasicStochasticReferenceStages<T>::emit(BasicStochasticMomentBlock<T> m,BasicStochasticFitBlock<T> f) {
    auto& s=*state_; auto& w=*s.w; detail::check_stochastic_block(s,m);
    if (f.first!=m.first || f.count!=m.count || !f.gamma || !f.offset || !f.offset2 || !f.first_scale || !f.second_scale || !f.status || !f.objective)
        throw std::invalid_argument("invalid stochastic fit block");
    for (std::size_t j=0;j<m.count;++j) {
        if (!std::isfinite(f.gamma[j]) || !std::isfinite(f.offset[j]) || !std::isfinite(f.offset2[j]) ||
            !std::isfinite(f.first_scale[j]) || !std::isfinite(f.second_scale[j]) || !std::isfinite(f.objective[j]) || f.status[j]>std::uint8_t(StochasticStatus::nonfinite))
            throw std::invalid_argument("invalid stochastic fit diagnostics");
        auto k=m.first+j;
        s.gamma[k]=f.gamma[j]; s.offset[k]=f.offset[j]; s.offset2[k]=f.offset2[j];
        s.scale1[k]=f.first_scale[j]; s.scale2[k]=f.second_scale[j]; s.status[k]=f.status[j]; s.objective[k]=f.objective[j];
    }
    if (s.sink.write) {
        StochasticOutputBlock out; out.first=m.first; out.count=m.count;
        std::size_t slot=0;
        auto copy=[&](int field) {
            detail::output_dispatch(w.dtype,[&]<class O>() {
                O* dest;
                if constexpr(std::is_same_v<O,float>) dest=w.out32.data()+slot*w.elements;
                else dest=w.out64.data()+slot*w.elements;
                if constexpr(std::is_same_v<O,T>) {
                    if(field==0 && w.compiled && m.stride==std::ptrdiff_t(m.count)) {
                        detail::cpu::residuals(m.spliced,m.unspliced,s.gamma+m.first,s.offset+m.first,dest,m.rows,m.count);
                        return;
                    }
                }
                for(std::size_t i=0;i<m.rows;++i) for(std::size_t j=0;j<m.count;++j) {
                    auto q=std::ptrdiff_t(i)*m.stride+std::ptrdiff_t(j);auto k=m.first+j;
                    T x=m.spliced[q],y=m.unspliced[q],value=0;
                    if(field==0) value=y-s.gamma[k]*x-s.offset[k];
                    else if(field==1 && s.status[k]==std::uint8_t(StochasticStatus::ok))
                        value=(2*m.cross[q]+y-2*x*y)-s.gamma[k]*(2*m.spliced_second[q]-x-2*x*x)+2*s.offset[k]*x-s.offset2[k];
                    else if(field>=2) {
                        const T* source=field==2?m.spliced:field==3?m.unspliced:field==4?m.spliced_second:field==5?m.cross:m.unspliced_second;
                        if(!source && m.rows) throw std::invalid_argument("requested stochastic output is absent");
                        value=source[q];
                    }
                    dest[i*m.count+j]=detail::output_cast<O>(value);
                }
            });
            return w.view(slot++,m.count);
        };
        if(w.options.velocity) out.velocity=copy(0);
        if(w.options.second_residual) out.second_residual=copy(1);
        if(w.options.spliced_moments) out.spliced=copy(2);
        if(w.options.unspliced_moments) out.unspliced=copy(3);
        if(w.options.spliced_second) out.spliced_second=copy(4);
        if(w.options.cross) out.cross=copy(5);
        if(w.options.unspliced_second) out.unspliced_second=copy(6);
        s.sink.write(s.sink.context,out);
    }
    s.next+=m.count;
}
template<class T> StochasticFitResult BasicStochasticReferenceStages<T>::finish() {
    auto& s=*state_;
    if(s.finished || !s.resolved || s.next!=s.p->genes.size()) throw std::invalid_argument("unfinished stochastic blocks");
    s.finished=true; return std::move(s.result);
}

namespace reference {

template<class T> void moments_impl(const PreparedInput& p, std::size_t first, std::size_t count, T* ms, T* mu) {
    const auto& s = detail::Access::get(p);
    if (first > s.genes.size() || count > s.genes.size() - first || (count && s.input.n_obs() && (!ms || !mu)))
        throw std::invalid_argument("invalid moment output or range");
    (void)detail::mul(s.input.n_obs(), count);
    if (s.checked) s.input.check_values();
    if (!count) return;
    auto bytes = detail::mul(detail::mul(s.input.n_obs(), count), sizeof(T));
    auto a = detail::contiguous_span(reinterpret_cast<const std::byte*>(ms), bytes);
    auto b = detail::contiguous_span(reinterpret_cast<const std::byte*>(mu), bytes);
    if (detail::overlaps(a, b)) throw std::invalid_argument("moment outputs overlap");
    detail::check_input_alias(s, a); detail::check_input_alias(s, b);
    s.input.moments_selected(s.genes.data() + first, count, ms, mu);
}

template<class T> FitResult steady_state_impl(const PreparedInput& p, Workspace& w, FitOptions options, BlockSink sink, bool compiled=false) {
    BasicReferenceStages<T> stages(p, w, options, sink);
    detail::Access::get(stages).w->compiled=compiled;
    return steady_state_with(stages, p.n_genes(), w.block_genes());
}

template<class T> void residuals_impl(const BasicMomentBlock<T>& m, const BasicFitBlock<T>& fit, anndatapp::ArrayEdit output) {
    if (m.count != fit.count || m.first != fit.first || m.stride < std::ptrdiff_t(m.count) ||
        (m.rows && m.count && (!m.spliced || !m.unspliced || !fit.gamma || !fit.offset)))
        throw std::invalid_argument("incompatible moment and fit blocks");
    detail::check_output_shape(output.view(), m.rows, m.count);
    auto dst = output.bytes();
    auto target = detail::span(dst.data(), m.rows, m.count, dst.stride_bytes(0), dst.stride_bytes(1), dst.element_size());
    auto mr = detail::mul(detail::mul(std::size_t(m.stride), sizeof(T)), m.rows ? m.rows - 1 : 0);
    auto input_bytes = detail::add(mr, detail::mul(m.count, sizeof(T)));
    if (m.rows && m.count && (detail::overlaps(target, detail::contiguous_span(reinterpret_cast<const std::byte*>(m.spliced), input_bytes)) ||
                             detail::overlaps(target, detail::contiguous_span(reinterpret_cast<const std::byte*>(m.unspliced), input_bytes)) ||
                             detail::overlaps(target, detail::contiguous_span(reinterpret_cast<const std::byte*>(fit.gamma), detail::mul(m.count, sizeof(T)))) ||
                             detail::overlaps(target, detail::contiguous_span(reinterpret_cast<const std::byte*>(fit.offset), detail::mul(m.count, sizeof(T))))))
        throw std::invalid_argument("output aliases moment or fit buffers");
    auto rs = dst.stride_bytes(0), cs = dst.stride_bytes(1);
    auto write = [&]<class O>() {
        for (std::size_t i = 0; i < m.rows; ++i) for (std::size_t j = 0; j < m.count; ++j) {
            O value = detail::output_cast<O>(m.unspliced[i*std::size_t(m.stride) + j] - fit.gamma[j]*m.spliced[i*std::size_t(m.stride) + j] - fit.offset[j]);
            auto* ptr = dst.data() + std::ptrdiff_t(i)*rs + std::ptrdiff_t(j)*cs;
            std::memcpy(ptr, &value, sizeof value);
        }
    };
    detail::output_dispatch(output.view().dtype(),write);
}

template<class T> FitResult steady_state_into_impl(const PreparedInput& p, Workspace& w, anndatapp::ArrayEdit output, FitOptions options) {
    detail::check_output_shape(output.view(), p.n_cells(), p.n_genes());
    auto dst = output.bytes();
    auto target = detail::span(dst.data(), p.n_cells(), p.n_genes(), dst.stride_bytes(0), dst.stride_bytes(1), dst.element_size());
    detail::check_input_alias(detail::Access::get(p), target);
    auto& ws = detail::Access::get(w).get<T>();
    if (detail::overlaps(target, detail::contiguous_span(reinterpret_cast<const std::byte*>(ws.ms.data()), detail::mul(ws.ms.size(), sizeof(T)))) ||
        detail::overlaps(target, detail::contiguous_span(reinterpret_cast<const std::byte*>(ws.mu.data()), detail::mul(ws.mu.size(), sizeof(T)))))
        throw std::invalid_argument("output aliases workspace");
    BasicReferenceStages<T> stages(p, w, options);
    auto rs = dst.stride_bytes(0), cs = dst.stride_bytes(1);
    auto write = [&]<class O>() {
        for (std::size_t first = 0; first < p.n_genes();) {
            auto count = std::min(w.block_genes(), p.n_genes() - first);
            auto m = stages.moments(first, count); auto f = stages.fit(m);
            for (std::size_t i = 0; i < m.rows; ++i) for (std::size_t j = 0; j < count; ++j) {
                O value = detail::output_cast<O>(m.unspliced[i*count + j] - f.gamma[j]*m.spliced[i*count + j] - f.offset[j]);
                auto* ptr = dst.data() + std::ptrdiff_t(i)*rs + std::ptrdiff_t(first + j)*cs;
                std::memcpy(ptr, &value, sizeof value);
            }
            stages.emit(m, f); first += count;
        }
    };
    detail::output_dispatch(output.view().dtype(),write);
    return stages.finish();
}

}
namespace reference {
template<class T> void second_moments_impl(const PreparedInput& p,std::size_t first,std::size_t count,T* ms,T* mu,
                    T* mss,T* mus,T* muu,T* source_s,T* source_u) {
    auto& s=detail::Access::get(p);
    if(first>s.genes.size() || count>s.genes.size()-first || (count && s.input.n_obs() && ((!source_s && (ms||mss||mus)) || (!source_u && (mu||mus||muu)))))
        throw std::invalid_argument("invalid second moment range or scratch");
    if (!count || !s.input.n_obs()) return;
    if (s.checked) s.input.check_values();
    auto bytes=detail::mul(detail::mul(s.input.n_obs(),count),sizeof(T));
    std::array<detail::Span,7> buffers{};
    std::size_t n=0;
    for (auto* out : {ms,mu,mss,mus,muu}) if (out) buffers[n++]=detail::contiguous_span(reinterpret_cast<const std::byte*>(out),bytes);
    auto source_bytes=detail::mul(s.input.n_obs(),sizeof(T));
    if (ms||mss||mus) buffers[n++]=detail::contiguous_span(reinterpret_cast<const std::byte*>(source_s),source_bytes);
    if (mu||mus||muu) buffers[n++]=detail::contiguous_span(reinterpret_cast<const std::byte*>(source_u),source_bytes);
    for (std::size_t i=0;i<n;++i) {
        detail::check_input_alias(s,buffers[i]);
        for (std::size_t j=0;j<i;++j) if (detail::overlaps(buffers[i],buffers[j]))
            throw std::invalid_argument("second moment outputs or scratch overlap");
    }
    s.input.second_moments_selected(s.genes.data()+first,count,ms,mu,mss,mus,muu,source_s,source_u);
}
namespace {
template<class T> void check_stochastic_gene_impl(const T* ms,const T* mu,const T* mss,const T* mus,
                           std::size_t rows,std::ptrdiff_t stride,const StochasticOptions& o) {
    T dummy=0;
    reference::fit_gene(nullptr,nullptr,0,1,o.initialization,&dummy);
    if (!rows) return;
    if (!ms || !mu || !mss || !mus || !stride) throw std::invalid_argument("invalid stochastic gene buffers");
    auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(rows>1 && detail::magnitude(stride)>limit/(rows-1)) throw std::invalid_argument("stochastic gene stride overflows");
}
}
template<class T> BasicStochasticScales<T> stochastic_scales_impl(const T* ms,const T* mu,const T* mss,const T* mus,
    std::size_t rows,std::ptrdiff_t stride,const BasicGeneFit<T>& initial,const StochasticOptions& o,T*,const std::uint8_t* mask) {
    check_stochastic_gene_impl(ms,mu,mss,mus,rows,stride,o);
    return detail::stochastic_scales(ms,mu,mss,mus,rows,stride,initial,o,mask);
}
template<class T> BasicStochasticGeneFit<T> stochastic_fit_gene_impl(const T* ms,const T* mu,const T* mss,const T* mus,
    std::size_t rows,std::ptrdiff_t stride,const BasicGeneFit<T>& initial,BasicStochasticScales<T> scales,const StochasticOptions& o,T* scratch,const std::uint8_t* mask) {
    check_stochastic_gene_impl(ms,mu,mss,mus,rows,stride,o);
    if (rows && o.initialization.trimming!=Trimming::none && !scratch) throw std::invalid_argument("stochastic fitting scratch is null");
    return detail::stochastic_fit_gene(ms,mu,mss,mus,rows,stride,initial,scales,o,scratch,mask);
}
template<class T> void stochastic_residuals_impl(const BasicStochasticMomentBlock<T>& m,const BasicStochasticFitBlock<T>& f,anndatapp::ArrayEdit velocity,
                          std::optional<anndatapp::ArrayEdit> second) {
    detail::check_output_shape(velocity.view(),m.rows,m.count);
    if(second) detail::check_output_shape(second->view(),m.rows,m.count);
    if(f.first!=m.first || f.count!=m.count) throw std::invalid_argument("incompatible stochastic fit block");
    if (!m.rows || !m.count) return;
    if (!f.gamma || !f.offset || !f.offset2 || !f.status || !m.spliced || !m.unspliced ||
        (second && (!m.spliced_second || !m.cross)) || !m.stride)
        throw std::invalid_argument("invalid stochastic residual inputs");
    auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if (m.count-1>limit || detail::magnitude(m.stride)>limit/sizeof(T) ||
        (m.rows>1 && detail::magnitude(m.stride)>(limit-(m.count-1))/(m.rows-1)))
        throw std::invalid_argument("stochastic residual stride overflows");
    std::array<detail::Span,2> targets{};
    auto target=[&](anndatapp::ArrayEdit dest) {
        auto b=dest.bytes();
        return detail::span(b.data(),m.rows,m.count,b.stride_bytes(0),b.stride_bytes(1),b.element_size());
    };
    targets[0]=target(velocity); if(second) targets[1]=target(*second);
    if(second && detail::overlaps(targets[0],targets[1])) throw std::invalid_argument("stochastic residual outputs overlap");
    for (auto dest : targets) {
        for (auto* source : {m.spliced,m.unspliced,second?m.spliced_second:nullptr,second?m.cross:nullptr})
            if (source && detail::overlaps(dest,detail::span(reinterpret_cast<const std::byte*>(source),m.rows,m.count,
                    m.stride*std::ptrdiff_t(sizeof(T)),sizeof(T),sizeof(T))))
                throw std::invalid_argument("stochastic residual output aliases moments");
        for (auto* source : {f.gamma,f.offset,f.offset2})
            if(detail::overlaps(dest,detail::contiguous_span(reinterpret_cast<const std::byte*>(source),detail::mul(m.count,sizeof(T)))))
                throw std::invalid_argument("stochastic residual output aliases fit");
        if(detail::overlaps(dest,detail::contiguous_span(reinterpret_cast<const std::byte*>(f.status),m.count)))
            throw std::invalid_argument("stochastic residual output aliases status");
    }
    auto write=[&](anndatapp::ArrayEdit dest,bool second_field) {
        auto b=dest.bytes();
        detail::output_dispatch(dest.view().dtype(),[&]<class O>() {
        for(std::size_t i=0;i<m.rows;++i) for(std::size_t j=0;j<m.count;++j) {
            auto q=std::ptrdiff_t(i)*m.stride+std::ptrdiff_t(j);
            T x=m.spliced[q], y=m.unspliced[q];
            T value=second_field ? (f.status[j]==std::uint8_t(StochasticStatus::ok)
                ? (2*m.cross[q]+y-2*x*y)-f.gamma[j]*(2*m.spliced_second[q]-x-2*x*x)+2*f.offset[j]*x-f.offset2[j] : 0)
                : y-f.gamma[j]*x-f.offset[j];
            auto* ptr=b.data()+std::ptrdiff_t(i)*b.stride_bytes(0)+std::ptrdiff_t(j)*b.stride_bytes(1);
            O v=detail::output_cast<O>(value);std::memcpy(ptr,&v,sizeof v);
        }
        });
    };
    write(velocity,false); if(second) write(*second,true);
}
template<class T> StochasticFitResult stochastic_impl(const PreparedInput& p,StochasticWorkspace& w,StochasticOptions o,StochasticBlockSink b,bool compiled=false) {
    BasicStochasticReferenceStages<T> stages(p,w,o,b);
    detail::Access::get(stages).w->compiled=compiled;
    return stochastic_with(stages,p.n_genes(),w.block_genes());
}
template<class T> StochasticFitResult stochastic_into_impl(const PreparedInput& p,StochasticWorkspace& w,anndatapp::ArrayEdit output,
                                   StochasticOptions o,std::optional<anndatapp::ArrayEdit> second) {
    auto& ps=detail::Access::get(p); auto& ws=detail::Access::get(w).get<T>();
    if(ws.owner!=&ps) throw std::invalid_argument("stochastic workspace belongs to another input");
    auto check=[&](anndatapp::ArrayEdit dest) {
        detail::check_output_shape(dest.view(),p.n_cells(),p.n_genes());
        auto b=dest.bytes(); auto target=detail::span(b.data(),p.n_cells(),p.n_genes(),b.stride_bytes(0),b.stride_bytes(1),b.element_size());
        detail::check_input_alias(ps,target);
        for(auto* v : {&ws.ms,&ws.mu,&ws.mss,&ws.mus,&ws.muu,&ws.source_s,&ws.source_u,&ws.score})
            if(detail::overlaps(target,detail::contiguous_span(reinterpret_cast<const std::byte*>(v->data()),detail::mul(v->size(),sizeof(T)))))
                throw std::invalid_argument("stochastic output aliases workspace");
        if(detail::overlaps(target,detail::contiguous_span(reinterpret_cast<const std::byte*>(ws.out64.data()),detail::mul(ws.out64.size(),sizeof(double))))) throw std::invalid_argument("stochastic output aliases workspace");
        if(detail::overlaps(target,detail::contiguous_span(reinterpret_cast<const std::byte*>(ws.out32.data()),detail::mul(ws.out32.size(),sizeof(float)))))
            throw std::invalid_argument("stochastic output aliases workspace");
        return target;
    };
    auto target=check(output);
    if(second && detail::overlaps(target,check(*second))) throw std::invalid_argument("stochastic outputs overlap");
    BasicStochasticReferenceStages<T> stages(p,w,o);
    for(std::size_t first=0;first<p.n_genes();) {
        auto count=std::min(w.block_genes(),p.n_genes()-first); auto m=stages.initial_moments(first,count);
        stages.accept_initialization(m,stages.initial_fit(m)); first+=count;
    }
    stages.resolve_selection();
    for(std::size_t first=0;first<p.n_genes();) {
        auto count=std::min(w.block_genes(),p.n_genes()-first); auto m=stages.moments(first,count);
        auto f=stages.fit(m,stages.scales(m));
        auto write=[&](anndatapp::ArrayEdit dest,bool second_field) {
            auto b=dest.bytes();
        detail::output_dispatch(dest.view().dtype(),[&]<class O>() {
            for(std::size_t i=0;i<m.rows;++i) for(std::size_t j=0;j<count;++j) {
                auto q=i*count+j; T x=m.spliced[q],y=m.unspliced[q];
                T value=second_field ? (f.status[j]==std::uint8_t(StochasticStatus::ok)
                    ? (2*m.cross[q]+y-2*x*y)-f.gamma[j]*(2*m.spliced_second[q]-x-2*x*x)+2*f.offset[j]*x-f.offset2[j] : 0)
                    : y-f.gamma[j]*x-f.offset[j];
                auto* ptr=b.data()+std::ptrdiff_t(i)*b.stride_bytes(0)+std::ptrdiff_t(first+j)*b.stride_bytes(1);
                O v=detail::output_cast<O>(value);std::memcpy(ptr,&v,sizeof v);
            }
            });
        };
        write(output,false); if(second) write(*second,true);
        stages.emit(m,f); first+=count;
    }
    return stages.finish();
}

}

}
namespace velocitypp::reference {
#define VELOCITYPP_PRIMITIVE_WRAPPERS(T) \
void moments(const PreparedInput& p,std::size_t first,std::size_t count,T* ms,T* mu){moments_impl(p,first,count,ms,mu);} \
void residuals(const BasicMomentBlock<T>& m,const BasicFitBlock<T>& f,anndatapp::ArrayEdit o){residuals_impl(m,f,o);} \
void second_moments(const PreparedInput& p,std::size_t first,std::size_t count,T* ms,T* mu,T* mss,T* mus,T* muu,T* ss,T* su){second_moments_impl(p,first,count,ms,mu,mss,mus,muu,ss,su);} \
BasicStochasticScales<T> stochastic_scales(const T* ms,const T* mu,const T* mss,const T* mus,std::size_t rows,std::ptrdiff_t stride,const BasicGeneFit<T>& initial,const StochasticOptions& o,T* scratch,const std::uint8_t* mask){return stochastic_scales_impl(ms,mu,mss,mus,rows,stride,initial,o,scratch,mask);} \
BasicStochasticGeneFit<T> stochastic_fit_gene(const T* ms,const T* mu,const T* mss,const T* mus,std::size_t rows,std::ptrdiff_t stride,const BasicGeneFit<T>& initial,BasicStochasticScales<T> scales,const StochasticOptions& o,T* scratch,const std::uint8_t* mask){return stochastic_fit_gene_impl(ms,mu,mss,mus,rows,stride,initial,scales,o,scratch,mask);} \
void stochastic_residuals(const BasicStochasticMomentBlock<T>& m,const BasicStochasticFitBlock<T>& f,anndatapp::ArrayEdit output,std::optional<anndatapp::ArrayEdit> second){stochastic_residuals_impl(m,f,output,second);}
VELOCITYPP_PRIMITIVE_WRAPPERS(float)
VELOCITYPP_PRIMITIVE_WRAPPERS(double)
#undef VELOCITYPP_PRIMITIVE_WRAPPERS
FitResult steady_state(const PreparedInput& p,Workspace& w,FitOptions o,BlockSink b){
    if(w.compute_precision()==ComputePrecision::float32)return steady_state_impl<float>(p,w,o,b);
    return steady_state_impl<double>(p,w,o,b);
}
FitResult steady_state_into(const PreparedInput& p,Workspace& w,anndatapp::ArrayEdit output,FitOptions o){
    if(w.compute_precision()==ComputePrecision::float32)return steady_state_into_impl<float>(p,w,output,o);
    return steady_state_into_impl<double>(p,w,output,o);
}
StochasticFitResult stochastic(const PreparedInput& p,StochasticWorkspace& w,StochasticOptions o,StochasticBlockSink b){
    if(w.compute_precision()==ComputePrecision::float32)return stochastic_impl<float>(p,w,o,b);
    return stochastic_impl<double>(p,w,o,b);
}
StochasticFitResult stochastic_into(const PreparedInput& p,StochasticWorkspace& w,anndatapp::ArrayEdit output,StochasticOptions o,std::optional<anndatapp::ArrayEdit> second){
    if(w.compute_precision()==ComputePrecision::float32)return stochastic_into_impl<float>(p,w,output,o,second);
    return stochastic_into_impl<double>(p,w,output,o,second);
}
}
namespace velocitypp::cpu {
FitResult steady_state(const PreparedInput& p,Workspace& w,FitOptions o,BlockSink b){
    if(w.compute_precision()==ComputePrecision::float32)return reference::steady_state_impl<float>(p,w,o,b,true);
    return reference::steady_state_impl<double>(p,w,o,b,true);
}
StochasticFitResult stochastic(const PreparedInput& p,StochasticWorkspace& w,StochasticOptions o,StochasticBlockSink b){
    if(w.compute_precision()==ComputePrecision::float32)return reference::stochastic_impl<float>(p,w,o,b,true);
    return reference::stochastic_impl<double>(p,w,o,b,true);
}
}
namespace velocitypp {
namespace detail {
template<class T,class Block> void check_replacement(const Block& m,std::size_t offset,std::size_t count,anndatapp::ArrayView s,anndatapp::ArrayView u) {
    s.validate();u.validate();
    if(offset>m.count || count>m.count-offset || s.layout()!=anndatapp::Layout::dense || u.layout()!=anndatapp::Layout::dense || s.rank()!=2 || u.rank()!=2 || s.extent(0)!=m.rows || u.extent(0)!=m.rows || s.extent(1)!=count || u.extent(1)!=count || m.stride<std::ptrdiff_t(m.count))
        throw std::invalid_argument("invalid alternate source block or destination range");
    auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(m.rows>1 && std::size_t(m.stride)>limit/sizeof(T))throw std::invalid_argument("replacement stride overflows byte offsets");
    auto byte_stride=m.rows>1 ? m.stride*std::ptrdiff_t(sizeof(T)) : 0;
    std::array<T*,5> destinations{m.spliced,m.unspliced,nullptr,nullptr,nullptr};
    if constexpr(requires {m.spliced_second;}) {
        destinations[2]=m.spliced_second;destinations[3]=m.cross;destinations[4]=m.unspliced_second;
    }
    for(auto* destination:destinations) if(destination) {
        auto target=span(reinterpret_cast<const std::byte*>(destination),m.rows,m.count,byte_stride,sizeof(T),sizeof(T));
        if(aliases(s,target)||aliases(u,target))throw std::invalid_argument("alternate source aliases moments");
    }
}
}
template<class T> void BasicReferenceStages<T>::replace_moments(BasicMomentBlock<T>& m,std::size_t offset,std::size_t count,anndatapp::ArrayView a,anndatapp::ArrayView b) {
    auto& s=*state_;detail::check_block(s,m);detail::check_replacement<T>(m,offset,count,a,b);
    if(!count || !m.rows)return;
    s.p->input.moments_from(a,b,m.spliced+offset,m.unspliced+offset,std::size_t(m.stride),s.p->checked);
}
template<class T> void BasicStochasticReferenceStages<T>::replace_moments(BasicMomentBlock<T>& m,std::size_t offset,std::size_t count,anndatapp::ArrayView a,anndatapp::ArrayView b) {
    auto& s=*state_;detail::check_stochastic_range(s,m.first,m.count,false);
    if(m.rows!=s.w->n || (m.rows && (!m.spliced||!m.unspliced)))throw std::invalid_argument("invalid initialization moments");
    detail::check_replacement<T>(m,offset,count,a,b);
    if(!count || !m.rows)return;
    s.p->input.moments_from(a,b,m.spliced+offset,m.unspliced+offset,std::size_t(m.stride),s.p->checked);
}
template<class T> void BasicStochasticReferenceStages<T>::replace_moments(BasicStochasticMomentBlock<T>& m,std::size_t offset,std::size_t count,anndatapp::ArrayView a,anndatapp::ArrayView b) {
    auto& s=*state_;detail::check_stochastic_block(s,m);detail::check_replacement<T>(m,offset,count,a,b);
    if(!count || !m.rows)return;
    for(auto* scratch:{s.w->source_s.data(),s.w->source_u.data()}) {
        auto target=detail::contiguous_span(reinterpret_cast<const std::byte*>(scratch),detail::mul(m.rows,sizeof(T)));
        if(detail::aliases(a,target)||detail::aliases(b,target))throw std::invalid_argument("alternate source aliases source scratch");
    }
    s.p->input.second_moments_from(a,b,m.spliced+offset,m.unspliced+offset,m.spliced_second+offset,m.cross+offset,
        m.unspliced_second?m.unspliced_second+offset:nullptr,s.w->source_s.data(),s.w->source_u.data(),std::size_t(m.stride),s.p->checked);
}
namespace reference {
template<class T> void moments_from(const PreparedInput& p,anndatapp::ArrayView s,anndatapp::ArrayView u,T* ms,T* mu,std::size_t stride) {
    auto& state=detail::Access::get(p);s.validate();u.validate();
    if(s.rank()!=2 || u.rank()!=2 || s.extent(0)!=p.n_cells() || u.extent(0)!=p.n_cells() || s.extent(1)!=u.extent(1) || stride<s.extent(1))throw std::invalid_argument("invalid alternate source shapes or stride");
    if(stride>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(T))throw std::invalid_argument("alternate destination stride overflows byte offsets");
    auto count=s.extent(1);
    if(!count || !p.n_cells())return;
    if(!ms||!mu)throw std::invalid_argument("alternate moments output is null");
    auto a=detail::span(reinterpret_cast<const std::byte*>(ms),p.n_cells(),count,detail::mul(stride,sizeof(T)),sizeof(T),sizeof(T));
    auto b=detail::span(reinterpret_cast<const std::byte*>(mu),p.n_cells(),count,detail::mul(stride,sizeof(T)),sizeof(T),sizeof(T));
    if(detail::overlaps(a,b)||detail::aliases(s,a)||detail::aliases(s,b)||detail::aliases(u,a)||detail::aliases(u,b))throw std::invalid_argument("alternate source or moment buffers overlap");
    detail::check_input_alias(state,a);detail::check_input_alias(state,b);
    state.input.moments_from(s,u,ms,mu,stride,state.checked);
}
template<class T> void second_moments_from(const PreparedInput& p,anndatapp::ArrayView s,anndatapp::ArrayView u,T* ms,T* mu,T* mss,T* mus,T* muu,T* source_s,T* source_u,std::size_t stride) {
    auto& state=detail::Access::get(p);s.validate();u.validate();
    if(s.rank()!=2 || u.rank()!=2 || s.extent(0)!=p.n_cells() || u.extent(0)!=p.n_cells() || s.extent(1)!=u.extent(1) || stride<s.extent(1))throw std::invalid_argument("invalid alternate source shapes or stride");
    if(stride>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(T))throw std::invalid_argument("alternate destination stride overflows byte offsets");
    auto count=s.extent(1);
    if(!count || !p.n_cells())return;
    if((ms||mss||mus)&&!source_s)throw std::invalid_argument("alternate source_s scratch is null");
    if((mu||mus||muu)&&!source_u)throw std::invalid_argument("alternate source_u scratch is null");
    std::array<detail::Span,7> spans{};std::size_t n=0;
    for(auto* out:{ms,mu,mss,mus,muu})if(out)spans[n++]=detail::span(reinterpret_cast<const std::byte*>(out),p.n_cells(),count,detail::mul(stride,sizeof(T)),sizeof(T),sizeof(T));
    if(ms||mss||mus)spans[n++]=detail::contiguous_span(reinterpret_cast<const std::byte*>(source_s),detail::mul(p.n_cells(),sizeof(T)));
    if(mu||mus||muu)spans[n++]=detail::contiguous_span(reinterpret_cast<const std::byte*>(source_u),detail::mul(p.n_cells(),sizeof(T)));
    for(std::size_t i=0;i<n;++i) {
        detail::check_input_alias(state,spans[i]);
        if(detail::aliases(s,spans[i])||detail::aliases(u,spans[i]))throw std::invalid_argument("alternate source aliases moment buffers");
        for(std::size_t j=0;j<i;++j)if(detail::overlaps(spans[i],spans[j]))throw std::invalid_argument("alternate moment buffers overlap");
    }
    state.input.second_moments_from(s,u,ms,mu,mss,mus,muu,source_s,source_u,stride,state.checked);
}
#define VELOCITYPP_ALTERNATE_INSTANTIATE(T) \
template void moments_from<T>(const PreparedInput&,anndatapp::ArrayView,anndatapp::ArrayView,T*,T*,std::size_t); \
template void second_moments_from<T>(const PreparedInput&,anndatapp::ArrayView,anndatapp::ArrayView,T*,T*,T*,T*,T*,T*,T*,std::size_t);
VELOCITYPP_ALTERNATE_INSTANTIATE(float)
VELOCITYPP_ALTERNATE_INSTANTIATE(double)
#undef VELOCITYPP_ALTERNATE_INSTANTIATE
}
}
namespace velocitypp {
template<class T> bool BasicReferenceStages<T>::bound_to(const PreparedInput& p,const Workspace& w) const noexcept {
    auto* ws=detail::Access::pointer(w);
    if(!state_||!ws||state_->p!=detail::Access::pointer(p))return false;
    if constexpr(std::is_same_v<T,float>)return state_->w==ws->f32.get();
    else return state_->w==ws->f64.get();
}
template<class T> bool BasicStochasticReferenceStages<T>::bound_to(const PreparedInput& p,const StochasticWorkspace& w) const noexcept {
    auto* ws=detail::Access::pointer(w);
    if(!state_||!ws||state_->p!=detail::Access::pointer(p))return false;
    if constexpr(std::is_same_v<T,float>)return state_->w==ws->f32.get();
    else return state_->w==ws->f64.get();
}
}
template class velocitypp::BasicReferenceStages<float>;
template class velocitypp::BasicReferenceStages<double>;
template class velocitypp::BasicStochasticReferenceStages<float>;
template class velocitypp::BasicStochasticReferenceStages<double>;
