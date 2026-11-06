#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

import velocitypp;

namespace {

int failures = 0;

void expect(bool ok, const char* message) {
    if (ok) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

bool close(double a, double b, double tol = 2e-6) {
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}

template<class F> void expect_invalid_argument(F&& f, const char* message) {
    try {
        f();
        expect(false, message);
    } catch (const std::invalid_argument&) {
    } catch (...) {
        expect(false, message);
    }
}

constexpr std::size_t n = 4, g = 2;

struct DenseFixture {
    std::vector<double> s{1.0, 1.0, 1.0 + 1e-8, 2.0,
                          1.0 + 2e-8, 3.0, 1.0 + 3e-8, 4.0};
    std::vector<double> u{2.0, 2.1, 3.0, 4.2,
                          4.0, 6.4, 5.0, 8.5};
    std::vector<double> w = std::vector<double>(n * n, 0.0);
    anndatapp::Array sa = anndatapp::Array::borrow_dense(s.data(), {n, g}, {std::ptrdiff_t(g), 1});
    anndatapp::Array ua = anndatapp::Array::borrow_dense(u.data(), {n, g}, {std::ptrdiff_t(g), 1});
    anndatapp::Array wa = [&] {
        for (std::size_t i = 0; i < n; ++i) w[i * n + i] = 1.0;
        return anndatapp::Array::borrow_dense(w.data(), {n, n}, {std::ptrdiff_t(n), 1});
    }();
    velocitypp::PreparedInput prepared{{sa.view(), ua.view(), wa.view()}};
};

velocitypp::FitOptions fit_options() {
    velocitypp::FitOptions options;
    options.trimming = velocitypp::Trimming::none;
    options.fit_intercept = true;
    options.positive_intercept = false;
    options.min_r2 = 0;
    options.min_ratio = 0;
    options.low_signal = velocitypp::LowSignal::keep_filters;
    return options;
}

std::vector<double> floating_values(anndatapp::ArrayView view) {
    const auto rows = view.extent(0);
    const auto columns = view.rank() == 1 ? std::size_t(1) : view.extent(1);
    std::vector<double> values(rows * columns);
    auto copy = [&](auto typed) {
        for (std::size_t i = 0; i < rows; ++i) for (std::size_t j = 0; j < columns; ++j)
            values[i * columns + j] = typed.data()[std::ptrdiff_t(i) * typed.stride(0) +
                (view.rank() == 1 ? 0 : std::ptrdiff_t(j) * typed.stride(1))];
    };
    if (view.dtype() == anndatapp::DType::float32) copy(view.dense<float>());
    else if (view.dtype() == anndatapp::DType::float64) copy(view.dense<double>());
    else throw std::runtime_error("expected float32 or float64 array");
    return values;
}

bool numeric_arrays_match(anndatapp::ArrayView a, anndatapp::ArrayView b, double tol) {
    if (a.dtype() != b.dtype() || a.rank() != b.rank()) return false;
    for (std::size_t d = 0; d < a.rank(); ++d)
        if (a.extent(d) != b.extent(d)) return false;
    const auto av = floating_values(a), bv = floating_values(b);
    if (av.size() != bv.size()) return false;
    for (std::size_t i = 0; i < av.size(); ++i)
        if (!close(av[i], bv[i], tol)) return false;
    return true;
}

bool byte_arrays_match(anndatapp::ArrayView a, anndatapp::ArrayView b) {
    if (a.dtype() != anndatapp::DType::uint8 || b.dtype() != anndatapp::DType::uint8 ||
        a.rank() != b.rank()) return false;
    for (std::size_t d = 0; d < a.rank(); ++d)
        if (a.extent(d) != b.extent(d)) return false;
    auto av = a.dense<std::uint8_t>(), bv = b.dense<std::uint8_t>();
    if (a.rank() != 1) return false;
    for (std::size_t i = 0; i < a.extent(0); ++i)
        if (av.data()[std::ptrdiff_t(i) * av.stride(0)] !=
            bv.data()[std::ptrdiff_t(i) * bv.stride(0)]) return false;
    return true;
}

bool fit_results_match(const velocitypp::FitResult& a, const velocitypp::FitResult& b, double tol) {
    if (!numeric_arrays_match(a.gamma.view(), b.gamma.view(), tol) ||
        !numeric_arrays_match(a.offset.view(), b.offset.view(), tol) ||
        !numeric_arrays_match(a.r2.view(), b.r2.view(), tol)) return false;
    auto ag = a.genes.view().dense<bool>(), bg = b.genes.view().dense<bool>();
    auto as = a.status.view().dense<std::uint8_t>(), bs = b.status.view().dense<std::uint8_t>();
    for (std::size_t i = 0; i < ag.extent(0); ++i)
        if (ag.data()[std::ptrdiff_t(i) * ag.stride(0)] != bg.data()[std::ptrdiff_t(i) * bg.stride(0)] ||
            as.data()[std::ptrdiff_t(i) * as.stride(0)] != bs.data()[std::ptrdiff_t(i) * bs.stride(0)]) return false;
    return a.low_signal == b.low_signal;
}

bool stochastic_results_match(const velocitypp::StochasticFitResult& a,
                              const velocitypp::StochasticFitResult& b, double tol) {
    return fit_results_match(a.initialization, b.initialization, tol) &&
        numeric_arrays_match(a.gamma.view(), b.gamma.view(), tol) &&
        numeric_arrays_match(a.offset.view(), b.offset.view(), tol) &&
        numeric_arrays_match(a.offset2.view(), b.offset2.view(), tol) &&
        numeric_arrays_match(a.first_scale.view(), b.first_scale.view(), tol) &&
        numeric_arrays_match(a.second_scale.view(), b.second_scale.view(), tol) &&
        numeric_arrays_match(a.objective.view(), b.objective.view(), tol) &&
        byte_arrays_match(a.status.view(), b.status.view()) &&
        a.refit_all == b.refit_all;
}

struct OutputCapture {
    std::size_t rows = n, genes = g;
    std::vector<double> velocity = std::vector<double>(n * g, 0.0);
    std::vector<double> second = std::vector<double>(n * g, 0.0);
};

void capture_steady(void* context, const velocitypp::OutputBlock& block) {
    auto& capture = *static_cast<OutputCapture*>(context);
    auto values = floating_values(block.velocity);
    for (std::size_t i = 0; i < capture.rows; ++i)
        for (std::size_t j = 0; j < block.count; ++j)
            capture.velocity[i * capture.genes + block.first + j] = values[i * block.count + j];
}

void capture_stochastic(void* context, const velocitypp::StochasticOutputBlock& block) {
    auto& capture = *static_cast<OutputCapture*>(context);
    auto velocity_values = floating_values(block.velocity);
    auto second_values = floating_values(block.second_residual);
    for (std::size_t i = 0; i < capture.rows; ++i)
        for (std::size_t j = 0; j < block.count; ++j) {
            const auto out = i * capture.genes + block.first + j;
            capture.velocity[out] = velocity_values[i * block.count + j];
            capture.second[out] = second_values[i * block.count + j];
        }
}

bool captures_match(const OutputCapture& a, const OutputCapture& b, double tol) {
    if (a.velocity.size() != b.velocity.size() || a.second.size() != b.second.size()) return false;
    for (std::size_t i = 0; i < a.velocity.size(); ++i)
        if (!close(a.velocity[i], b.velocity[i], tol) || !close(a.second[i], b.second[i], tol)) return false;
    return true;
}

void default_precision_resolution_and_budgets() {
    expect(velocitypp::RunOptions{}.compute_precision == velocitypp::ComputePrecision::automatic &&
           velocitypp::StochasticRunOptions{}.compute_precision == velocitypp::ComputePrecision::automatic,
           "both run-option families default to automatic compute precision");

    auto fs = anndatapp::Array::zeros<float>({1, 1});
    auto fu = anndatapp::Array::zeros<float>({1, 1});
    auto fw = anndatapp::Array::zeros<float>({1, 1});
    fw.edit().dense<float>().data()[0] = 1.0f;
    velocitypp::PreparedInput all_float{{fs.view(), fu.view(), fw.view()}};
    velocitypp::Workspace float_workspace(all_float);
    velocitypp::StochasticWorkspace float_stochastic_workspace(all_float);
    expect(float_workspace.compute_precision() == velocitypp::ComputePrecision::float64 &&
           float_stochastic_workspace.compute_precision() == velocitypp::ComputePrecision::float64,
           "automatic computation selects FP64 for supported FP32 storage");
    expect(float_workspace.scratch_bytes() <= 64u * 1024u * 1024u &&
           float_stochastic_workspace.scratch_bytes() <= 64u * 1024u * 1024u,
           "default steady-state and stochastic numeric scratch stay within 64 MiB");

    auto wd = anndatapp::Array::zeros<double>({1, 1});
    wd.edit().dense<double>().data()[0] = 1.0;
    velocitypp::PreparedInput mixed_float_double{{fs.view(), fu.view(), wd.view()}};
    velocitypp::Workspace mixed_workspace(mixed_float_double);
    expect(mixed_workspace.compute_precision() == velocitypp::ComputePrecision::float64,
           "mixed FP32/FP64 storage uses FP64 automatic computation");

    velocitypp::RunOptions force32;
    force32.compute_precision = velocitypp::ComputePrecision::float32;
    velocitypp::RunOptions force64;
    force64.compute_precision = velocitypp::ComputePrecision::float64;
    velocitypp::Workspace explicit32(mixed_float_double, force32);
    velocitypp::Workspace explicit64(all_float, force64);
    expect(explicit32.compute_precision() == velocitypp::ComputePrecision::float32 &&
           explicit64.compute_precision() == velocitypp::ComputePrecision::float64,
           "explicit compute-precision choices select either supported arithmetic type");
}

void typed_stages_expose_real_float32_arithmetic_and_result_dtypes() {
    DenseFixture f;
    velocitypp::RunOptions f32_options;
    f32_options.compute_precision = velocitypp::ComputePrecision::float32;
    f32_options.output_dtype = anndatapp::DType::float64;
    velocitypp::RunOptions f64_options;
    f64_options.compute_precision = velocitypp::ComputePrecision::float64;
    f64_options.output_dtype = anndatapp::DType::float64;
    velocitypp::Workspace w32(f.prepared, f32_options), w64(f.prepared, f64_options);
    velocitypp::ReferenceStages32 s32(f.prepared, w32, fit_options());
    velocitypp::ReferenceStages s64(f.prepared, w64, fit_options());
    auto m32 = s32.moments(0, g);
    auto m64 = s64.moments(0, g);
    static_assert(std::is_same_v<decltype(m32.spliced), float*>);
    static_assert(std::is_same_v<decltype(m64.spliced), double*>);
    auto fit32 = s32.fit(m32);
    auto fit64 = s64.fit(m64);
    expect(m32.spliced[0] == 1.0f && m32.spliced[2] == 1.0f &&
           m64.spliced[0] == 1.0 && m64.spliced[2] == 1.0 + 1e-8,
           "FP32 stages convert source values before fitting instead of merely casting FP64 results");
    expect(static_cast<velocitypp::FitStatus>(fit32.status[0]) == velocitypp::FitStatus::singular &&
           static_cast<velocitypp::FitStatus>(fit64.status[0]) == velocitypp::FitStatus::ok,
           "source variation lost in FP32 is reflected in the fit status");
    s32.emit(m32, fit32);
    s64.emit(m64, fit64);
    auto r32 = s32.finish(), r64 = s64.finish();
    expect(r32.gamma.view().dtype() == anndatapp::DType::float32 &&
           r32.offset.view().dtype() == anndatapp::DType::float32 &&
           r32.r2.view().dtype() == anndatapp::DType::float32 &&
           r64.gamma.view().dtype() == anndatapp::DType::float64 &&
           r64.offset.view().dtype() == anndatapp::DType::float64 &&
           r64.r2.view().dtype() == anndatapp::DType::float64,
           "typed steady-state stages retain float or double result storage");

    velocitypp::StochasticRunOptions sr32;
    sr32.compute_precision = velocitypp::ComputePrecision::float32;
    velocitypp::StochasticRunOptions sr64;
    sr64.compute_precision = velocitypp::ComputePrecision::float64;
    velocitypp::StochasticWorkspace sw32(f.prepared, sr32), sw64(f.prepared, sr64);
    velocitypp::StochasticReferenceStages32 ss32(f.prepared, sw32, velocitypp::StochasticOptions{fit_options()});
    velocitypp::StochasticReferenceStages ss64(f.prepared, sw64, velocitypp::StochasticOptions{fit_options()});
    auto im32 = ss32.initial_moments(0, g);
    auto im64 = ss64.initial_moments(0, g);
    auto if32 = ss32.initial_fit(im32);
    auto if64 = ss64.initial_fit(im64);
    ss32.accept_initialization(im32, if32);
    ss64.accept_initialization(im64, if64);
    ss32.resolve_selection();
    ss64.resolve_selection();
    auto sm32 = ss32.moments(0, g);
    auto sm64 = ss64.moments(0, g);
    static_assert(std::is_same_v<decltype(sm32.spliced), float*>);
    static_assert(std::is_same_v<decltype(sm64.spliced), double*>);
    auto scales32 = ss32.scales(sm32);
    auto scales64 = ss64.scales(sm64);
    auto sf32 = ss32.fit(sm32, scales32);
    auto sf64 = ss64.fit(sm64, scales64);
    ss32.emit(sm32, sf32);
    ss64.emit(sm64, sf64);
    auto stochastic32 = ss32.finish(), stochastic64 = ss64.finish();
    expect(stochastic32.gamma.view().dtype() == anndatapp::DType::float32 &&
           stochastic32.initialization.gamma.view().dtype() == anndatapp::DType::float32 &&
           stochastic32.first_scale.view().dtype() == anndatapp::DType::float32 &&
           stochastic64.gamma.view().dtype() == anndatapp::DType::float64 &&
           stochastic64.initialization.gamma.view().dtype() == anndatapp::DType::float64 &&
           stochastic64.first_scale.view().dtype() == anndatapp::DType::float64,
           "typed stochastic stages retain float or double diagnostics and fit results");
}

struct DuplicateCapture {
    std::vector<double> ms,mu,mss,mus,muu;
};
void capture_duplicate_steady(void* context,const velocitypp::OutputBlock& b) {
    auto& out=*static_cast<DuplicateCapture*>(context);
    out.ms=floating_values(b.spliced);out.mu=floating_values(b.unspliced);
}
void capture_duplicate_stochastic(void* context,const velocitypp::StochasticOutputBlock& b) {
    auto& out=*static_cast<DuplicateCapture*>(context);
    out.ms=floating_values(b.spliced);out.mu=floating_values(b.unspliced);
    out.mss=floating_values(b.spliced_second);out.mus=floating_values(b.cross);out.muu=floating_values(b.unspliced_second);
}
template<class C> void duplicate_first_and_second_moments_case(anndatapp::Layout source_layout,bool unsorted,bool sparse_graph) {
    // The expected logical coordinates are exactly S=[1,2], U=[3,4].
    // Weighting the large cancelling entries separately returns .125 rather
    // than .1234567 in both FP32 and FP64 cancellation fixtures.
    const C large=std::is_same_v<C,float> ? C(10000000) : C(4503599627370496.);
    C sv[]{large,C(1)-large,C(2)};
    C uv[]{large,C(3)-large,C(4)};
    std::int32_t source_indices[]{0,0,source_layout==anndatapp::Layout::csr ? 0 : 1};
    if(unsorted) {std::swap(sv[1],sv[2]);std::swap(uv[1],uv[2]);source_indices[1]=1;source_indices[2]=0;}
    const std::int64_t csr_offsets[]{0,2,3},csc_offsets[]{0,3};
    auto* source_offsets = source_layout==anndatapp::Layout::csr ? csr_offsets : csc_offsets;
    auto sa=anndatapp::Array::borrow_sparse(source_layout,2,1,3,sv,source_indices,source_offsets);
    auto ua=anndatapp::Array::borrow_sparse(source_layout,2,1,3,uv,source_indices,source_offsets);
    const float weights[]{.1234567f,0,0,.2345678f},sparse_weights[]{weights[0],weights[3]};
    const std::int32_t graph_indices[]{0,1};const std::int64_t graph_offsets[]{0,1,2};
    auto wa=sparse_graph ? anndatapp::Array::borrow_sparse(anndatapp::Layout::csr,2,2,2,sparse_weights,graph_indices,graph_offsets)
                        : anndatapp::Array::borrow_dense(weights,{2,2},{2,1});
    velocitypp::PreparedInput p{{sa.view(),ua.view(),wa.view()}};
    C ms[2]{},mu[2]{},ms2[2]{},mu2[2]{},mss[2]{},mus[2]{},muu[2]{},ss[2]{},su[2]{};
    velocitypp::reference::moments(p,0,1,ms,mu);
    velocitypp::reference::second_moments(p,0,1,ms2,mu2,mss,mus,muu,ss,su);
    bool direct=true;
    for(std::size_t i=0;i<2;++i) {
        C x=C(i+1),y=C(i+3),w=C(sparse_weights[i]);
        direct=direct && ms[i]==w*x && mu[i]==w*y && ms[i]==ms2[i] && mu[i]==mu2[i]
            && mss[i]==w*(x*x) && mus[i]==w*(x*y) && muu[i]==w*(y*y);
    }
    expect(direct,"first and second moments coalesce sparse expression coordinates before weighting in selected precision");
    velocitypp::RunOptions run;run.compute_precision=std::is_same_v<C,float>?velocitypp::ComputePrecision::float32:velocitypp::ComputePrecision::float64;
    run.velocity=false;run.spliced_moments=run.unspliced_moments=true;run.output_dtype=std::is_same_v<C,float>?anndatapp::DType::float32:anndatapp::DType::float64;
    velocitypp::Workspace workspace(p,run);DuplicateCapture compiled;
    velocitypp::cpu::steady_state(p,workspace,{}, {&compiled,capture_duplicate_steady});
    velocitypp::StochasticRunOptions stochastic_run;stochastic_run.compute_precision=run.compute_precision;stochastic_run.output_dtype=run.output_dtype;
    stochastic_run.velocity=false;stochastic_run.spliced_moments=stochastic_run.unspliced_moments=true;
    stochastic_run.spliced_second=stochastic_run.cross=stochastic_run.unspliced_second=true;
    velocitypp::StochasticWorkspace stochastic_workspace(p,stochastic_run);DuplicateCapture compiled_second;
    velocitypp::cpu::stochastic(p,stochastic_workspace,{}, {&compiled_second,capture_duplicate_stochastic});
    bool optimized=compiled.ms.size()==2 && compiled_second.ms.size()==2;
    for(std::size_t i=0;i<2 && optimized;++i)
        optimized=compiled.ms[i]==double(ms[i]) && compiled.mu[i]==double(mu[i])
            && compiled_second.ms[i]==double(ms[i]) && compiled_second.mu[i]==double(mu[i])
            && compiled_second.mss[i]==double(mss[i]) && compiled_second.mus[i]==double(mus[i]) && compiled_second.muu[i]==double(muu[i]);
    expect(optimized,"explicit compiled CPU moment paths preserve coalescing and first/second consistency");
}
void sparse_expression_duplicates_are_coalesced_before_first_weighting() {
    for(bool sparse_graph:{false,true})for(int storage_order=0;storage_order<3;++storage_order) {
        auto layout=storage_order==0?anndatapp::Layout::csr:anndatapp::Layout::csc;
        bool unsorted=storage_order==2;
        duplicate_first_and_second_moments_case<float>(layout,unsorted,sparse_graph);
        duplicate_first_and_second_moments_case<double>(layout,unsorted,sparse_graph);
    }
}

template<class T> void graph_source_order_case(anndatapp::Layout graph_layout,int expression_layout,bool duplicate_edge) {
    T large=std::is_same_v<T,float>?T(1e8):T(1e16);
    T s[]{large,-large,1},u[]{2*large,-2*large,3};
    const std::int32_t expression_indices[]{0,0,0};const std::int64_t expression_offsets[]{0,1,2,3};
    const std::int32_t csc_expression_indices[]{0,2,1};const std::int64_t csc_expression_offsets[]{0,3};
    T shuffled_s[]{large,1,-large},shuffled_u[]{2*large,3,-2*large};
    auto sa=expression_layout==0?anndatapp::Array::borrow_dense(s,{3,1},{1,1})
           :expression_layout==1?anndatapp::Array::borrow_sparse(anndatapp::Layout::csr,3,1,3,s,expression_indices,expression_offsets)
           :anndatapp::Array::borrow_sparse(anndatapp::Layout::csc,3,1,3,shuffled_s,csc_expression_indices,csc_expression_offsets);
    auto ua=expression_layout==0?anndatapp::Array::borrow_dense(u,{3,1},{1,1})
           :expression_layout==1?anndatapp::Array::borrow_sparse(anndatapp::Layout::csr,3,1,3,u,expression_indices,expression_offsets)
           :anndatapp::Array::borrow_sparse(anndatapp::Layout::csc,3,1,3,shuffled_u,csc_expression_indices,csc_expression_offsets);
    // CSR row0 is [source0,source2,source1,(source2)]. CSC has shuffled
    // destination rows in columns1/2. Duplicate graph edges remain additive.
    const std::int32_t csr_indices[]{0,2,1,1,2},csr_duplicate_indices[]{0,2,1,2,1,2};
    const std::int64_t csr_offsets[]{0,3,4,5},csr_duplicate_offsets[]{0,4,5,6};
    const std::int32_t csc_indices[]{0,1,0,2,0},csc_duplicate_indices[]{0,1,0,2,0,0};
    const std::int64_t csc_offsets[]{0,1,3,5},csc_duplicate_offsets[]{0,1,3,6};
    T weights[]{1,1,1,1,1},csr_duplicate_weights[]{1,1,1,2,1,1},csc_duplicate_weights[]{1,1,1,1,1,2};
    auto wa=graph_layout==anndatapp::Layout::csr
        ?anndatapp::Array::borrow_sparse(graph_layout,3,3,duplicate_edge?6:5,duplicate_edge?csr_duplicate_weights:weights,duplicate_edge?csr_duplicate_indices:csr_indices,duplicate_edge?csr_duplicate_offsets:csr_offsets)
        :anndatapp::Array::borrow_sparse(graph_layout,3,3,duplicate_edge?6:5,duplicate_edge?csc_duplicate_weights:weights,duplicate_edge?csc_duplicate_indices:csc_indices,duplicate_edge?csc_duplicate_offsets:csc_offsets);
    velocitypp::PreparedInput p{{sa.view(),ua.view(),wa.view()}};
    T ms[3]{},mu[3]{},ms2[3]{},mu2[3]{},mss[3]{},mus[3]{},muu[3]{},ss[3]{},su[3]{};
    velocitypp::reference::moments(p,0,1,ms,mu);
    velocitypp::reference::second_moments(p,0,1,ms2,mu2,mss,mus,muu,ss,su);
    T expected=duplicate_edge?T(3):T(1);
    bool ordered=ms[0]==expected && mu[0]==3*expected;
    for(std::size_t i=0;i<3;++i)ordered=ordered && ms[i]==ms2[i] && mu[i]==mu2[i];
    expect(ordered,"graph source-cell order is canonical for dense/sparse expressions and both moment passes");
    velocitypp::RunOptions run;run.compute_precision=std::is_same_v<T,float>?velocitypp::ComputePrecision::float32:velocitypp::ComputePrecision::float64;
    run.output_dtype=std::is_same_v<T,float>?anndatapp::DType::float32:anndatapp::DType::float64;
    run.velocity=false;run.spliced_moments=run.unspliced_moments=true;
    velocitypp::Workspace workspace(p,run);DuplicateCapture first;
    velocitypp::cpu::steady_state(p,workspace,{}, {&first,capture_duplicate_steady});
    velocitypp::StochasticRunOptions stochastic_run;stochastic_run.compute_precision=run.compute_precision;stochastic_run.output_dtype=run.output_dtype;
    stochastic_run.velocity=false;stochastic_run.spliced_moments=stochastic_run.unspliced_moments=true;
    stochastic_run.spliced_second=stochastic_run.cross=stochastic_run.unspliced_second=true;
    velocitypp::StochasticWorkspace stochastic_workspace(p,stochastic_run);DuplicateCapture second;
    velocitypp::cpu::stochastic(p,stochastic_workspace,{}, {&second,capture_duplicate_stochastic});
    bool compiled=first.ms.size()==3 && second.ms.size()==3;
    for(std::size_t i=0;i<3 && compiled;++i)compiled=first.ms[i]==double(ms[i]) && first.mu[i]==double(mu[i]) && second.ms[i]==double(ms[i]) && second.mu[i]==double(mu[i]);
    expect(compiled,"compiled CPU entry points preserve canonical graph order and additive duplicate edges");
}
void shuffled_graphs_preserve_source_order() {
    for(auto layout:{anndatapp::Layout::csr,anndatapp::Layout::csc})
        for(int expression=0;expression<3;++expression)for(bool duplicates:{false,true}) {
            graph_source_order_case<float>(layout,expression,duplicates);
            graph_source_order_case<double>(layout,expression,duplicates);
        }
}

void explicit_float32_output_stores_float_residuals() {
    DenseFixture f;
    velocitypp::RunOptions options;
    options.compute_precision = velocitypp::ComputePrecision::float32;
    options.output_dtype = anndatapp::DType::float32;
    velocitypp::Workspace workspace(f.prepared, options);
    auto output = anndatapp::Array::zeros<float>({n, g});
    auto output_edit = output.edit();
    auto result = velocitypp::reference::steady_state_into(f.prepared, workspace, output_edit, fit_options());
    auto actual = output.view().dense<float>();
    auto gamma = result.gamma.view().dense<float>();
    auto offset = result.offset.view().dense<float>();
    bool ok = output.view().dtype() == anndatapp::DType::float32 &&
              result.gamma.view().dtype() == anndatapp::DType::float32;
    for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < g; ++j) {
        const auto k = i * g + j;
        const float expected = float(f.u[k]) - gamma.data()[j] * float(f.s[k]) - offset.data()[j];
        ok = ok && actual.data()[std::ptrdiff_t(i) * actual.stride(0) + std::ptrdiff_t(j) * actual.stride(1)] ==
                   expected;
    }
    expect(ok, "caller-selected FP32 output stores computed residuals while diagnostics remain FP32");
}

void degenerate_inputs_keep_constant_nonfinite_rank_and_fallback_statuses() {
    velocitypp::FitOptions fit;
    fit.trimming = velocitypp::Trimming::none;
    fit.fit_intercept = true;
    float x[]{1, 2, 3}, constant[]{4, 4, 4}, scratch[3]{};
    auto constant_fit = velocitypp::reference::fit_gene(x, constant, 3, 1, fit, scratch);
    expect(constant_fit.status == velocitypp::FitStatus::constant_response,
           "constant response remains classified as a degenerate fit");

    float nonfinite[]{2, std::numeric_limits<float>::quiet_NaN(), 4};
    auto invalid_fit = velocitypp::reference::fit_gene(x, nonfinite, 3, 1, fit, scratch);
    expect(invalid_fit.status == velocitypp::FitStatus::singular,
           "nonfinite observations produce a singular fit status");

    float dense_s[]{1, 2, 3}, dense_u[]{2, 4, 6}, graph_values[9]{};
    graph_values[0] = graph_values[4] = graph_values[8] = 1;
    auto s_rank1 = anndatapp::Array::borrow_dense(dense_s, {3}, {1});
    auto u_rank2 = anndatapp::Array::borrow_dense(dense_u, {3, 1}, {1, 1});
    auto graph = anndatapp::Array::borrow_dense(graph_values, {3, 3}, {3, 1});
    expect_invalid_argument([&] { velocitypp::PreparedInput invalid({s_rank1.view(), u_rank2.view(), graph.view()}); },
                            "rank-one expression matrices are rejected at input acquisition");

    const float ms[]{1, 1, 1}, mu[]{2, 3, 4}, mss[]{.5f, .5f, .5f}, mus[]{1, 1.5f, 2};
    velocitypp::GeneFit32 initial{.75f, .25f, .8f, velocitypp::FitStatus::ok, true};
    velocitypp::StochasticScales32 scales{1, 1, velocitypp::StochasticStatus::ok};
    velocitypp::StochasticOptions stochastic;
    stochastic.initialization = fit;
    auto rank_fit = velocitypp::reference::stochastic_fit_gene(ms, mu, mss, mus, 3, 1,
        initial, scales, stochastic, nullptr);
    expect(rank_fit.status == velocitypp::StochasticStatus::singular &&
           rank_fit.gamma == initial.gamma && rank_fit.offset == initial.offset,
           "rank-deficient stochastic refinement reports singular and retains initialization fallback");
}

void fp32_stochastic_residual_floor_is_scale_relative() {
    constexpr std::size_t rows = 4;
    const float ms[]{1, 2, 3, 4};
    const float mss[]{1, 4, 9, 16};
    const float mus[]{3, 7, 20, 30};
    const float below_floor_mu[]{2.0f + 1e-6f, 4.0f - 1e-6f,
                                 6.0f + 1e-6f, 8.0f - 1e-6f};
    const float above_floor_mu[]{2.0f + 1e-3f, 4.0f - 1e-3f,
                                 6.0f + 1e-3f, 8.0f - 1e-3f};
    velocitypp::GeneFit32 initial{2, 0, .9f, velocitypp::FitStatus::ok, true};
    velocitypp::StochasticOptions options;
    options.initialization.trimming = velocitypp::Trimming::none;
    float scratch[rows]{};
    auto below = velocitypp::reference::stochastic_scales(ms, below_floor_mu, mss, mus,
        rows, 1, initial, options, scratch);
    expect(below.status == velocitypp::StochasticStatus::zero_scale &&
           below.first > 0 && below.second > 0 &&
           std::isfinite(below.first) && std::isfinite(below.second),
           "positive FP32 residual scales below the relative floor are retained as finite zero-scale diagnostics");

    auto above = velocitypp::reference::stochastic_scales(ms, above_floor_mu, mss, mus,
        rows, 1, initial, options, scratch);
    expect(above.status == velocitypp::StochasticStatus::ok && above.first > 0 && above.second > 0,
           "FP32 residual perturbations above the relative floor remain usable stochastic scales");

    constexpr float factor = 1e-6f;
    float scaled_ms[rows], scaled_mu[rows], scaled_mss[rows], scaled_mus[rows];
    for (std::size_t i = 0; i < rows; ++i) {
        scaled_ms[i] = factor * ms[i];
        scaled_mu[i] = factor * above_floor_mu[i];
        scaled_mss[i] = factor * mss[i];
        scaled_mus[i] = factor * mus[i];
    }
    auto scaled = velocitypp::reference::stochastic_scales(scaled_ms, scaled_mu, scaled_mss, scaled_mus,
        rows, 1, initial, options, scratch);
    expect(scaled.status == velocitypp::StochasticStatus::ok &&
           close(double(scaled.first / factor), double(above.first), 2e-4) &&
           close(double(scaled.second / factor), double(above.second), 2e-4),
           "uniformly rescaled FP32 moments preserve relative residual-floor classification and scales");

    for (std::size_t i = 0; i < rows; ++i) scaled_mu[i] = factor * below_floor_mu[i];
    auto scaled_below = velocitypp::reference::stochastic_scales(scaled_ms, scaled_mu, scaled_mss, scaled_mus,
        rows, 1, initial, options, scratch);
    expect(scaled_below.status == velocitypp::StochasticStatus::zero_scale &&
           scaled_below.first > 0 && std::isfinite(scaled_below.first),
           "a positive below-floor FP32 residual remains zero-scale under uniform input rescaling");
}

void cpu_matches_reference_for_precision(
    DenseFixture& f, velocitypp::ComputePrecision precision, double tol) {
    velocitypp::RunOptions options;
    options.compute_precision = precision;
    options.output_dtype = anndatapp::DType::float64;
    velocitypp::Workspace rw(f.prepared, options), cw(f.prepared, options);
    OutputCapture reference_capture, cpu_capture;
    auto rr = velocitypp::reference::steady_state(f.prepared, rw, fit_options(),
        {&reference_capture, capture_steady});
    auto cr = velocitypp::cpu::steady_state(f.prepared, cw, fit_options(),
        {&cpu_capture, capture_steady});
    expect(fit_results_match(rr, cr, tol) && captures_match(reference_capture, cpu_capture, tol),
           "scalar reference and explicit CPU steady-state paths agree in the selected precision");

    velocitypp::StochasticRunOptions stochastic_run;
    stochastic_run.compute_precision = precision;
    stochastic_run.output_dtype = anndatapp::DType::float64;
    stochastic_run.second_residual = true;
    velocitypp::StochasticWorkspace rsw(f.prepared, stochastic_run), csw(f.prepared, stochastic_run);
    OutputCapture stochastic_reference_capture, stochastic_cpu_capture;
    velocitypp::StochasticOptions stochastic;
    stochastic.initialization = fit_options();
    auto sr = velocitypp::reference::stochastic(f.prepared, rsw, stochastic,
        {&stochastic_reference_capture, capture_stochastic});
    auto sc = velocitypp::cpu::stochastic(f.prepared, csw, stochastic,
        {&stochastic_cpu_capture, capture_stochastic});
    expect(stochastic_results_match(sr, sc, tol) &&
           captures_match(stochastic_reference_capture, stochastic_cpu_capture, tol),
           "scalar reference and explicit CPU stochastic paths agree in the selected precision");
}

void scalar_reference_and_cpu_entries_agree() {
    DenseFixture f;
    cpu_matches_reference_for_precision(f, velocitypp::ComputePrecision::float32, 1e-5);
    cpu_matches_reference_for_precision(f, velocitypp::ComputePrecision::float64, 1e-10);
}

}

int main() {
    default_precision_resolution_and_budgets();
    typed_stages_expose_real_float32_arithmetic_and_result_dtypes();
    sparse_expression_duplicates_are_coalesced_before_first_weighting();
    shuffled_graphs_preserve_source_order();
    explicit_float32_output_stores_float_residuals();
    degenerate_inputs_keep_constant_nonfinite_rank_and_fallback_statuses();
    fp32_stochastic_residual_floor_is_scale_relative();
    scalar_reference_and_cpu_entries_agree();
    if (!failures) std::cout << "CPU precision tests passed\n";
    return failures ? 1 : 0;
}
