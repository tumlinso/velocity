#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

import velocitypp;
#include "support.hh"
#include "precision_test_support.hh"

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

constexpr std::size_t n = 4, g = 2;

struct DenseFixture {
    std::array<double, n * g> s{1.0, 1.0, 1.0 + 1e-8, 2.0,
                                1.0 + 2e-8, 3.0, 1.0 + 3e-8, 4.0};
    std::array<double, n * g> u{2.0, 2.1, 3.0, 4.2,
                                4.0, 6.4, 5.0, 8.5};
    std::array<double, n * n> w{};
    anndatapp::Array sa = anndatapp::Array::borrow_dense(s.data(), {n, g}, {std::ptrdiff_t(g), 1});
    anndatapp::Array ua = anndatapp::Array::borrow_dense(u.data(), {n, g}, {std::ptrdiff_t(g), 1});
    anndatapp::Array wa = anndatapp::Array::borrow_dense(w.data(), {n, n}, {std::ptrdiff_t(n), 1});
    Acquired acquired_s{sa.view()}, acquired_u{ua.view()}, acquired_w{wa.view()};
    velocitypp::Input input{};
    velocitypp::PreparedInput prepared{};
    std::vector<std::size_t> indices;
    std::vector<velocitypp::Entry> entries;
    bool ready = false;

    DenseFixture() {
        for (std::size_t i = 0; i < n; ++i) w[i * n + i] = 1;
        input = {acquired_s.matrix, acquired_u.matrix, acquired_w.matrix};
        ready = velocitypp_test::prepare(input, indices, entries, prepared);
    }
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

template<class T> bool fit_results_match(const velocitypp_test::FitStorage<T>& a,
                                         const velocitypp_test::FitStorage<T>& b, double tol) {
    for (std::size_t j = 0; j < a.gamma.size(); ++j)
        if (!close(a.gamma[j], b.gamma[j], tol) || !close(a.offset[j], b.offset[j], tol) ||
            !close(a.r2[j], b.r2[j], tol) || a.selected[j] != b.selected[j] || a.status[j] != b.status[j]) return false;
    return a.result.low_signal == b.result.low_signal;
}

template<class T> bool stochastic_results_match(const velocitypp_test::StochasticStorage<T>& a,
    const velocitypp_test::StochasticStorage<T>& b, double tol) {
    if (!fit_results_match(a.initialization, b.initialization, tol)) return false;
    for (std::size_t j = 0; j < a.gamma.size(); ++j)
        if (!close(a.gamma[j], b.gamma[j], tol) || !close(a.offset[j], b.offset[j], tol) ||
            !close(a.offset2[j], b.offset2[j], tol) || !close(a.first_scale[j], b.first_scale[j], tol) ||
            !close(a.second_scale[j], b.second_scale[j], tol) || !close(a.objective[j], b.objective[j], tol) ||
            a.status[j] != b.status[j]) return false;
    return a.result.refit_all == b.result.refit_all;
}

template<class T> struct MomentCapture {
    std::size_t rows = 0, genes = 0;
    std::vector<T> ms, mu, mss, mus, muu;
};

template<class T> void capture_steady(void* context, const velocitypp::BasicMomentBlock<T>& moments,
                                      const velocitypp::BasicFitBlock<T>&) {
    auto& out = *static_cast<MomentCapture<T>*>(context);
    for (std::size_t i = 0; i < moments.rows; ++i) for (std::size_t j = 0; j < moments.count; ++j) {
        const auto src = std::ptrdiff_t(i) * moments.stride + std::ptrdiff_t(j);
        const auto dst = i * out.genes + moments.first + j;
        out.ms[dst] = moments.spliced[src];
        out.mu[dst] = moments.unspliced[src];
    }
}

template<class T> void capture_stochastic(void* context,
    const velocitypp::BasicStochasticMomentBlock<T>& moments,
    const velocitypp::BasicStochasticFitBlock<T>&) {
    auto& out = *static_cast<MomentCapture<T>*>(context);
    for (std::size_t i = 0; i < moments.rows; ++i) for (std::size_t j = 0; j < moments.count; ++j) {
        const auto src = std::ptrdiff_t(i) * moments.stride + std::ptrdiff_t(j);
        const auto dst = i * out.genes + moments.first + j;
        out.ms[dst] = moments.spliced[src];
        out.mu[dst] = moments.unspliced[src];
        out.mss[dst] = moments.spliced_second[src];
        out.mus[dst] = moments.cross[src];
        out.muu[dst] = moments.unspliced_second[src];
    }
}

template<class T> bool moments_match(const MomentCapture<T>& a, const MomentCapture<T>& b, double tol) {
    if (a.ms.size() != b.ms.size() || a.mu.size() != b.mu.size() || a.mss.size() != b.mss.size() ||
        a.mus.size() != b.mus.size() || a.muu.size() != b.muu.size()) return false;
    for (std::size_t i = 0; i < a.ms.size(); ++i)
        if (!close(a.ms[i], b.ms[i], tol) || !close(a.mu[i], b.mu[i], tol)) return false;
    for (std::size_t i = 0; i < a.mss.size(); ++i)
        if (!close(a.mss[i], b.mss[i], tol) || !close(a.mus[i], b.mus[i], tol) || !close(a.muu[i], b.muu[i], tol)) return false;
    return true;
}

void caller_workspace_sizes_and_capacity_checks() {
    DenseFixture f;
    expect(f.ready, "dense precision fixture prepares into caller-owned descriptors");
    if (!f.ready) return;
    velocitypp::RunOptions run;
    velocitypp::WorkspaceSize f32_size{}, f64_size{}, stochastic_size{};
    expect(bool(velocitypp::workspace_size<float>(f.prepared, run, false, f32_size)) &&
           bool(velocitypp::workspace_size<double>(f.prepared, run, false, f64_size)) &&
           f32_size.scalars > 0 && f64_size.scalars > 0 && f32_size.masks == n && f64_size.masks == n,
           "typed workspace sizing reports scalar and mask capacities before allocation");
    std::vector<float> scalars(f32_size.scalars);
    std::vector<std::uint8_t> mask(f32_size.masks);
    velocitypp::Workspace<float> workspace;
    expect(bool(velocitypp::bind_workspace(f.prepared, run, false, scalars.data(), scalars.size(),
                                           mask.data(), mask.size(), workspace)) &&
           workspace.block > 0 && workspace.scratch_bytes == f32_size.bytes,
           "caller buffers bind to the requested FP32 reference workspace");
    auto short_scalar = velocitypp::bind_workspace(f.prepared, run, false, scalars.data(), scalars.size() - 1,
                                                    mask.data(), mask.size(), workspace);
    expect(short_scalar.code == velocitypp::Code::capacity && short_scalar.required == f32_size.scalars,
           "undersized scalar workspace reports exact required element capacity");
    auto short_mask = velocitypp::bind_workspace(f.prepared, run, false, scalars.data(), scalars.size(),
                                                  mask.data(), mask.size() - 1, workspace);
    expect(short_mask.code == velocitypp::Code::capacity && short_mask.location == 1 && short_mask.required == n,
           "undersized fit-mask storage reports its separate required capacity");
    run.block_genes = 1;
    expect(bool(velocitypp::workspace_size<float>(f.prepared, run, true, stochastic_size)) &&
           stochastic_size.block == 1 && stochastic_size.scalars > f32_size.scalars,
           "stochastic workspace sizing accounts for second moments and source columns");
    std::vector<float> stochastic_scalars(stochastic_size.scalars);
    std::vector<std::uint8_t> stochastic_mask(stochastic_size.masks);
    velocitypp::Workspace<float> stochastic_workspace;
    expect(bool(velocitypp::bind_workspace(f.prepared, run, true, stochastic_scalars.data(),
                                           stochastic_scalars.size(), stochastic_mask.data(),
                                           stochastic_mask.size(), stochastic_workspace)) &&
           stochastic_workspace.stochastic,
           "caller buffers bind to the typed stochastic workspace");
}

void typed_stages_expose_real_float32_arithmetic_and_result_storage() {
    DenseFixture f;
    expect(f.ready, "dense precision fixture prepares before typed stage execution");
    if (!f.ready) return;
    velocitypp_test::WorkspaceStorage<float> w32;
    velocitypp_test::WorkspaceStorage<double> w64;
    expect(w32.bind(f.prepared, g, false) && w64.bind(f.prepared, g, false),
           "float and double stage workspaces bind from caller storage");
    if (!w32.workspace.block || !w64.workspace.block) return;
    velocitypp_test::FitStorage<float> r32(g);
    velocitypp_test::FitStorage<double> r64(g);
    velocitypp::ReferenceStages32 s32{};
    velocitypp::ReferenceStages s64{};
    expect(bool(velocitypp::bind_stages(f.prepared, w32.workspace, fit_options(), r32.result, s32)) &&
           bool(velocitypp::bind_stages(f.prepared, w64.workspace, fit_options(), r64.result, s64)),
           "caller-owned typed fit diagnostics bind to the corresponding reference stages");
    auto m32 = s32.moments(0, g);
    auto m64 = s64.moments(0, g);
    static_assert(std::is_same_v<decltype(m32.spliced), float*>);
    static_assert(std::is_same_v<decltype(m64.spliced), double*>);
    auto f32 = s32.fit(m32);
    auto f64 = s64.fit(m64);
    expect(m32.spliced[0] == 1.0f && m32.spliced[2] == 1.0f &&
           m64.spliced[0] == 1.0 && m64.spliced[2] == 1.0 + 1e-8,
           "FP32 stages convert source values before fitting instead of merely casting FP64 results");
    expect(static_cast<velocitypp::FitStatus>(f32.status[0]) == velocitypp::FitStatus::singular &&
           static_cast<velocitypp::FitStatus>(f64.status[0]) == velocitypp::FitStatus::ok,
           "source variation lost in FP32 is reflected in fit status");
    s32.emit(m32, f32);
    s64.emit(m64, f64);
    auto out32 = s32.finish();
    auto out64 = s64.finish();
    static_assert(std::is_same_v<decltype(out32.gamma), float*>);
    static_assert(std::is_same_v<decltype(out64.gamma), double*>);
    expect(out32.gamma == r32.gamma.data() && out64.gamma == r64.gamma.data() &&
           out32.status == r32.status.data() && out64.status == r64.status.data(),
           "typed diagnostics remain in caller-provided compute precision storage");

    velocitypp_test::WorkspaceStorage<float> sw32;
    velocitypp_test::WorkspaceStorage<double> sw64;
    expect(sw32.bind(f.prepared, g, true) && sw64.bind(f.prepared, g, true),
           "float and double stochastic workspaces bind from caller storage");
    velocitypp_test::StochasticStorage<float> sr32(g);
    velocitypp_test::StochasticStorage<double> sr64(g);
    velocitypp::StochasticReferenceStages32 ss32{};
    velocitypp::StochasticReferenceStages ss64{};
    velocitypp::StochasticOptions options;
    options.initialization = fit_options();
    expect(bool(velocitypp::bind_stages(f.prepared, sw32.workspace, options, sr32.result, ss32)) &&
           bool(velocitypp::bind_stages(f.prepared, sw64.workspace, options, sr64.result, ss64)),
           "typed stochastic results bind to caller-owned fit and refinement arrays");
    auto im32 = ss32.initial_moments(0, g);
    auto im64 = ss64.initial_moments(0, g);
    auto initial32 = ss32.initial_fit(im32);
    auto initial64 = ss64.initial_fit(im64);
    ss32.accept_initialization(im32, initial32);
    ss64.accept_initialization(im64, initial64);
    ss32.resolve_selection();
    ss64.resolve_selection();
    auto sm32 = ss32.moments(0, g);
    auto sm64 = ss64.moments(0, g);
    static_assert(std::is_same_v<decltype(sm32.spliced), float*>);
    static_assert(std::is_same_v<decltype(sm64.spliced), double*>);
    auto scales32 = ss32.scales(sm32);
    auto scales64 = ss64.scales(sm64);
    auto final32 = ss32.fit(sm32, scales32);
    auto final64 = ss64.fit(sm64, scales64);
    ss32.emit(sm32, final32);
    ss64.emit(sm64, final64);
    auto result32 = ss32.finish();
    auto result64 = ss64.finish();
    expect(result32.gamma == sr32.gamma.data() && result32.initialization.gamma == sr32.initialization.gamma.data() &&
           result32.first_scale == sr32.first_scale.data() && result64.gamma == sr64.gamma.data() &&
           result64.initialization.gamma == sr64.initialization.gamma.data() && result64.first_scale == sr64.first_scale.data(),
           "stochastic diagnostics remain in their caller-owned float or double arrays");
}

template<class C> void duplicate_first_and_second_moments_case(
    anndatapp::Layout source_layout, bool unsorted, bool sparse_graph) {
    const C large = std::is_same_v<C, float> ? C(10000000) : C(4503599627370496.);
    C sv[]{large, C(1) - large, C(2)};
    C uv[]{large, C(3) - large, C(4)};
    std::int32_t source_indices[]{0, 0, source_layout == anndatapp::Layout::csr ? 0 : 1};
    if (unsorted) {
        std::swap(sv[1], sv[2]); std::swap(uv[1], uv[2]);
        source_indices[1] = 1; source_indices[2] = 0;
    }
    const std::int64_t csr_offsets[]{0, 2, 3}, csc_offsets[]{0, 3};
    auto* source_offsets = source_layout == anndatapp::Layout::csr ? csr_offsets : csc_offsets;
    auto sa = anndatapp::Array::borrow_sparse(source_layout, 2, 1, 3, sv, source_indices, source_offsets);
    auto ua = anndatapp::Array::borrow_sparse(source_layout, 2, 1, 3, uv, source_indices, source_offsets);
    const float dense_weights[]{.1234567f, 0, 0, .2345678f};
    const float sparse_weights[]{dense_weights[0], dense_weights[3]};
    const std::int32_t graph_indices[]{0, 1};
    const std::int64_t graph_offsets[]{0, 1, 2};
    auto wa = sparse_graph ? anndatapp::Array::borrow_sparse(anndatapp::Layout::csr, 2, 2, 2,
            sparse_weights, graph_indices, graph_offsets)
        : anndatapp::Array::borrow_dense(dense_weights, {2, 2}, {2, 1});
    Acquired as{sa.view()}, au{ua.view()}, aw{wa.view()};
    velocitypp::Input input{as.matrix, au.matrix, aw.matrix};
    velocitypp::PreparedInput prepared{};
    std::vector<std::size_t> indices;
    std::vector<velocitypp::Entry> entries;
    expect(velocitypp_test::prepare(input, indices, entries, prepared),
           "sparse duplicate fixture prepares caller-owned column and graph indices");
    if (!prepared.rows) return;

    C ms[2]{}, mu[2]{}, ms2[2]{}, mu2[2]{}, mss[2]{}, mus[2]{}, muu[2]{}, source_s[2]{}, source_u[2]{};
    velocitypp::reference::moments(prepared, 0, 1, ms, mu);
    velocitypp::reference::second_moments(prepared, 0, 1, ms2, mu2, mss, mus, muu, source_s, source_u);
    bool direct = true;
    for (std::size_t i = 0; i < 2; ++i) {
        const C x = C(i + 1), y = C(i + 3), weight = C(sparse_weights[i]);
        direct = direct && ms[i] == weight * x && mu[i] == weight * y && ms[i] == ms2[i] && mu[i] == mu2[i] &&
            mss[i] == weight * (x * x) && mus[i] == weight * (x * y) && muu[i] == weight * (y * y);
    }
    expect(direct, "first and second moments coalesce sparse expression coordinates before weighting in selected precision");

    MomentCapture<C> steady{2, 1, std::vector<C>(2), std::vector<C>(2)};
    velocitypp_test::WorkspaceStorage<C> workspace;
    velocitypp_test::FitStorage<C> result(1);
    expect(workspace.bind(prepared, 1, false) && bool(velocitypp::cpu::steady_state(
        prepared, workspace.workspace, result.result, fit_options(), {&steady, capture_steady<C>})),
        "explicit CPU deterministic path runs with caller-bound scratch and moment capture");
    MomentCapture<C> stochastic_capture{2, 1, std::vector<C>(2), std::vector<C>(2), std::vector<C>(2),
                                        std::vector<C>(2), std::vector<C>(2)};
    velocitypp_test::WorkspaceStorage<C> stochastic_workspace;
    velocitypp_test::StochasticStorage<C> stochastic_result(1);
    velocitypp::StochasticOptions stochastic_options;
    stochastic_options.initialization = fit_options();
    expect(stochastic_workspace.bind(prepared, 1, true, 1u << 20, true) && bool(velocitypp::cpu::stochastic(
        prepared, stochastic_workspace.workspace, stochastic_result.result, stochastic_options,
        {&stochastic_capture, capture_stochastic<C>})),
        "explicit CPU second-moment path runs with caller-bound stochastic scratch");
    bool compiled = true;
    for (std::size_t i = 0; i < 2; ++i)
        compiled = compiled && steady.ms[i] == ms[i] && steady.mu[i] == mu[i] &&
            stochastic_capture.ms[i] == ms[i] && stochastic_capture.mu[i] == mu[i] &&
            stochastic_capture.mss[i] == mss[i] && stochastic_capture.mus[i] == mus[i] &&
            stochastic_capture.muu[i] == muu[i];
    expect(compiled, "compiled CPU moments preserve coalescing and first/second consistency");
}

void sparse_expression_duplicates_are_coalesced_before_first_weighting() {
    for (bool sparse_graph : {false, true}) for (int order = 0; order < 3; ++order) {
        const auto layout = order == 0 ? anndatapp::Layout::csr : anndatapp::Layout::csc;
        const bool unsorted = order == 2;
        duplicate_first_and_second_moments_case<float>(layout, unsorted, sparse_graph);
        duplicate_first_and_second_moments_case<double>(layout, unsorted, sparse_graph);
    }
}

template<class T> void graph_source_order_case(
    anndatapp::Layout graph_layout, int expression_layout, bool duplicate_edge) {
    const T large = std::is_same_v<T, float> ? T(1e8) : T(1e16);
    T dense_s[]{large, -large, 1}, dense_u[]{2 * large, -2 * large, 3};
    const std::int32_t csr_expression_indices[]{0, 0, 0};
    const std::int64_t csr_expression_offsets[]{0, 1, 2, 3};
    const std::int32_t csc_expression_indices[]{0, 2, 1};
    const std::int64_t csc_expression_offsets[]{0, 3};
    T shuffled_s[]{large, 1, -large}, shuffled_u[]{2 * large, 3, -2 * large};
    auto sa = expression_layout == 0 ? anndatapp::Array::borrow_dense(dense_s, {3, 1}, {1, 1})
        : expression_layout == 1 ? anndatapp::Array::borrow_sparse(anndatapp::Layout::csr, 3, 1, 3,
              dense_s, csr_expression_indices, csr_expression_offsets)
        : anndatapp::Array::borrow_sparse(anndatapp::Layout::csc, 3, 1, 3,
              shuffled_s, csc_expression_indices, csc_expression_offsets);
    auto ua = expression_layout == 0 ? anndatapp::Array::borrow_dense(dense_u, {3, 1}, {1, 1})
        : expression_layout == 1 ? anndatapp::Array::borrow_sparse(anndatapp::Layout::csr, 3, 1, 3,
              dense_u, csr_expression_indices, csr_expression_offsets)
        : anndatapp::Array::borrow_sparse(anndatapp::Layout::csc, 3, 1, 3,
              shuffled_u, csc_expression_indices, csc_expression_offsets);
    const std::int32_t csr_indices[]{0, 2, 1, 1, 2}, csr_duplicate_indices[]{0, 2, 1, 2, 1, 2};
    const std::int64_t csr_offsets[]{0, 3, 4, 5}, csr_duplicate_offsets[]{0, 4, 5, 6};
    const std::int32_t csc_indices[]{0, 1, 0, 2, 0}, csc_duplicate_indices[]{0, 1, 0, 2, 0, 0};
    const std::int64_t csc_offsets[]{0, 1, 3, 5}, csc_duplicate_offsets[]{0, 1, 3, 6};
    T weights[]{1, 1, 1, 1, 1}, csr_duplicate_weights[]{1, 1, 1, 2, 1, 1};
    T csc_duplicate_weights[]{1, 1, 1, 1, 1, 2};
    auto wa = graph_layout == anndatapp::Layout::csr
        ? anndatapp::Array::borrow_sparse(graph_layout, 3, 3, duplicate_edge ? 6 : 5,
            duplicate_edge ? csr_duplicate_weights : weights,
            duplicate_edge ? csr_duplicate_indices : csr_indices,
            duplicate_edge ? csr_duplicate_offsets : csr_offsets)
        : anndatapp::Array::borrow_sparse(graph_layout, 3, 3, duplicate_edge ? 6 : 5,
            duplicate_edge ? csc_duplicate_weights : weights,
            duplicate_edge ? csc_duplicate_indices : csc_indices,
            duplicate_edge ? csc_duplicate_offsets : csc_offsets);
    Acquired as{sa.view()}, au{ua.view()}, aw{wa.view()};
    velocitypp::Input input{as.matrix, au.matrix, aw.matrix};
    velocitypp::PreparedInput prepared{};
    std::vector<std::size_t> indices;
    std::vector<velocitypp::Entry> entries;
    expect(velocitypp_test::prepare(input, indices, entries, prepared),
           "mixed-layout graph fixture prepares canonical traversal arrays");
    if (!prepared.rows) return;
    T ms[3]{}, mu[3]{}, ms2[3]{}, mu2[3]{}, mss[3]{}, mus[3]{}, muu[3]{}, source_s[3]{}, source_u[3]{};
    velocitypp::reference::moments(prepared, 0, 1, ms, mu);
    velocitypp::reference::second_moments(prepared, 0, 1, ms2, mu2, mss, mus, muu, source_s, source_u);
    const T expected = duplicate_edge ? T(3) : T(1);
    bool ordered = ms[0] == expected && mu[0] == 3 * expected;
    for (std::size_t i = 0; i < 3; ++i) ordered = ordered && ms[i] == ms2[i] && mu[i] == mu2[i];
    expect(ordered, "graph source-cell order is canonical for dense/sparse expressions and both moment passes");

    MomentCapture<T> capture{3, 1, std::vector<T>(3), std::vector<T>(3)};
    velocitypp_test::WorkspaceStorage<T> workspace;
    velocitypp_test::FitStorage<T> result(1);
    expect(workspace.bind(prepared, 1, false) && bool(velocitypp::cpu::steady_state(
        prepared, workspace.workspace, result.result, fit_options(), {&capture, capture_steady<T>})),
        "explicit CPU path processes shuffled graph and expression layouts");
    bool same = true;
    for (std::size_t i = 0; i < 3; ++i) same = same && capture.ms[i] == ms[i] && capture.mu[i] == mu[i];
    expect(same, "compiled CPU entry preserves canonical graph order and additive duplicate edges");
}

void shuffled_graphs_preserve_source_order() {
    for (auto graph : {anndatapp::Layout::csr, anndatapp::Layout::csc})
        for (int expression = 0; expression < 3; ++expression) for (bool duplicates : {false, true}) {
            graph_source_order_case<float>(graph, expression, duplicates);
            graph_source_order_case<double>(graph, expression, duplicates);
        }
}

void explicit_float32_output_stores_float_residuals() {
    DenseFixture f;
    expect(f.ready, "dense fixture prepares before residual output");
    if (!f.ready) return;
    velocitypp_test::WorkspaceStorage<float> workspace;
    velocitypp_test::FitStorage<float> result(g);
    std::vector<float> output(n * g);
    expect(workspace.bind(f.prepared, g, false) && bool(velocitypp::reference::steady_state_into(
        f.prepared, workspace.workspace, result.result, output.data(), output.size(), g, fit_options())),
        "FP32 compute writes residuals into caller-owned FP32 output storage");
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < g; ++j) {
        const auto k = i * g + j;
        const float expected = float(f.u[k]) - result.gamma[j] * float(f.s[k]) - result.offset[j];
        ok = ok && output[k] == expected;
    }
    expect(ok, "caller-selected FP32 output stores computed residuals while diagnostics remain FP32");
}

void degenerate_inputs_keep_constant_nonfinite_rank_and_fallback_statuses() {
    velocitypp::FitOptions fit = fit_options();
    const float x[]{1, 2, 3}, constant[]{4, 4, 4}, scratch[3]{};
    auto constant_fit = velocitypp::reference::fit_gene(x, constant, 3, 1, fit, const_cast<float*>(scratch));
    expect(constant_fit.status == velocitypp::FitStatus::constant_response,
           "constant response remains classified as a degenerate fit");
    const float nonfinite[]{2, std::numeric_limits<float>::quiet_NaN(), 4};
    auto invalid_fit = velocitypp::reference::fit_gene(x, nonfinite, 3, 1, fit, const_cast<float*>(scratch));
    expect(invalid_fit.status == velocitypp::FitStatus::singular,
           "nonfinite observations produce a singular fit status");

    float dense_s[]{1, 2, 3}, dense_u[]{2, 4, 6}, graph_values[9]{};
    graph_values[0] = graph_values[4] = graph_values[8] = 1;
    auto s_rank1 = anndatapp::Array::borrow_dense(dense_s, {3}, {1});
    auto u_rank2 = anndatapp::Array::borrow_dense(dense_u, {3, 1}, {1, 1});
    auto graph = anndatapp::Array::borrow_dense(graph_values, {3, 3}, {3, 1});
    Acquired as{s_rank1.view()}, au{u_rank2.view()}, aw{graph.view()};
    velocitypp::Input invalid{as.matrix, au.matrix, aw.matrix};
    expect(velocitypp::validate(invalid).code == velocitypp::Code::shape,
           "rank-one expression matrices are rejected by explicit input validation");

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
    const float below_floor_mu[]{2.0f + 1e-6f, 4.0f - 1e-6f, 6.0f + 1e-6f, 8.0f - 1e-6f};
    const float above_floor_mu[]{2.0f + 1e-3f, 4.0f - 1e-3f, 6.0f + 1e-3f, 8.0f - 1e-3f};
    velocitypp::GeneFit32 initial{2, 0, .9f, velocitypp::FitStatus::ok, true};
    velocitypp::StochasticOptions options;
    options.initialization.trimming = velocitypp::Trimming::none;
    float scratch[rows]{};
    auto below = velocitypp::reference::stochastic_scales(ms, below_floor_mu, mss, mus, rows, 1, initial, options, scratch);
    expect(below.status == velocitypp::StochasticStatus::zero_scale && below.first > 0 && below.second > 0 &&
           std::isfinite(below.first) && std::isfinite(below.second),
           "positive FP32 residual scales below the relative floor remain finite zero-scale diagnostics");
    auto above = velocitypp::reference::stochastic_scales(ms, above_floor_mu, mss, mus, rows, 1, initial, options, scratch);
    expect(above.status == velocitypp::StochasticStatus::ok && above.first > 0 && above.second > 0,
           "FP32 residual perturbations above the relative floor remain usable stochastic scales");

    constexpr float factor = 1e-6f;
    float scaled_ms[rows], scaled_mu[rows], scaled_mss[rows], scaled_mus[rows];
    for (std::size_t i = 0; i < rows; ++i) {
        scaled_ms[i] = factor * ms[i]; scaled_mu[i] = factor * above_floor_mu[i];
        scaled_mss[i] = factor * mss[i]; scaled_mus[i] = factor * mus[i];
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
    expect(scaled_below.status == velocitypp::StochasticStatus::zero_scale && scaled_below.first > 0 &&
           std::isfinite(scaled_below.first),
           "positive below-floor FP32 residual remains zero-scale under uniform input rescaling");
}

template<class T> void cpu_matches_reference_for_precision(DenseFixture& f, double tol) {
    velocitypp_test::WorkspaceStorage<T> reference_workspace, cpu_workspace;
    expect(reference_workspace.bind(f.prepared, g, false) && cpu_workspace.bind(f.prepared, g, false),
           "reference and CPU deterministic workspaces bind from independent caller buffers");
    velocitypp_test::FitStorage<T> reference_result(g), cpu_result(g);
    MomentCapture<T> reference_moments{n, g, std::vector<T>(n * g), std::vector<T>(n * g)};
    MomentCapture<T> cpu_moments{n, g, std::vector<T>(n * g), std::vector<T>(n * g)};
    const auto ref = velocitypp::reference::steady_state(f.prepared, reference_workspace.workspace,
        reference_result.result, fit_options(), {&reference_moments, capture_steady<T>});
    const auto cpu = velocitypp::cpu::steady_state(f.prepared, cpu_workspace.workspace,
        cpu_result.result, fit_options(), {&cpu_moments, capture_steady<T>});
    expect(bool(ref) && bool(cpu) && fit_results_match(reference_result, cpu_result, tol) &&
           moments_match(reference_moments, cpu_moments, tol),
           "scalar reference and explicit CPU steady-state paths agree in the selected typed arithmetic");

    velocitypp_test::WorkspaceStorage<T> stochastic_reference_workspace, stochastic_cpu_workspace;
    expect(stochastic_reference_workspace.bind(f.prepared, g, true, 1u << 20, true) &&
           stochastic_cpu_workspace.bind(f.prepared, g, true, 1u << 20, true),
           "reference and CPU stochastic workspaces bind from independent caller buffers");
    velocitypp_test::StochasticStorage<T> stochastic_reference(g), stochastic_cpu(g);
    MomentCapture<T> stochastic_reference_moments{n, g, std::vector<T>(n * g), std::vector<T>(n * g),
        std::vector<T>(n * g), std::vector<T>(n * g), std::vector<T>(n * g)};
    MomentCapture<T> stochastic_cpu_moments{n, g, std::vector<T>(n * g), std::vector<T>(n * g),
        std::vector<T>(n * g), std::vector<T>(n * g), std::vector<T>(n * g)};
    velocitypp::StochasticOptions stochastic_options;
    stochastic_options.initialization = fit_options();
    const auto sr = velocitypp::reference::stochastic(f.prepared, stochastic_reference_workspace.workspace,
        stochastic_reference.result, stochastic_options, {&stochastic_reference_moments, capture_stochastic<T>});
    const auto sc = velocitypp::cpu::stochastic(f.prepared, stochastic_cpu_workspace.workspace,
        stochastic_cpu.result, stochastic_options, {&stochastic_cpu_moments, capture_stochastic<T>});
    expect(bool(sr) && bool(sc) && stochastic_results_match(stochastic_reference, stochastic_cpu, tol) &&
           moments_match(stochastic_reference_moments, stochastic_cpu_moments, tol),
           "scalar reference and explicit CPU stochastic paths agree in the selected typed arithmetic");
}

void scalar_reference_and_cpu_entries_agree() {
    DenseFixture f;
    expect(f.ready, "dense fixture prepares before reference/CPU comparison");
    if (!f.ready) return;
    cpu_matches_reference_for_precision<float>(f, 1e-5);
    cpu_matches_reference_for_precision<double>(f, 1e-10);
}

}

int main() {
    caller_workspace_sizes_and_capacity_checks();
    typed_stages_expose_real_float32_arithmetic_and_result_storage();
    sparse_expression_duplicates_are_coalesced_before_first_weighting();
    shuffled_graphs_preserve_source_order();
    explicit_float32_output_stores_float_residuals();
    degenerate_inputs_keep_constant_nonfinite_rank_and_fallback_statuses();
    fp32_stochastic_residual_floor_is_scale_relative();
    scalar_reference_and_cpu_entries_agree();
    if (!failures) std::cout << "CPU precision tests passed\n";
    return failures ? 1 : 0;
}
