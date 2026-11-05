#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

import velocity;

namespace {

int failures = 0;

void expect(bool ok, const char* message) {
    if (ok) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

bool close(double a, double b, double tol = 1e-12) {
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

constexpr std::size_t n = 3, g = 3;
constexpr std::array<float, n * g> spliced{
    1, 2, 3,
    3, 4, 5,
    5, 6, 7
};
constexpr std::array<float, n * g> unspliced{
    2, 4, 6,
    6, 8, 10,
    10, 12, 14
};
constexpr std::array<float, n * n> weights{
    .5f, .5f, 0,
    0, 1, 0,
    .25f, 0, .75f
};
constexpr std::array<double, n * g> expected_ms{
    2, 3, 4,
    3, 4, 5,
    4, 5, 6
};
constexpr std::array<double, n * g> expected_mu{
    4, 6, 8,
    6, 8, 10,
    8, 10, 12
};

struct SparseStorage {
    anndatapp::Layout layout;
    std::size_t rows, columns;
    std::vector<float> values;
    std::vector<std::int32_t> indices;
    std::vector<std::int64_t> offsets;

    anndatapp::Array array() {
        return anndatapp::Array::borrow_sparse(layout, rows, columns, values.size(),
            values.data(), indices.data(), offsets.data());
    }
};

SparseStorage compressed(anndatapp::Layout layout, std::size_t rows, std::size_t columns,
                         const float* dense) {
    SparseStorage out{layout, rows, columns};
    const auto major = layout == anndatapp::Layout::csr ? rows : columns;
    const auto minor = layout == anndatapp::Layout::csr ? columns : rows;
    out.offsets.reserve(major + 1);
    out.offsets.push_back(0);
    for (std::size_t i = 0; i < major; ++i) {
        for (std::size_t j = 0; j < minor; ++j) {
            const auto row = layout == anndatapp::Layout::csr ? i : j;
            const auto column = layout == anndatapp::Layout::csr ? j : i;
            const auto value = dense[row * columns + column];
            if (value == 0) continue;
            out.values.push_back(value);
            out.indices.push_back(static_cast<std::int32_t>(j));
        }
        out.offsets.push_back(static_cast<std::int64_t>(out.values.size()));
    }
    return out;
}

velocity::Input input_from(anndatapp::ArrayView s, anndatapp::ArrayView u,
                           anndatapp::ArrayView w) {
    return velocity::Input{std::move(s), std::move(u), std::move(w)};
}

velocity::FitOptions exact_fit_options() {
    velocity::FitOptions options;
    options.trimming = velocity::Trimming::none;
    options.min_r2 = 0;
    options.min_ratio = 0;
    options.low_signal = velocity::LowSignal::keep_filters;
    return options;
}

void expect_moments(const velocity::PreparedInput& prepared,
                    const std::array<double, n * g>& ms_expected,
                    const std::array<double, n * g>& mu_expected,
                    const char* message) {
    std::array<double, n * g> ms{}, mu{};
    velocity::reference::moments(prepared, 0, g, ms.data(), mu.data());
    bool ok = true;
    for (std::size_t i = 0; i < ms.size(); ++i)
        ok = ok && close(ms[i], ms_expected[i]) && close(mu[i], mu_expected[i]);
    expect(ok, message);
}

void csr_graph_and_sparse_moments() {
    auto s_store = compressed(anndatapp::Layout::csr, n, g, spliced.data());
    auto u_store = compressed(anndatapp::Layout::csc, n, g, unspliced.data());
    auto w_store = compressed(anndatapp::Layout::csr, n, n, weights.data());
    auto sa = s_store.array(), ua = u_store.array(), wa = w_store.array();
    velocity::PreparedInput prepared(input_from(sa.view(), ua.view(), wa.view()));

    expect_moments(prepared, expected_ms, expected_mu,
                   "CSR graph with CSR/CSC expression inputs matches hand-computed W*S and W*U");
}

void csc_graph_and_dense_negative_stride() {
    auto w_store = compressed(anndatapp::Layout::csc, n, n, weights.data());
    auto wa = w_store.array();
    std::array<float, n * g> reverse_s{}, reverse_u{};
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < g; ++j) {
            reverse_s[i * g + j] = spliced[(n - 1 - i) * g + j];
            reverse_u[i * g + j] = unspliced[(n - 1 - i) * g + j];
        }
    auto sa = anndatapp::Array::borrow_dense(reverse_s.data() + (n - 1) * g,
        {n, g}, {-static_cast<std::ptrdiff_t>(g), 1});
    auto ua = anndatapp::Array::borrow_dense(reverse_u.data() + (n - 1) * g,
        {n, g}, {-static_cast<std::ptrdiff_t>(g), 1});
    velocity::PreparedInput prepared(input_from(sa.view(), ua.view(), wa.view()));

    expect_moments(prepared, expected_ms, expected_mu,
                   "CSC graph and negative-row-stride dense inputs preserve logical order");
}

void selected_genes_preserve_order_and_duplicates() {
    auto s_store = compressed(anndatapp::Layout::csc, n, g, spliced.data());
    auto u_store = compressed(anndatapp::Layout::csr, n, g, unspliced.data());
    auto w_store = compressed(anndatapp::Layout::csc, n, n, weights.data());
    auto sa = s_store.array(), ua = u_store.array(), wa = w_store.array();
    velocity::Input input = input_from(sa.view(), ua.view(), wa.view());
    input.genes = anndatapp::Axis::gather({2, 0, 2});
    velocity::PreparedInput prepared(std::move(input));

    expect(prepared.n_genes() == 3 && prepared.source_genes() == 3 &&
           prepared.gene_index(0) == 2 && prepared.gene_index(1) == 0 &&
           prepared.gene_index(2) == 2,
           "selected gene mapping retains requested order and duplicate source indices");
    constexpr std::array<double, n * g> selected_ms{
        4, 2, 4,
        5, 3, 5,
        6, 4, 6
    };
    constexpr std::array<double, n * g> selected_mu{
        8, 4, 8,
        10, 6, 10,
        12, 8, 12
    };
    expect_moments(prepared, selected_ms, selected_mu,
                   "selected sparse moments honor arbitrary gather order and duplicates");
}

void values_are_checked_for_finiteness() {
    auto s = spliced;
    s[4] = std::numeric_limits<float>::quiet_NaN();
    auto sa = anndatapp::Array::borrow_dense(s.data(), {n, g}, {static_cast<std::ptrdiff_t>(g), 1});
    auto ua = anndatapp::Array::borrow_dense(unspliced.data(), {n, g}, {static_cast<std::ptrdiff_t>(g), 1});
    auto w_store = compressed(anndatapp::Layout::csr, n, n, weights.data());
    auto wa = w_store.array();
    velocity::PreparedInput prepared(input_from(sa.view(), ua.view(), wa.view()),
                                      anndatapp::Validation::none);
    expect_invalid_argument([&] { prepared.validate_values(); },
                            "explicit value validation rejects a nonfinite dense expression value");
}

void workspace_respects_the_requested_bound() {
    auto s = anndatapp::Array::borrow_dense(spliced.data(), {n, g}, {static_cast<std::ptrdiff_t>(g), 1});
    auto u = anndatapp::Array::borrow_dense(unspliced.data(), {n, g}, {static_cast<std::ptrdiff_t>(g), 1});
    auto w_store = compressed(anndatapp::Layout::csr, n, n, weights.data());
    auto w = w_store.array();
    velocity::PreparedInput prepared(input_from(s.view(), u.view(), w.view()));

    velocity::RunOptions options;
    options.workspace_bytes = 1u << 20;
    options.block_genes = 64;
    velocity::Workspace workspace(prepared, options);
    expect(workspace.scratch_bytes() <= options.workspace_bytes &&
           workspace.block_genes() <= options.block_genes &&
           workspace.block_genes() <= prepared.n_genes(),
           "workspace reports bounded scratch and a block no larger than the request or gene count");

    options.workspace_bytes = 1;
    expect_invalid_argument([&] { velocity::Workspace too_small(prepared, options); },
                            "workspace rejects a cap too small for one gene and mandatory scratch");
}

struct SinkCapture {
    std::size_t rows = 0, genes = 0, calls = 0;
    std::vector<double> velocity;
    std::vector<std::size_t> blocks;
};

void capture_velocity(void* context, const velocity::OutputBlock& block) {
    auto& capture = *static_cast<SinkCapture*>(context);
    ++capture.calls;
    capture.blocks.push_back(block.count);
    auto values = block.velocity.dense<double>();
    for (std::size_t i = 0; i < capture.rows; ++i)
        for (std::size_t j = 0; j < block.count; ++j)
            capture.velocity[i * capture.genes + block.first + j] =
                values.data()[std::ptrdiff_t(i) * values.stride(0) +
                              std::ptrdiff_t(j) * values.stride(1)];
}

void reference_fit_sink_and_caller_output() {
    auto sa = anndatapp::Array::borrow_dense(spliced.data(), {n, g}, {static_cast<std::ptrdiff_t>(g), 1});
    auto ua = anndatapp::Array::borrow_dense(unspliced.data(), {n, g}, {static_cast<std::ptrdiff_t>(g), 1});
    auto w_store = compressed(anndatapp::Layout::csr, n, n, weights.data());
    auto wa = w_store.array();
    velocity::PreparedInput prepared(input_from(sa.view(), ua.view(), wa.view()));

    velocity::RunOptions run;
    run.block_genes = 2;
    run.output_dtype = velocity::StorageType::float64;
    velocity::Workspace workspace(prepared, run);
    SinkCapture capture{n, g, 0, std::vector<double>(n * g, 99), {}};
    velocity::BlockSink sink{&capture, capture_velocity};
    auto result = velocity::reference::steady_state(
        prepared, workspace, exact_fit_options(), sink);

    const auto gamma = result.gamma.view().dense<double>();
    const auto offset = result.offset.view().dense<double>();
    const auto r2 = result.r2.view().dense<double>();
    const auto selected = result.genes.view().dense<bool>();
    const auto status = result.status.view().dense<std::uint8_t>();
    bool fits_ok = capture.calls == 2;
    for (std::size_t j = 0; j < g; ++j)
        fits_ok = fits_ok && close(gamma.data()[j], 2) && close(offset.data()[j], 0) &&
            r2.data()[j] > .999999 && selected.data()[j] &&
            status.data()[j] == static_cast<std::uint8_t>(velocity::FitStatus::ok);
    expect(fits_ok, "callable reference fit recovers exact proportional steady-state genes");
    bool sink_ok = capture.calls == 2 && capture.blocks.size() == 2 &&
        capture.blocks[0] == 2 && capture.blocks[1] == 1;
    for (double value : capture.velocity) sink_ok = sink_ok && close(value, 0);
    expect(sink_ok, "reference sink consumes reusable velocity blocks synchronously");

    auto caller_velocity = anndatapp::Array::zeros<double>({n, g});
    auto into = velocity::reference::steady_state_into(
        prepared, workspace, caller_velocity.edit(), exact_fit_options());
    auto out = caller_velocity.view().dense<double>();
    bool into_ok = true;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < g; ++j)
            into_ok = into_ok && close(out.data()[std::ptrdiff_t(i) * out.stride(0) +
                                                  std::ptrdiff_t(j) * out.stride(1)], 0);
    const auto gamma_into = into.gamma.view().dense<double>();
    for (std::size_t j = 0; j < g; ++j)
        into_ok = into_ok && close(gamma_into.data()[j], 2);
    expect(into_ok, "caller-owned output route writes expected residual velocities");
}

void masks_and_low_signal_selection() {
    constexpr std::size_t rows = 4, genes = 7;
    constexpr std::array<float, rows * rows> identity{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    constexpr std::array<double, 5> noise_scale{0, .25, .5, .75, 1};
    constexpr std::array<double, rows> noise{1, -1, 1, -1};
    std::vector<float> s(rows * genes), u(rows * genes);
    for (std::size_t i = 0; i < rows; ++i) {
        const double x = double(i + 1);
        for (std::size_t j = 0; j < 5; ++j) {
            s[i * genes + j] = float(x);
            u[i * genes + j] = float(2*x + noise_scale[j]*noise[i]);
        }
        s[i * genes + 5] = 0;
        u[i * genes + 5] = float(2*x);
        s[i * genes + 6] = float(x);
        u[i * genes + 6] = float(2*x + .9*noise[i]);
    }
    auto sa = anndatapp::Array::borrow_dense(s.data(), {rows, genes}, {std::ptrdiff_t(genes), 1});
    auto ua = anndatapp::Array::borrow_dense(u.data(), {rows, genes}, {std::ptrdiff_t(genes), 1});
    auto w_store = compressed(anndatapp::Layout::csr, rows, rows, identity.data());
    auto wa = w_store.array();
    std::array<bool, rows> fit_values{true, false, true, true};
    std::array<bool, genes> gene_values{true, false, true, true, true, true, true};
    auto fit_mask = anndatapp::Array::borrow_dense(fit_values.data(), {rows}, {1});
    auto gene_mask = anndatapp::Array::borrow_dense(gene_values.data(), {genes}, {1});
    velocity::Input input = input_from(sa.view(), ua.view(), wa.view());
    input.genes = anndatapp::Axis::gather({4, 5, 0, 2, 1, 3, 6});
    input.fit_mask = fit_mask.view();
    input.gene_mask = gene_mask.view();
    velocity::PreparedInput prepared(std::move(input));
    velocity::Workspace workspace(prepared);

    auto options = exact_fit_options();
    auto kept = velocity::reference::steady_state(prepared, workspace, options);
    auto gamma = kept.gamma.view().dense<double>();
    auto r2 = kept.r2.view().dense<double>();
    auto selected = kept.genes.view().dense<bool>();
    auto status = kept.status.view().dense<std::uint8_t>();
    const auto no_signal = std::uint8_t(velocity::FitStatus::no_signal);
    const auto ok = std::uint8_t(velocity::FitStatus::ok);
    bool masks_ok = !kept.low_signal &&
        close(gamma.data()[0], 2) && r2.data()[0] > .7 && r2.data()[0] < .99 &&
        selected.data()[0] && status.data()[0] == ok &&
        status.data()[1] == no_signal && !selected.data()[1] &&
        selected.data()[2] && close(gamma.data()[2], 2) &&
        !selected.data()[4] && selected.data()[5] && selected.data()[6];
    expect(masks_ok,
           "fit mask trains on its rows, R2 uses all rows, and source-gene mask follows a gather axis");

    options.min_r2 = .9999;
    auto filtered = velocity::reference::steady_state(prepared, workspace, options);
    auto filtered_selected = filtered.genes.view().dense<bool>();
    std::size_t filtered_count = 0;
    for (std::size_t j = 0; j < genes; ++j) filtered_count += filtered_selected.data()[j];
    expect(!filtered.low_signal && filtered_count == 1 && filtered_selected.data()[2],
           "default keep_filters preserves the per-gene threshold result when fewer than two genes pass");

    options.low_signal = velocity::LowSignal::scvelo_percentile;
    auto percentile = velocity::reference::steady_state(prepared, workspace, options);
    auto percentile_selected = percentile.genes.view().dense<bool>();
    auto percentile_status = percentile.status.view().dense<std::uint8_t>();
    std::size_t selected_count = 0;
    for (std::size_t j = 0; j < genes; ++j) selected_count += percentile_selected.data()[j];
    expect(percentile.low_signal && selected_count == 1 && percentile_selected.data()[2] &&
           percentile_status.data()[1] == no_signal && !percentile_selected.data()[1] &&
           !percentile_selected.data()[4] && !percentile_selected.data()[6],
           "scvelo percentile selects above the valid-fit 80th percentile and excludes invalid or masked genes");
}

void degenerate_gene_reports_no_signal() {
    constexpr std::array<double, 4> x{0, 0, 0, 0};
    constexpr std::array<double, 4> y{1, 2, 3, 4};
    std::array<double, 4> score_scratch{};
    const auto fit = velocity::reference::fit_gene(x.data(), y.data(), x.size(), 1,
        exact_fit_options(), score_scratch.data());
    expect(fit.status == velocity::FitStatus::no_signal && !fit.selected,
           "zero-expression fit has no_signal status and is not selected");
}

std::size_t load_unsigned(const std::byte* data, std::size_t width, std::size_t i) {
    data += width * i;
    if (width == 1) { std::uint8_t x; std::memcpy(&x, data, sizeof x); return x; }
    if (width == 2) { std::uint16_t x; std::memcpy(&x, data, sizeof x); return x; }
    if (width == 4) { std::uint32_t x; std::memcpy(&x, data, sizeof x); return x; }
    if (width == 8) { std::uint64_t x; std::memcpy(&x, data, sizeof x); return static_cast<std::size_t>(x); }
    throw std::runtime_error("unexpected sparse integer width");
}

template<std::size_t N, std::size_t I, std::size_t V>
void expect_csr(const anndatapp::Array& actual,
                const std::array<std::size_t, N>& offsets,
                const std::array<std::uint8_t, I>& indices,
                const std::array<float, V>& values,
                const char* message) {
    const auto view = actual.view();
    bool ok = view.rank() == 2 && view.extent(0) == 4 && view.extent(1) == 4 &&
        view.layout() == anndatapp::Layout::csr && view.dtype() == anndatapp::DType::float32;
    if (ok) {
        const auto sparse = view.sparse_bytes();
        ok = sparse.nnz() == V && offsets.size() == 5 && indices.size() == V &&
            sparse.value_size() == sizeof(float);
        for (std::size_t i = 0; ok && i < offsets.size(); ++i)
            ok = load_unsigned(sparse.offsets(), sparse.offset_size(), i) == offsets[i];
        for (std::size_t i = 0; ok && i < indices.size(); ++i) {
            float value;
            std::memcpy(&value, sparse.values() + i * sizeof(float), sizeof value);
            ok = load_unsigned(sparse.indices(), sparse.index_size(), i) == indices[i] &&
                std::abs(double(value) - double(values[i])) < 1e-7;
        }
    }
    expect(ok, message);
}

void neighborhood_preparation_normalizes_sparse_and_dense_graphs() {
    constexpr std::array<std::size_t, 5> output_offsets{0, 3, 4, 7, 8};
    constexpr std::array<std::uint8_t, 8> output_indices{0, 1, 2, 1, 1, 2, 3, 3};
    constexpr std::array<float, 8> output_values{
        1.f/3, 1.f/3, 1.f/3, 1, 1.f/3, 1.f/3, 1.f/3, 1
    };
    constexpr std::array<float, 12> csr_values{1, 1, 2, -2, 1, 0, 0, -1, 2, 3, -1, 0};
    constexpr std::array<std::int32_t, 12> csr_indices{2, 1, 1, 0, 0, 3, 3, 0, 3, 1, 1, 2};
    constexpr std::array<std::int64_t, 5> csr_offsets{0, 6, 8, 12, 12};
    auto csr = anndatapp::Array::borrow_sparse(anndatapp::Layout::csr, 4, 4,
        csr_values.size(), csr_values.data(), csr_indices.data(), csr_offsets.data());
    auto csr_prepared = velocity::reference::prepare_neighborhood(csr.view());
    expect_csr(csr_prepared, output_offsets, output_indices, output_values,
        "noncanonical CSR coalesces duplicates, inserts one diagonal, and normalizes each row");

    constexpr std::array<float, 12> csc_values{-1, -2, 1, 3, -1, 1, 2, 1, 0, 0, 0, 2};
    constexpr std::array<std::int32_t, 12> csc_indices{1, 0, 0, 2, 2, 0, 0, 0, 2, 0, 1, 2};
    constexpr std::array<std::int64_t, 5> csc_offsets{0, 3, 7, 9, 12};
    auto csc = anndatapp::Array::borrow_sparse(anndatapp::Layout::csc, 4, 4,
        csc_values.size(), csc_values.data(), csc_indices.data(), csc_offsets.data());
    auto csc_prepared = velocity::reference::prepare_neighborhood(csc.view());
    expect_csr(csc_prepared, output_offsets, output_indices, output_values,
        "noncanonical CSC with unsorted rows and duplicates produces the same normalized graph");

    constexpr std::array<float, 16> dense_values{
        -1, 3, 1, 0,
        -2, 0, -2, 0,
        0, 2, -3, 2,
        -1, 0, 0, 0
    };
    auto dense = anndatapp::Array::borrow_dense(dense_values.data(), {4, 4}, {4, 1});
    auto dense_prepared = velocity::reference::prepare_neighborhood(dense.view());
    expect_csr(dense_prepared, output_offsets, output_indices, output_values,
        "dense graph retains positive off-diagonals and one diagonal before row normalization");
}

void neighborhood_preparation_handles_empty_and_nonfinite_graphs() {
    auto empty = anndatapp::Array::csr<float, std::uint8_t, std::uint8_t>(0, 0, 0);
    auto prepared_empty = velocity::reference::prepare_neighborhood(empty.view());
    const auto empty_view = prepared_empty.view();
    const auto sparse = empty_view.sparse_bytes();
    expect(empty_view.extent(0) == 0 && empty_view.extent(1) == 0 &&
           empty_view.layout() == anndatapp::Layout::csr && sparse.nnz() == 0,
           "empty sparse matrix prepares to an empty CSR graph");

    constexpr std::array<float, 4> dense_values{1, 0, 0, 1};
    auto dense_bad_values = dense_values;
    dense_bad_values[1] = std::numeric_limits<float>::infinity();
    auto dense_bad = anndatapp::Array::borrow_dense(dense_bad_values.data(), {2, 2}, {2, 1});
    expect_invalid_argument([&] { (void)velocity::reference::prepare_neighborhood(dense_bad.view()); },
        "dense neighborhood preparation rejects nonfinite connectivity values");

    constexpr std::array<float, 1> sparse_values{std::numeric_limits<float>::quiet_NaN()};
    constexpr std::array<std::int32_t, 1> sparse_indices{1};
    constexpr std::array<std::int64_t, 3> sparse_offsets{0, 1, 1};
    auto sparse_bad = anndatapp::Array::borrow_sparse(anndatapp::Layout::csr, 2, 2, 1,
        sparse_values.data(), sparse_indices.data(), sparse_offsets.data());
    expect_invalid_argument([&] { (void)velocity::reference::prepare_neighborhood(sparse_bad.view()); },
        "sparse neighborhood preparation rejects nonfinite stored connectivity values");
}

}

int main() {
    csr_graph_and_sparse_moments();
    csc_graph_and_dense_negative_stride();
    selected_genes_preserve_order_and_duplicates();
    values_are_checked_for_finiteness();
    workspace_respects_the_requested_bound();
    reference_fit_sink_and_caller_output();
    masks_and_low_signal_selection();
    degenerate_gene_reports_no_signal();
    neighborhood_preparation_normalizes_sparse_and_dense_graphs();
    neighborhood_preparation_handles_empty_and_nonfinite_graphs();
    if (!failures) std::cout << "reference tests passed\n";
    return failures ? 1 : 0;
}
