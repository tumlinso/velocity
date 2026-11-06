#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
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

bool close(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}

struct Prepared {
    std::vector<std::size_t> indices;
    std::vector<velocitypp::Entry> entries;
    velocitypp::PreparedInput value;
    explicit Prepared(const velocitypp::Input& input) {
        velocitypp::PreparationSize size;
        expect(bool(velocitypp::validate(input)), "input validation succeeds");
        expect(bool(velocitypp::preparation_size(input, size)), "preparation sizing succeeds");
        indices.resize(size.indices); entries.resize(size.entries);
        expect(bool(velocitypp::prepare(input, {indices.data(), indices.size(), entries.data(), entries.size()}, value)),
               "caller-owned preparation succeeds");
    }
};

template<class T> struct Storage {
    velocitypp::Workspace<T> workspace;
    std::vector<T> scalars;
    std::vector<std::uint8_t> masks;
    Storage(const velocitypp::PreparedInput& input, velocitypp::RunOptions options = {}) {
        velocitypp::WorkspaceSize size;
        expect(bool(velocitypp::workspace_size<T>(input, options, false, size)), "workspace sizing succeeds");
        scalars.resize(size.scalars); masks.resize(size.masks);
        expect(bool(velocitypp::bind_workspace(input, options, false, scalars.data(), scalars.size(),
                                                masks.data(), masks.size(), workspace)), "workspace binding succeeds");
    }
};

template<class T> struct Result {
    std::vector<T> gamma, offset, r2;
    std::unique_ptr<bool[]> genes;
    std::vector<std::uint8_t> status;
    velocitypp::BasicFitResult<T> value;
    explicit Result(std::size_t n) : gamma(n), offset(n), r2(n), genes(new bool[n]{}), status(n),
        value{gamma.data(), offset.data(), r2.data(), genes.get(), status.data(), n} {}
};

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

struct Acquired {
    std::optional<velocitypp::DenseInput> dense;
    std::optional<velocitypp::SparseInput> sparse;
    velocitypp::AcquiredMatrix value;
    explicit Acquired(anndatapp::ArrayView view) {
        value.dtype = view.dtype(); value.layout = view.layout();
        if (value.layout == anndatapp::Layout::dense) {
            dense.emplace(view.bytes()); value.dense = &*dense;
        } else {
            sparse.emplace(view.sparse_bytes()); value.sparse = &*sparse;
            value.index_type = view.index_dtype(); value.offset_type = view.offset_dtype();
        }
    }
};
struct InputOwner {
    Acquired s, u, w;
    velocitypp::Input value;
    InputOwner(anndatapp::ArrayView sv, anndatapp::ArrayView uv, anndatapp::ArrayView wv)
        : s(sv), u(uv), w(wv), value{s.value, u.value, w.value} {}
};

velocitypp::FitOptions exact_fit_options() {
    velocitypp::FitOptions options;
    options.trimming = velocitypp::Trimming::none;
    options.min_r2 = 0;
    options.min_ratio = 0;
    options.low_signal = velocitypp::LowSignal::keep_filters;
    return options;
}

void expect_moments(const velocitypp::PreparedInput& prepared,
                    const std::array<double, n * g>& ms_expected,
                    const std::array<double, n * g>& mu_expected,
                    const char* message) {
    std::array<double, n * g> ms{}, mu{};
    velocitypp::reference::moments(prepared, 0, g, ms.data(), mu.data());
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
    InputOwner input_owner(sa.view(), ua.view(), wa.view());
    Prepared owner(input_owner.value);
    auto& prepared = owner.value;

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
    InputOwner input_owner(sa.view(), ua.view(), wa.view());
    Prepared owner(input_owner.value);
    auto& prepared = owner.value;

    expect_moments(prepared, expected_ms, expected_mu,
                   "CSC graph and negative-row-stride dense inputs preserve logical order");
}

void selected_genes_preserve_order_and_duplicates() {
    auto s_store = compressed(anndatapp::Layout::csc, n, g, spliced.data());
    auto u_store = compressed(anndatapp::Layout::csr, n, g, unspliced.data());
    auto w_store = compressed(anndatapp::Layout::csc, n, n, weights.data());
    auto sa = s_store.array(), ua = u_store.array(), wa = w_store.array();
    InputOwner input_owner(sa.view(), ua.view(), wa.view());
    auto& input = input_owner.value;
    std::array<std::size_t, 3> selection{2, 0, 2};
    input.genes = selection.data(); input.gene_count = selection.size();
    Prepared owner(input);
    auto& prepared = owner.value;

    expect(prepared.genes == 3 && prepared.columns == 3 &&
           prepared.input.genes[0] == 2 && prepared.input.genes[1] == 0 &&
           prepared.input.genes[2] == 2,
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
    InputOwner input_owner(sa.view(), ua.view(), wa.view());
    auto& input = input_owner.value;
    expect(velocitypp::validate(input).code == velocitypp::Code::nonfinite,
           "explicit value validation rejects a nonfinite dense expression value");
}

void workspace_respects_the_requested_bound() {
    auto s = anndatapp::Array::borrow_dense(spliced.data(), {n, g}, {static_cast<std::ptrdiff_t>(g), 1});
    auto u = anndatapp::Array::borrow_dense(unspliced.data(), {n, g}, {static_cast<std::ptrdiff_t>(g), 1});
    auto w_store = compressed(anndatapp::Layout::csr, n, n, weights.data());
    auto w = w_store.array();
    InputOwner input_owner(s.view(), u.view(), w.view());
    Prepared owner(input_owner.value);
    auto& prepared = owner.value;

    velocitypp::RunOptions options;
    options.workspace_bytes = 1u << 20;
    options.block_genes = 64;
    velocitypp::WorkspaceSize size;
    expect(bool(velocitypp::workspace_size<double>(prepared, options, false, size)) &&
           size.bytes <= options.workspace_bytes && size.block <= options.block_genes &&
           size.block <= prepared.genes,
           "workspace reports bounded scratch and a block no larger than the request or gene count");
    options.workspace_bytes = 1;
    expect(!velocitypp::workspace_size<double>(prepared, options, false, size),
           "workspace rejects a cap too small for one gene and mandatory scratch");
}

struct SinkCapture {
    std::size_t rows = 0, genes = 0, calls = 0;
    std::vector<double> velocity;
    std::vector<std::size_t> blocks;
};

void capture_velocity(void* context, const velocitypp::MomentBlock& moments,
                      const velocitypp::FitBlock& fit) {
    auto& capture = *static_cast<SinkCapture*>(context);
    ++capture.calls;
    capture.blocks.push_back(moments.count);
    velocitypp::reference::residuals(moments, fit,
        capture.velocity.data() + moments.first, std::ptrdiff_t(capture.genes));
}

void reference_fit_sink_and_caller_output() {
    auto sa = anndatapp::Array::borrow_dense(spliced.data(), {n, g}, {static_cast<std::ptrdiff_t>(g), 1});
    auto ua = anndatapp::Array::borrow_dense(unspliced.data(), {n, g}, {static_cast<std::ptrdiff_t>(g), 1});
    auto w_store = compressed(anndatapp::Layout::csr, n, n, weights.data());
    auto wa = w_store.array();
    InputOwner input_owner(sa.view(), ua.view(), wa.view());
    Prepared owner(input_owner.value);
    auto& prepared = owner.value;

    velocitypp::RunOptions run;
    run.block_genes = 2;
    Storage<double> storage(prepared, run);
    SinkCapture capture{n, g, 0, std::vector<double>(n * g, 99), {}};
    velocitypp::BlockSink sink{&capture, capture_velocity};
    Result<double> result(g);
    expect(bool(velocitypp::reference::steady_state(prepared, storage.workspace, result.value,
                                                    exact_fit_options(), sink)), "reference fit succeeds");
    bool fits_ok = capture.calls == 2;
    for (std::size_t j = 0; j < g; ++j)
        fits_ok = fits_ok && close(result.gamma[j], 2) && close(result.offset[j], 0) &&
            result.r2[j] > .999999 && result.genes[j] &&
            result.status[j] == static_cast<std::uint8_t>(velocitypp::FitStatus::ok);
    expect(fits_ok, "callable reference fit recovers exact proportional steady-state genes");
    bool sink_ok = capture.calls == 2 && capture.blocks.size() == 2 &&
        capture.blocks[0] == 2 && capture.blocks[1] == 1;
    for (double value : capture.velocity) sink_ok = sink_ok && close(value, 0);
    expect(sink_ok, "reference sink consumes reusable velocity blocks synchronously");

    std::array<double, n*g> caller_velocity{};
    Result<double> into(g);
    bool into_ok = bool(velocitypp::reference::steady_state_into(
        prepared, storage.workspace, into.value, caller_velocity.data(), caller_velocity.size(), g, exact_fit_options()));
    for (double value : caller_velocity) into_ok = into_ok && close(value, 0);
    for (std::size_t j = 0; j < g; ++j) into_ok = into_ok && close(into.gamma[j], 2);
    expect(into_ok, "caller-owned output route writes expected residual velocities");
}

void caller_output_capacity_precision_and_negative_pitch() {
    constexpr std::size_t rows = 3, genes = 2, pitch = 4;
    std::array<double, rows * genes> s{1, 2, 2, 3, 4, 5};
    std::array<double, rows * genes> u{2, 5, 5, 7, 7, 11};
    std::array<double, rows * rows> w{1, 0, 0, 0, 1, 0, 0, 0, 1};
    auto sa = anndatapp::Array::borrow_dense(s.data(), {rows, genes}, {genes, 1});
    auto ua = anndatapp::Array::borrow_dense(u.data(), {rows, genes}, {genes, 1});
    auto wa = anndatapp::Array::borrow_dense(w.data(), {rows, rows}, {rows, 1});
    InputOwner input(sa.view(), ua.view(), wa.view());
    Prepared owner(input.value);
    Storage<double> storage(owner.value);
    Result<double> result(genes);
    std::array<float, (rows - 1) * pitch + genes> out;
    out.fill(123.0f);
    auto status = velocitypp::reference::steady_state_into(
        owner.value, storage.workspace, result.value, out.data() + (rows - 1) * pitch,
        out.size() - 1, -std::ptrdiff_t(pitch), exact_fit_options());
    bool untouched = status.code == velocitypp::Code::capacity;
    for (float x : out) untouched = untouched && x == 123.0f;
    expect(untouched, "insufficient output capacity reports capacity without writing output");

    status = velocitypp::reference::steady_state_into(
        owner.value, storage.workspace, result.value, out.data() + (rows - 1) * pitch,
        out.size(), -std::ptrdiff_t(pitch), exact_fit_options());
    bool correct = bool(status);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < genes; ++j) {
            const double expected = u[i*genes+j] - result.gamma[j]*s[i*genes+j] - result.offset[j];
            correct = correct && close(out[(rows-1-i)*pitch+j], float(expected), 1e-6);
        }
    for (std::size_t i = 0; i + 1 < rows; ++i)
        for (std::size_t j = genes; j < pitch; ++j) correct = correct && out[i*pitch+j] == 123.0f;
    expect(correct, "double compute writes float residuals through a negative row pitch without changing gaps");
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
    std::array<std::uint8_t, rows> fit_values{true, false, true, true};
    std::array<std::uint8_t, genes> gene_values{true, false, true, true, true, true, true};
    InputOwner input_owner(sa.view(), ua.view(), wa.view());
    auto& input = input_owner.value;
    std::array<std::size_t, genes> selection{4, 5, 0, 2, 1, 3, 6};
    input.genes = selection.data(); input.gene_count = selection.size();
    input.fit_mask = fit_values.data(); input.gene_mask = gene_values.data();
    Prepared owner(input);
    auto& prepared = owner.value;
    Storage<double> storage(prepared);

    auto options = exact_fit_options();
    Result<double> kept(genes);
    expect(bool(velocitypp::reference::steady_state(prepared, storage.workspace, kept.value, options)), "masked fit succeeds");
    const auto no_signal = std::uint8_t(velocitypp::FitStatus::no_signal);
    const auto ok = std::uint8_t(velocitypp::FitStatus::ok);
    bool masks_ok = !kept.value.low_signal &&
        close(kept.gamma[0], 2) && kept.r2[0] > .7 && kept.r2[0] < .99 &&
        kept.genes[0] && kept.status[0] == ok &&
        kept.status[1] == no_signal && !kept.genes[1] &&
        kept.genes[2] && close(kept.gamma[2], 2) &&
        !kept.genes[4] && kept.genes[5] && kept.genes[6];
    expect(masks_ok,
           "fit mask trains on its rows, R2 uses all rows, and source-gene mask follows a gather axis");

    options.min_r2 = .9999;
    Result<double> filtered(genes);
    expect(bool(velocitypp::reference::steady_state(prepared, storage.workspace, filtered.value, options)), "filtered fit succeeds");
    std::size_t filtered_count = 0;
    for (std::size_t j = 0; j < genes; ++j) filtered_count += filtered.genes[j];
    expect(!filtered.value.low_signal && filtered_count == 1 && filtered.genes[2],
           "default keep_filters preserves the per-gene threshold result when fewer than two genes pass");

    options.low_signal = velocitypp::LowSignal::scvelo_percentile;
    Result<double> percentile(genes);
    expect(bool(velocitypp::reference::steady_state(prepared, storage.workspace, percentile.value, options)), "percentile fit succeeds");
    std::size_t selected_count = 0;
    for (std::size_t j = 0; j < genes; ++j) selected_count += percentile.genes[j];
    expect(percentile.value.low_signal && selected_count == 1 && percentile.genes[2] &&
           percentile.status[1] == no_signal && !percentile.genes[1] &&
           !percentile.genes[4] && !percentile.genes[6],
           "scvelo percentile selects above the valid-fit 80th percentile and excludes invalid or masked genes");
}

void degenerate_gene_reports_no_signal() {
    constexpr std::array<double, 4> x{0, 0, 0, 0};
    constexpr std::array<double, 4> y{1, 2, 3, 4};
    std::array<double, 4> score_scratch{};
    const auto fit = velocitypp::reference::fit_gene(x.data(), y.data(), x.size(), 1,
        exact_fit_options(), score_scratch.data());
    expect(fit.status == velocitypp::FitStatus::no_signal && !fit.selected,
           "zero-expression fit has no_signal status and is not selected");
}

struct Neighborhood {
    Acquired input;
    velocitypp::reference::NeighborhoodSize size;
    std::vector<std::size_t> indices, columns, offsets;
    std::vector<velocitypp::Entry> scratch;
    std::vector<float> values;
    std::size_t used = 0;
    velocitypp::Status status;
    explicit Neighborhood(anndatapp::ArrayView view) : input(view) {
        status = velocitypp::reference::neighborhood_size(input.value, size);
        if (!status) return;
        indices.resize(size.indices); scratch.resize(size.scratch_entries);
        values.resize(size.entries); columns.resize(size.entries); offsets.resize(view.extent(0) + 1);
        status = velocitypp::reference::prepare_neighborhood(input.value,
            {indices.data(), indices.size(), scratch.data(), scratch.size(), values.data(),
             columns.data(), offsets.data(), values.size(), offsets.size()}, used);
    }
};

template<std::size_t N, std::size_t I, std::size_t V>
void expect_csr(const Neighborhood& actual, const std::array<std::size_t, N>& offsets,
                const std::array<std::uint8_t, I>& indices,
                const std::array<float, V>& values, const char* message) {
    bool ok = bool(actual.status) && actual.used == V && actual.offsets.size() == N;
    for (std::size_t i = 0; ok && i < N; ++i) ok = actual.offsets[i] == offsets[i];
    for (std::size_t i = 0; ok && i < V; ++i)
        ok = actual.columns[i] == indices[i] && std::abs(double(actual.values[i]) - values[i]) < 1e-7;
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
    Neighborhood csr_prepared(csr.view());
    expect_csr(csr_prepared, output_offsets, output_indices, output_values,
        "noncanonical CSR coalesces duplicates, inserts one diagonal, and normalizes each row");

    constexpr std::array<float, 12> csc_values{-1, -2, 1, 3, -1, 1, 2, 1, 0, 0, 0, 2};
    constexpr std::array<std::int32_t, 12> csc_indices{1, 0, 0, 2, 2, 0, 0, 0, 2, 0, 1, 2};
    constexpr std::array<std::int64_t, 5> csc_offsets{0, 3, 7, 9, 12};
    auto csc = anndatapp::Array::borrow_sparse(anndatapp::Layout::csc, 4, 4,
        csc_values.size(), csc_values.data(), csc_indices.data(), csc_offsets.data());
    Neighborhood csc_prepared(csc.view());
    expect_csr(csc_prepared, output_offsets, output_indices, output_values,
        "noncanonical CSC with unsorted rows and duplicates produces the same normalized graph");

    constexpr std::array<float, 16> dense_values{
        -1, 3, 1, 0,
        -2, 0, -2, 0,
        0, 2, -3, 2,
        -1, 0, 0, 0
    };
    auto dense = anndatapp::Array::borrow_dense(dense_values.data(), {4, 4}, {4, 1});
    Neighborhood dense_prepared(dense.view());
    expect_csr(dense_prepared, output_offsets, output_indices, output_values,
        "dense graph retains positive off-diagonals and one diagonal before row normalization");
}

void neighborhood_preparation_handles_empty_and_nonfinite_graphs() {
    auto empty = anndatapp::Array::csr<float, std::uint8_t, std::uint8_t>(0, 0, 0);
    Neighborhood prepared_empty(empty.view());
    expect(bool(prepared_empty.status) && prepared_empty.used == 0 &&
           prepared_empty.offsets.size() == 1 && prepared_empty.offsets[0] == 0,
           "empty sparse matrix prepares to an empty CSR graph");

    constexpr std::array<float, 4> dense_values{1, 0, 0, 1};
    auto dense_bad_values = dense_values;
    dense_bad_values[1] = std::numeric_limits<float>::infinity();
    auto dense_bad = anndatapp::Array::borrow_dense(dense_bad_values.data(), {2, 2}, {2, 1});
    expect(Neighborhood(dense_bad.view()).status.code == velocitypp::Code::nonfinite,
        "dense neighborhood preparation rejects nonfinite connectivity values");

    constexpr std::array<float, 1> sparse_values{std::numeric_limits<float>::quiet_NaN()};
    constexpr std::array<std::int32_t, 1> sparse_indices{1};
    constexpr std::array<std::int64_t, 3> sparse_offsets{0, 1, 1};
    auto sparse_bad = anndatapp::Array::borrow_sparse(anndatapp::Layout::csr, 2, 2, 1,
        sparse_values.data(), sparse_indices.data(), sparse_offsets.data());
    expect(Neighborhood(sparse_bad.view()).status.code == velocitypp::Code::nonfinite,
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
    caller_output_capacity_precision_and_negative_pitch();
    masks_and_low_signal_selection();
    degenerate_gene_reports_no_signal();
    neighborhood_preparation_normalizes_sparse_and_dense_graphs();
    neighborhood_preparation_handles_empty_and_nonfinite_graphs();
    if (!failures) std::cout << "reference tests passed\n";
    return failures ? 1 : 0;
}
