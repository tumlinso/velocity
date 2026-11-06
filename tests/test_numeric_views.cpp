#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

import velocitypp;

using anndatapp::Array;
using anndatapp::ArrayView;
using anndatapp::Axis;
using anndatapp::DType;
using anndatapp::Layout;
using velocitypp::AcquiredMatrix;
using velocitypp::DenseInput;
using velocitypp::Entry;
using velocitypp::PreparedInput;
using velocitypp::SparseInput;

namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<class F> void rejects(F f) {
    bool failed = false;
    try { f(); } catch (const std::exception&) { failed = true; }
    check(failed, "expected invalid array access to throw");
}

// This view outlives its move-only Array owner and keeps the native array
// state and its allocation available.
ArrayView dense_view_after_owner_destruction() {
    auto array = Array::empty<float>({2, 3});
    auto values = array.edit().dense<float>();
    for (std::size_t i = 0; i < 6; ++i) values.data()[i] = float(i);
    return array.view();
}

void test_dense_view_and_edit() {
    auto source = dense_view_after_owner_destruction();
    check(source.dtype() == DType::float32 && source.rank() == 2,
          "retained ArrayView preserves dtype and rank after the Array is destroyed");

    auto reversed = source.select({Axis::stride(1, 2, 1, true), Axis::all()});
    auto rows = reversed.dense<float>();
    check(rows.stride(0) == -3 && rows.data()[0] == 3 && rows.data()[2] == 5 &&
          rows.data()[rows.stride(0)] == 0,
          "affine ArrayView selection preserves a negative row stride and logical order");

    auto sampled = reversed.select({Axis::all(), Axis::stride(0, 2, 2)}).dense<float>();
    check(sampled.stride(1) == 2 && sampled.data()[0] == 3 &&
          sampled.data()[sampled.stride(1)] == 5 && sampled.data()[sampled.stride(0)] == 0 &&
          sampled.data()[sampled.stride(0) + sampled.stride(1)] == 2,
          "composed affine selections expose the expected element strides");

    auto gathered = source.select({Axis::all(), Axis::gather({2, 0})}).copy();
    auto gathered_values = gathered.view().dense<float>();
    check(gathered.extent(0) == 2 && gathered.extent(1) == 2 &&
          gathered_values.data()[0] == 2 && gathered_values.data()[1] == 0 &&
          gathered_values.data()[2] == 5 && gathered_values.data()[3] == 3,
          "gather selection materializes in the requested order");

    rejects([&] { (void)source.dense<double>(); });

    auto edited = Array::zeros<float>({2, 3});
    auto edit = edited.edit();
    edit.dense<float>().data()[4] = 7;
    auto observed = edit.view();
    check(observed.dtype() == DType::float32 && observed.dense<float>().data()[4] == 7,
          "ArrayEdit writes are visible through its native ArrayView");
}

struct GraphBuffers {
    float values[4] = {0, 2, 0, 0};
    std::int32_t indices[1] = {1};
    std::int64_t offsets[3] = {0, 1, 1};
    bool released = false;
};
struct GraphOwner {
    GraphBuffers* graph;
    ~GraphOwner() {
        for (auto& x : graph->values) x = 0;
        for (auto& x : graph->indices) x = 0;
        for (auto& x : graph->offsets) x = 0;
        graph->released = true;
    }
};

ArrayView retained_graph(GraphBuffers& graph, Layout layout) {
    auto owner = std::make_shared<GraphOwner>(&graph);
    if (layout == Layout::dense) {
        auto array = Array::share_dense(graph.values, {2, 2}, {2, 1}, owner);
        owner.reset();
        return array.view();
    }
    if (layout == Layout::csc) {
        graph.indices[0] = 0;
        graph.offsets[1] = 0;
    }
    auto array = Array::share_sparse(layout, 2, 2, 1, graph.values + 1,
                                     graph.indices, graph.offsets, owner);
    owner.reset();
    return array.view();
}

struct AcquiredView {
    ArrayView view;
    DenseInput dense;
    std::optional<SparseInput> sparse;
    AcquiredMatrix input;

    explicit AcquiredView(ArrayView retained) : view(std::move(retained)) {
        input.dtype = view.dtype();
        input.layout = view.layout();
        if (input.layout == Layout::dense) {
            dense = view.bytes();
            input.dense = &dense;
        } else {
            input.index_type = view.index_dtype();
            input.offset_type = view.offset_dtype();
            sparse.emplace(view.sparse_bytes());
            input.sparse = &*sparse;
        }
    }

    AcquiredView(const AcquiredView&) = delete;
    AcquiredView& operator=(const AcquiredView&) = delete;
};

std::uint64_t unsigned_at(const std::byte* p, std::size_t width, std::size_t i) {
    p += width * i;
    switch (width) {
    case 1: { std::uint8_t x; std::memcpy(&x, p, 1); return x; }
    case 2: { std::uint16_t x; std::memcpy(&x, p, 2); return x; }
    case 4: { std::uint32_t x; std::memcpy(&x, p, 4); return x; }
    case 8: { std::uint64_t x; std::memcpy(&x, p, 8); return x; }
    default: throw std::runtime_error("unexpected sparse integer width");
    }
}

struct NeighborhoodStorage {
    std::vector<std::size_t> indices;
    std::vector<Entry> scratch;
    std::vector<float> values;
    std::vector<std::size_t> columns, offsets;
    std::size_t used = 0;
    anndatapp::Array array;
};

NeighborhoodStorage prepare_neighborhood(const AcquiredMatrix& input) {
    velocitypp::reference::NeighborhoodSize size{};
    check(bool(velocitypp::reference::neighborhood_size(input, size)),
          "neighborhood sizing accepts the retained graph view");

    NeighborhoodStorage result;
    result.indices.resize(size.indices);
    result.scratch.resize(size.scratch_entries);
    result.values.resize(size.entries);
    result.columns.resize(size.entries);
    result.offsets.resize(input.layout == Layout::dense
        ? input.dense->extent(0) + 1 : input.sparse->rows() + 1);
    velocitypp::reference::NeighborhoodBuffers buffers{
        result.indices.data(), result.indices.size(),
        result.scratch.data(), result.scratch.size(),
        result.values.data(), result.columns.data(), result.offsets.data(), result.values.size(), result.offsets.size()
    };
    check(bool(velocitypp::reference::prepare_neighborhood(input, buffers, result.used)),
          "caller-owned buffers prepare the neighborhood graph");
    result.array = anndatapp::Array::borrow_sparse(Layout::csr,
        result.offsets.size() - 1, result.offsets.size() - 1, result.used,
        result.values.data(), result.columns.data(), result.offsets.data());
    return result;
}

void check_neighborhood(const NeighborhoodStorage& result) {
    auto view = result.array.view();
    check(view.layout() == Layout::csr && view.dtype() == DType::float32,
          "neighborhood preparation writes a caller-owned FP32 CSR graph");
    auto bytes = view.sparse_bytes();
    check(bytes.rows() == 2, "normalized sparse graph has two rows");
    check(bytes.columns() == 2, "normalized sparse graph has two columns");
    check(bytes.nnz() == 3, "normalized sparse graph has three entries");
    const std::uint64_t expected_offsets[] = {0, 2, 3};
    const std::uint64_t expected_indices[] = {0, 1, 1};
    const float expected_values[] = {.5f, .5f, 1.f};
    for (std::size_t i = 0; i < 3; ++i) {
        check(unsigned_at(bytes.offsets(), bytes.offset_size(), i) == expected_offsets[i] &&
              unsigned_at(bytes.indices(), bytes.index_size(), i) == expected_indices[i],
              "normalized sparse graph retains the expected CSR structure");
        float value;
        std::memcpy(&value, bytes.values() + i * sizeof(float), sizeof value);
        check(std::abs(value - expected_values[i]) < 1e-7f,
              "normalized sparse graph retains the expected row weights");
    }
}

NeighborhoodStorage test_temporary_graph_owner(Layout layout) {
    GraphBuffers graph;
    auto result = [&] {
        AcquiredView input(retained_graph(graph, layout));
        auto prepared = prepare_neighborhood(input.input);
        check(!graph.released,
              "caller-retained dense or sparse graph storage stays live during preparation");
        return prepared;
    }();
    check(graph.released,
          "retained graph storage is released when its caller-owned view leaves scope");
    check_neighborhood(result);
    return result;
}

struct PreparationStorage {
    std::vector<std::size_t> indices;
    std::vector<Entry> entries;
    PreparedInput prepared;

    void bind(const velocitypp::Input& input) {
        velocitypp::PreparationSize size{};
        check(bool(velocitypp::preparation_size(input, size)),
              "caller-owned preparation sizing accepts the matrix inputs");
        indices.resize(size.indices);
        entries.resize(size.entries);
        velocitypp::PreparationBuffers buffers{
            indices.data(), indices.size(), entries.data(), entries.size()
        };
        check(bool(velocitypp::prepare(input, buffers, prepared)),
              "caller-owned index and entry buffers prepare mixed sparse inputs");
    }
};

void test_sparse_view_and_mixed_precision_moments(const NeighborhoodStorage& graph) {
    float s_values[] = {1, 3};
    std::int16_t s_indices[] = {0, 0};
    std::int32_t s_offsets[] = {0, 1, 2};
    auto spliced = Array::borrow_sparse(Layout::csr, 2, 1, 2,
                                        s_values, s_indices, s_offsets);
    auto spliced_view = spliced.view();
    auto spliced_csr = spliced_view.csr<float, std::int16_t, std::int32_t>();
    check(spliced_view.dtype() == DType::float32 && spliced_csr.rows() == 2 &&
          spliced_csr.columns() == 1 && spliced_csr.nnz() == 2 &&
          spliced_csr.values()[1] == 3 && spliced_csr.offsets()[2] == 2,
          "native CSR views expose typed values and structural buffers");

    double u_values[] = {10, 30};
    std::uint32_t u_indices[] = {0, 1};
    std::uint64_t u_offsets[] = {0, 2};
    auto unspliced = Array::borrow_sparse(Layout::csc, 2, 1, 2,
                                          u_values, u_indices, u_offsets);
    auto unspliced_view = unspliced.view();
    auto unspliced_csc = unspliced_view.csc<double, std::uint32_t, std::uint64_t>();
    check(unspliced_view.dtype() == DType::float64 && unspliced_csc.rows() == 2 &&
          unspliced_csc.columns() == 1 && unspliced_csc.nnz() == 2 &&
          unspliced_csc.values()[0] == 10 && unspliced_csc.indices()[1] == 1,
          "native CSC views retain their layout and independent integer widths");

    AcquiredView spliced_input(spliced_view);
    AcquiredView unspliced_input(unspliced_view);
    AcquiredView graph_input(graph.array.view());
    velocitypp::Input input{};
    input.spliced = spliced_input.input;
    input.unspliced = unspliced_input.input;
    input.weights = graph_input.input;
    PreparationStorage storage;
    storage.bind(input);

    float ms32[2]{}, mu32[2]{};
    double ms64[2]{}, mu64[2]{};
    velocitypp::reference::moments(storage.prepared, 0, 1, ms32, mu32);
    velocitypp::reference::moments(storage.prepared, 0, 1, ms64, mu64);
    check(std::abs(ms32[0] - 2.f) < 1e-6f && std::abs(ms32[1] - 3.f) < 1e-6f &&
          std::abs(mu32[0] - 20.f) < 1e-6f && std::abs(mu32[1] - 30.f) < 1e-6f,
          "FP32 moment operation consumes mixed FP32/FP64 sparse sources");
    check(std::abs(ms64[0] - 2.) < 1e-12 && std::abs(ms64[1] - 3.) < 1e-12 &&
          std::abs(mu64[0] - 20.) < 1e-12 && std::abs(mu64[1] - 30.) < 1e-12,
          "FP64 moment operation consumes mixed FP32/FP64 sparse sources");
}
}

int main() {
    test_dense_view_and_edit();
    auto dense = test_temporary_graph_owner(Layout::dense);
    auto csr = test_temporary_graph_owner(Layout::csr);
    auto csc = test_temporary_graph_owner(Layout::csc);
    check(dense.array.layout() == Layout::csr && csc.array.layout() == Layout::csr,
          "dense, CSR, and CSC input graphs normalize to CSR");
    test_sparse_view_and_mixed_precision_moments(csr);
}
