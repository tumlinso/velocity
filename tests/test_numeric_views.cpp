#include <cassert>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

import velocity;

using namespace velocity;
using anndatapp::Axis;
using anndatapp::DType;
using anndatapp::Layout;

template<class F> void rejects(F f) {
    bool failed = false;
    try { f(); } catch (const std::exception&) { failed = true; }
    assert(failed);
}

// The stack buffers remain addressable after the retained owner's destruction.
// Poisoning them makes an early owner release observable even without ASan.
struct GraphBuffers {
    std::uint16_t values[4] = {0, 0x4000, 0, 0};
    std::int32_t indices[1] = {1};
    std::int32_t offsets[3] = {0, 1, 1};
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
NumericView retained_graph(GraphBuffers& graph, Layout layout) {
    auto owner = std::make_shared<GraphOwner>(&graph);
    if (layout == Layout::dense)
        return NumericView::retain_dense_f16(graph.values, {2,2}, {2,1}, std::move(owner));
    graph.values[0] = 0x4000;
    if (layout == Layout::csc) {
        graph.indices[0] = 0;
        graph.offsets[1] = 0;
    }
    SparseF16 buffers{graph.values,1,graph.indices,1,DType::int32,graph.offsets,3,DType::int32};
    return NumericView::retain_sparse_f16(layout,2,2,1,buffers,std::move(owner));
}
std::uint64_t unsigned_at(const std::byte* p, std::size_t width, std::size_t i) {
    p += width*i;
    switch (width) {
    case 1: { std::uint8_t x; std::memcpy(&x,p,1); return x; }
    case 2: { std::uint16_t x; std::memcpy(&x,p,2); return x; }
    case 4: { std::uint32_t x; std::memcpy(&x,p,4); return x; }
    case 8: { std::uint64_t x; std::memcpy(&x,p,8); return x; }
    default: throw std::runtime_error("unexpected sparse integer width");
    }
}
void check(bool ok) { if (!ok) throw std::runtime_error("temporary graph owner regression"); }
void test_temporary_graph_owner(Layout layout) {
    GraphBuffers graph;
    auto result = reference::prepare_neighborhood(retained_graph(graph, layout));
    check(graph.released);
    auto view = result.view();
    check(view.layout() == Layout::csr && view.dtype() == DType::float32);
    auto bytes = view.sparse_bytes();
    check(bytes.rows() == 2 && bytes.columns() == 2 && bytes.nnz() == 3);
    const std::uint64_t expected_offsets[] = {0,2,3}, expected_indices[] = {0,1,1};
    const float expected_values[] = {.5f,.5f,1.f};
    for (std::size_t i = 0; i < 3; ++i) {
        check(unsigned_at(bytes.offsets(),bytes.offset_size(),i) == expected_offsets[i]);
        check(unsigned_at(bytes.indices(),bytes.index_size(),i) == expected_indices[i]);
        float value;
        std::memcpy(&value,bytes.values()+i*sizeof(float),sizeof value);
        check(std::abs(value - expected_values[i]) < 1e-7f);
    }
}

int main() {
    test_temporary_graph_owner(Layout::dense);
    test_temporary_graph_owner(Layout::csr);
    test_temporary_graph_owner(Layout::csc);
    auto owner = std::make_shared<std::vector<std::uint16_t>>(std::initializer_list<std::uint16_t>{0,1,2,3,4,5});
    std::weak_ptr<std::vector<std::uint16_t>> weak = owner;
    auto v = NumericView::retain_dense_f16(owner->data()+3, {2,3}, {-3,1}, owner);
    owner.reset();
    assert(!weak.expired() && v.dtype() == StorageType::float16 && v.rank() == 2);
    auto b = v.bytes();
    assert(b.stride_bytes(0) == -6 && b.stride_bytes(1) == 2 && !b.contiguous());
    auto col = v.select({Axis::all(), Axis::stride(2,1)}).dense<std::uint16_t>();
    assert(col.data()[0] == 5 && col.data()[col.stride(0)] == 2);
    auto rev = v.select({Axis::stride(1,2,1,true), Axis::gather({2,1,0})}).dense<std::uint16_t>();
    assert(rev.data()[0] == 2 && rev.stride(0) == 3 && rev.stride(1) == -1);
    auto gather = v.select({Axis::all(), Axis::gather({0,2,1})});
    rejects([&]{ (void)gather.bytes(); });
    rejects([&]{ (void)v.select({Axis::all(), Axis::stride(2,2)}); });
    rejects([&]{ (void)v.dense<float>(); });
    rejects([&]{ (void)NumericView::borrow_dense_f16(nullptr,{1},{1}); });
    rejects([&]{ (void)NumericView::borrow_dense_f16(nullptr,{1,2},{1}); });
    rejects([&]{ (void)NumericView::borrow_dense_f16(nullptr,{0},{std::numeric_limits<std::ptrdiff_t>::max()}); });
    auto empty = NumericView::borrow_dense_f16(nullptr,{0,3},{3,1});
    assert(empty.bytes().data() == nullptr && empty.extent(0) == 0);
    std::uint16_t output[6]{};
    auto edit = NumericEdit::borrow_dense_f16(output,{2,3},{3,1});
    edit.dense<std::uint16_t>().data()[4] = 123;
    assert(edit.view().dense<std::uint16_t>().data()[4] == 123);
    auto array = anndatapp::Array::zeros<float>({2,3});
    NumericEdit e = array.edit();
    e.dense<float>().data()[1] = 7;
    NumericView x = array.view();
    assert(x.dtype() == StorageType::float32 && x.dense<float>().data()[1] == 7);
    auto ints = anndatapp::Array::zeros<std::uint16_t>({2});
    rejects([&]{ (void)NumericView(ints.view()).dtype(); });

    std::uint16_t values[] = {1,2,3};
    std::int16_t indices[] = {2,0,1};
    std::int32_t offsets[] = {0,2,3};
    SparseF16 s{values,3,indices,3,DType::int16,offsets,3,DType::int32};
    auto csr = NumericView::borrow_sparse_f16(Layout::csr,2,3,3,s);
    csr.validate(); auto sparse = csr.sparse_bytes();
    assert(sparse.nnz() == 3 && sparse.index_size() == 2 && sparse.offset_size() == 4 && sparse.rows() == 2);
    rejects([&]{ (void)csr.select({Axis::stride(0,1),Axis::all()}).sparse_bytes(); });
    auto bad = s; bad.indices_count = 2;
    rejects([&]{ (void)NumericView::borrow_sparse_f16(Layout::csr,2,3,3,bad); });
    indices[0] = -1; rejects([&]{ csr.validate(); }); indices[0] = 3; rejects([&]{ csr.validate(); }); indices[0] = 2;
    offsets[1] = 4; rejects([&]{ csr.validate(); }); offsets[1] = 2;
    offsets[2] = 2; rejects([&]{ csr.validate(); }); offsets[2] = 3;
    auto held = std::make_shared<int>(1);
    std::weak_ptr<int> sparse_weak = held;
    auto retained = NumericView::retain_sparse_f16(Layout::csr,2,3,3,s,held).sparse_bytes();
    held.reset();
    assert(!sparse_weak.expired() && retained.nnz() == 3);
    rejects([&]{ (void)NumericView::retain_sparse_f16(Layout::csr,2,3,3,s,{}); });
    auto csc = NumericView::borrow_sparse_f16(Layout::csc,3,2,3,s);
    assert(csc.sparse_bytes().columns() == 2);
}
