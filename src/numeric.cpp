module;
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

module velocity;
import :numeric;

namespace velocity {
namespace detail {
struct HalfStorage {
    const std::uint16_t* values = nullptr;
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides;
    anndatapp::Layout layout = anndatapp::Layout::dense;
    SparseF16 sparse;
    std::size_t nnz = 0;
    std::shared_ptr<const void> owner;
};
}
namespace {
using anndatapp::DType;
using anndatapp::Layout;

std::ptrdiff_t mul(std::ptrdiff_t a, std::ptrdiff_t b) {
    std::ptrdiff_t c;
    if (__builtin_mul_overflow(a, b, &c)) throw std::length_error("numeric stride overflow");
    return c;
}
std::ptrdiff_t add(std::ptrdiff_t a, std::ptrdiff_t b) {
    std::ptrdiff_t c;
    if (__builtin_add_overflow(a, b, &c)) throw std::length_error("numeric offset overflow");
    return c;
}
std::ptrdiff_t as_signed(std::size_t n) {
    if (n > std::size_t(std::numeric_limits<std::ptrdiff_t>::max())) throw std::length_error("numeric dimension overflow");
    return std::ptrdiff_t(n);
}
std::size_t product(std::size_t n, std::size_t width) {
    if (width && n > std::numeric_limits<std::size_t>::max()/width) throw std::length_error("numeric buffer size overflow");
    return n*width;
}
void address(const void* p, std::size_t n) {
    if (!n) return;
    if (!p) throw std::invalid_argument("null numeric buffer");
    auto base = reinterpret_cast<std::uintptr_t>(p);
    if (n > std::numeric_limits<std::uintptr_t>::max() - base) throw std::length_error("numeric address overflow");
}
void dense_validate(const detail::HalfStorage& h) {
    if (h.shape.empty() || h.shape.size() > 2 || h.shape.size() != h.strides.size())
        throw std::invalid_argument("binary16 dense storage requires rank one or two and matching strides");
    bool empty = false;
    for (auto n : h.shape) { (void)as_signed(n); empty |= n == 0; }
    std::ptrdiff_t lo = 0, hi = 0;
    for (std::size_t a = 0; a < h.shape.size(); ++a) {
        auto stride = mul(h.strides[a], 2);
        if (empty || !stride || h.shape[a] <= 1) continue;
        auto d = mul(as_signed(h.shape[a] - 1), stride);
        if (d < 0) lo = add(lo, d); else hi = add(hi, d);
    }
    if (empty) return;
    if (!h.values) throw std::invalid_argument("null binary16 dense buffer");
    if (reinterpret_cast<std::uintptr_t>(h.values) % alignof(std::uint16_t)) throw std::invalid_argument("unaligned binary16 dense buffer");
    if (lo == std::numeric_limits<std::ptrdiff_t>::min()) throw std::length_error("numeric span overflow");
    (void)add(add(hi, -lo), 2);
    auto base = reinterpret_cast<std::uintptr_t>(h.values);
    auto before = lo < 0 ? std::uintptr_t(-(lo + 1)) + 1 : 0;
    if (base < before || std::uintptr_t(hi) > std::numeric_limits<std::uintptr_t>::max() - base ||
        2 > std::numeric_limits<std::uintptr_t>::max() - (base + std::uintptr_t(hi)))
        throw std::length_error("binary16 dense address overflow");
}
std::size_t integer_width(DType t) {
    switch (t) {
    case DType::int8: case DType::uint8: return 1;
    case DType::int16: case DType::uint16: return 2;
    case DType::int32: case DType::uint32: return 4;
    case DType::int64: case DType::uint64: return 8;
    default: throw std::invalid_argument("sparse index and offset types must be integers");
    }
}
template<class T> std::size_t integer_value(const std::byte* p) {
    T x;
    std::memcpy(&x, p, sizeof x);
    if constexpr (std::is_signed_v<T>) if (x < 0) throw std::invalid_argument("negative sparse index or offset");
    auto n = std::uint64_t(x);
    if (n > std::numeric_limits<std::size_t>::max()) throw std::length_error("sparse index exceeds size_t");
    return std::size_t(n);
}
std::size_t integer(const void* data, DType t, std::size_t i) {
    auto p = static_cast<const std::byte*>(data) + product(i, integer_width(t));
    switch (t) {
    case DType::int8: return integer_value<std::int8_t>(p);
    case DType::uint8: return integer_value<std::uint8_t>(p);
    case DType::int16: return integer_value<std::int16_t>(p);
    case DType::uint16: return integer_value<std::uint16_t>(p);
    case DType::int32: return integer_value<std::int32_t>(p);
    case DType::uint32: return integer_value<std::uint32_t>(p);
    case DType::int64: return integer_value<std::int64_t>(p);
    case DType::uint64: return integer_value<std::uint64_t>(p);
    default: throw std::invalid_argument("invalid sparse integer type");
    }
}
void sparse_validate(const detail::HalfStorage& h) {
    if (h.layout != Layout::csr && h.layout != Layout::csc) throw std::invalid_argument("sparse binary16 layout must be CSR or CSC");
    auto major = h.layout == Layout::csr ? h.shape[0] : h.shape[1];
    auto minor = h.layout == Layout::csr ? h.shape[1] : h.shape[0];
    if (major == std::numeric_limits<std::size_t>::max()) throw std::length_error("sparse dimension overflow");
    auto iw = integer_width(h.sparse.index_type), ow = integer_width(h.sparse.offset_type);
    if (h.sparse.values_count < h.nnz || h.sparse.indices_count < h.nnz || h.sparse.offsets_count < major + 1)
        throw std::invalid_argument("sparse buffer capacity is too small");
    address(h.sparse.values, product(h.nnz, 2));
    if (h.nnz && reinterpret_cast<std::uintptr_t>(h.sparse.values) % alignof(std::uint16_t))
        throw std::invalid_argument("unaligned binary16 sparse values");
    address(h.sparse.indices, product(h.nnz, iw));
    address(h.sparse.offsets, product(major + 1, ow));
    std::size_t previous = integer(h.sparse.offsets, h.sparse.offset_type, 0);
    if (previous != 0) throw std::invalid_argument("sparse offsets must start at zero");
    for (std::size_t i = 1; i <= major; ++i) {
        auto next = integer(h.sparse.offsets, h.sparse.offset_type, i);
        if (next < previous || next > h.nnz) throw std::invalid_argument("invalid sparse offsets");
        previous = next;
    }
    if (previous != h.nnz) throw std::invalid_argument("sparse offsets must end at nnz");
    for (std::size_t i = 0; i < h.nnz; ++i)
        if (integer(h.sparse.indices, h.sparse.index_type, i) >= minor) throw std::invalid_argument("sparse index out of bounds");
}
std::shared_ptr<detail::HalfStorage> dense_storage(const std::uint16_t* data, std::vector<std::size_t> shape,
    std::vector<std::ptrdiff_t> strides, std::shared_ptr<const void> owner) {
    auto h = std::make_shared<detail::HalfStorage>();
    h->values = data; h->shape = std::move(shape); h->strides = std::move(strides); h->owner = std::move(owner);
    dense_validate(*h);
    return h;
}
std::shared_ptr<detail::HalfStorage> sparse_storage(Layout layout, std::size_t rows, std::size_t columns,
    std::size_t nnz, SparseF16 buffers, std::shared_ptr<const void> owner) {
    auto h = std::make_shared<detail::HalfStorage>();
    h->shape = {rows, columns}; h->layout = layout; h->sparse = buffers; h->nnz = nnz; h->owner = std::move(owner);
    sparse_validate(*h);
    return h;
}
void require_owner(const std::shared_ptr<const void>& owner) {
    if (!owner.use_count()) throw std::invalid_argument("retained binary16 buffer requires an owner");
}
}

NumericView::NumericView(anndatapp::ArrayView array) : array_(std::move(array)) {}
NumericView NumericView::borrow_dense_f16(const std::uint16_t* p, std::vector<std::size_t> shape, std::vector<std::ptrdiff_t> strides) {
    NumericView v; v.half_ = dense_storage(p, std::move(shape), std::move(strides), {}); return v;
}
NumericView NumericView::retain_dense_f16(const std::uint16_t* p, std::vector<std::size_t> shape, std::vector<std::ptrdiff_t> strides, std::shared_ptr<const void> owner) {
    require_owner(owner); NumericView v; v.half_ = dense_storage(p, std::move(shape), std::move(strides), std::move(owner)); return v;
}
NumericView NumericView::borrow_sparse_f16(Layout layout, std::size_t rows, std::size_t cols, std::size_t nnz, SparseF16 b) {
    NumericView v; v.half_ = sparse_storage(layout, rows, cols, nnz, b, {}); return v;
}
NumericView NumericView::retain_sparse_f16(Layout layout, std::size_t rows, std::size_t cols, std::size_t nnz, SparseF16 b, std::shared_ptr<const void> owner) {
    require_owner(owner); NumericView v; v.half_ = sparse_storage(layout, rows, cols, nnz, b, std::move(owner)); return v;
}
std::size_t NumericView::rank() const noexcept { return half_ ? half_->shape.size() : array_.rank(); }
std::size_t NumericView::extent(std::size_t axis) const {
    if (!half_) return array_.extent(axis);
    auto n = half_->shape.at(axis);
    return axes_.empty() ? n : axes_[axis].extent(n);
}
StorageType NumericView::dtype() const {
    if (half_) return StorageType::float16;
    auto t = array_.dtype();
    if (t == DType::float32) return StorageType::float32;
    if (t == DType::float64) return StorageType::float64;
    throw std::invalid_argument("numeric storage requires float32 or float64 AnnData array");
}
Layout NumericView::layout() const { return half_ ? half_->layout : array_.layout(); }
DType NumericView::index_dtype() const {
    if (!half_) return array_.index_dtype();
    if (layout() == Layout::dense) throw std::invalid_argument("dense storage has no sparse indices");
    return half_->sparse.index_type;
}
DType NumericView::offset_dtype() const {
    if (!half_) return array_.offset_dtype();
    if (layout() == Layout::dense) throw std::invalid_argument("dense storage has no sparse offsets");
    return half_->sparse.offset_type;
}
void NumericView::validate() const {
    if (!half_) { (void)dtype(); array_.validate(); return; }
    if (layout() == Layout::dense) dense_validate(*half_); else sparse_validate(*half_);
}
NumericView NumericView::select(const std::vector<anndatapp::Axis>& axes) const {
    if (!half_) return NumericView(array_.select(axes.data(), axes.size()));
    if (axes.size() != rank()) throw std::invalid_argument("numeric selection rank mismatch");
    NumericView out = *this; out.axes_.resize(rank());
    for (std::size_t a = 0; a < rank(); ++a)
        out.axes_[a] = anndatapp::detail::compose_axis(axes_.empty() ? anndatapp::Axis::all() : axes_[a], axes[a], half_->shape[a]);
    return out;
}
NumericBytes<const std::byte> NumericView::bytes() const {
    NumericBytes<const std::byte> out;
    if (!half_) {
        (void)dtype(); out.array_ = array_.bytes(); auto& b = *out.array_;
        out.data_ = b.data(); out.width_ = b.element_size(); out.contiguous_ = b.contiguous();
        for (std::size_t a = 0; a < b.rank(); ++a) { out.shape_.push_back(b.extent(a)); out.strides_.push_back(b.stride_bytes(a)); }
        return out;
    }
    if (layout() != Layout::dense) throw std::invalid_argument("sparse numeric storage has no dense bytes");
    out.half_ = half_; out.width_ = 2;
    bool empty = false;
    for (std::size_t a = 0; a < rank(); ++a) { out.shape_.push_back(extent(a)); empty |= extent(a) == 0; }
    std::ptrdiff_t origin = 0;
    for (std::size_t a = 0; a < rank(); ++a) {
        auto stride = mul(half_->strides[a], 2);
        if (!empty && !axes_.empty() && !axes_[a].is_all() && stride) {
            auto& axis = axes_[a]; auto start = as_signed(axis.index(0));
            std::ptrdiff_t step = 1;
            if (extent(a) > 1) {
                if (axis.is_gather()) {
                    step = add(as_signed(axis.index(1)), -start);
                    for (std::size_t i = 2; i < extent(a); ++i)
                        if (as_signed(axis.index(i)) != add(start, mul(as_signed(i), step)))
                            throw std::invalid_argument("gathered numeric selection has no strided view");
                } else { step = as_signed(axis.step()); if (axis.reverse()) step = -step; }
            }
            origin = add(origin, mul(start, stride)); stride = mul(stride, step);
        }
        out.strides_.push_back(stride);
    }
    auto p = reinterpret_cast<std::uintptr_t>(half_->values);
    if (!empty) p = origin < 0 ? p - (std::uintptr_t(-(origin + 1)) + 1) : p + std::uintptr_t(origin);
    out.data_ = reinterpret_cast<const std::byte*>(p);
    out.contiguous_ = true;
    std::ptrdiff_t expected = 2;
    for (std::size_t a = rank(); a-- > 0;) {
        if (out.shape_[a] > 1 && out.strides_[a] != expected) out.contiguous_ = false;
        if (out.shape_[a] && expected > std::numeric_limits<std::ptrdiff_t>::max()/as_signed(out.shape_[a])) out.contiguous_ = false;
        else expected *= as_signed(out.shape_[a]);
    }
    return out;
}
NumericSparseBytes NumericView::sparse_bytes() const {
    NumericSparseBytes out;
    if (!half_) {
        (void)dtype(); out.array_.emplace(array_.sparse_bytes()); auto& b = *out.array_;
        out.values_ = b.values(); out.indices_ = b.indices(); out.offsets_ = b.offsets();
        out.rows_ = b.rows(); out.columns_ = b.columns(); out.nnz_ = b.nnz();
        out.value_size_ = b.value_size(); out.index_size_ = b.index_size(); out.offset_size_ = b.offset_size();
        return out;
    }
    if (layout() == Layout::dense) throw std::invalid_argument("dense numeric storage has no sparse bytes");
    for (auto& axis : axes_) if (!axis.is_all()) throw std::invalid_argument("selected sparse numeric storage has no direct buffers");
    out.half_ = half_; out.values_ = reinterpret_cast<const std::byte*>(half_->sparse.values);
    out.indices_ = static_cast<const std::byte*>(half_->sparse.indices); out.offsets_ = static_cast<const std::byte*>(half_->sparse.offsets);
    out.rows_ = half_->shape[0]; out.columns_ = half_->shape[1]; out.nnz_ = half_->nnz; out.value_size_ = 2;
    out.index_size_ = integer_width(half_->sparse.index_type); out.offset_size_ = integer_width(half_->sparse.offset_type);
    return out;
}
NumericEdit::NumericEdit(anndatapp::ArrayEdit array) : array_(std::move(array)) {}
NumericEdit NumericEdit::borrow_dense_f16(std::uint16_t* p, std::vector<std::size_t> shape, std::vector<std::ptrdiff_t> strides) {
    NumericEdit e; e.half_ = dense_storage(p, std::move(shape), std::move(strides), {}); return e;
}
NumericEdit NumericEdit::retain_dense_f16(std::uint16_t* p, std::vector<std::size_t> shape, std::vector<std::ptrdiff_t> strides, std::shared_ptr<const void> owner) {
    require_owner(owner); NumericEdit e; e.half_ = dense_storage(p, std::move(shape), std::move(strides), std::move(owner)); return e;
}
NumericView NumericEdit::view() const { NumericView v; v.half_ = half_; if (!half_) v.array_ = array_.view(); return v; }
NumericBytes<std::byte> NumericEdit::bytes() const {
    NumericBytes<std::byte> out;
    if (!half_) {
        (void)view().dtype(); out.array_ = array_.bytes(); auto& b = *out.array_;
        out.data_ = b.data(); out.width_ = b.element_size(); out.contiguous_ = b.contiguous();
        for (std::size_t a = 0; a < b.rank(); ++a) { out.shape_.push_back(b.extent(a)); out.strides_.push_back(b.stride_bytes(a)); }
    } else {
        auto b = view().bytes(); out.half_ = half_; out.data_ = const_cast<std::byte*>(b.data());
        out.shape_ = std::move(b.shape_); out.strides_ = std::move(b.strides_); out.width_ = b.width_; out.contiguous_ = b.contiguous_;
    }
    return out;
}
}
