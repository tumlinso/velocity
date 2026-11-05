module;
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

export module velocity:numeric;
export import anndatapp.core;

export namespace velocity {

enum class StorageType { float16, float32, float64 };

// All capacities are element counts. Borrowed buffers and their sparse structure
// must outlive every view and remain stable while the view is used.
struct SparseF16 {
    const std::uint16_t* values = nullptr;
    std::size_t values_count = 0;
    const void* indices = nullptr;
    std::size_t indices_count = 0;
    anndatapp::DType index_type = anndatapp::DType::int32;
    const void* offsets = nullptr;
    std::size_t offsets_count = 0;
    anndatapp::DType offset_type = anndatapp::DType::int32;
};

namespace detail { struct HalfStorage; }

template<class Byte> class NumericBytes {
public:
    Byte* data() const noexcept { return data_; }
    std::size_t rank() const noexcept { return shape_.size(); }
    std::size_t extent(std::size_t axis) const { return shape_.at(axis); }
    std::ptrdiff_t stride_bytes(std::size_t axis) const { return strides_.at(axis); }
    std::size_t element_size() const noexcept { return width_; }
    bool contiguous() const noexcept { return contiguous_; }
private:
    template<class> friend class NumericDense;
    friend class NumericView;
    friend class NumericEdit;
    Byte* data_ = nullptr;
    std::vector<std::size_t> shape_;
    std::vector<std::ptrdiff_t> strides_;
    std::size_t width_ = 0;
    bool contiguous_ = false;
    std::shared_ptr<const detail::HalfStorage> half_;
    std::optional<anndatapp::DenseBytesView<Byte>> array_;
};

template<class T> class NumericDense {
public:
    T* data() const noexcept { return reinterpret_cast<T*>(bytes_.data()); }
    std::size_t rank() const noexcept { return bytes_.rank(); }
    std::size_t extent(std::size_t axis) const { return bytes_.extent(axis); }
    std::ptrdiff_t stride(std::size_t axis) const { return bytes_.stride_bytes(axis)/std::ptrdiff_t(sizeof(T)); }
    bool contiguous() const noexcept { return bytes_.contiguous(); }
    T* contiguous_data() const {
        if (!contiguous()) throw std::invalid_argument("numeric view is not contiguous");
        return data();
    }
private:
    friend class NumericView;
    friend class NumericEdit;
    explicit NumericDense(NumericBytes<std::conditional_t<std::is_const_v<T>, const std::byte, std::byte>> bytes)
        : bytes_(std::move(bytes)) {}
    NumericBytes<std::conditional_t<std::is_const_v<T>, const std::byte, std::byte>> bytes_;
};

class NumericSparseBytes {
public:
    const std::byte* values() const noexcept { return values_; }
    const std::byte* indices() const noexcept { return indices_; }
    const std::byte* offsets() const noexcept { return offsets_; }
    std::size_t rows() const noexcept { return rows_; }
    std::size_t columns() const noexcept { return columns_; }
    std::size_t nnz() const noexcept { return nnz_; }
    std::size_t value_size() const noexcept { return value_size_; }
    std::size_t index_size() const noexcept { return index_size_; }
    std::size_t offset_size() const noexcept { return offset_size_; }
private:
    friend class NumericView;
    const std::byte *values_ = nullptr, *indices_ = nullptr, *offsets_ = nullptr;
    std::size_t rows_ = 0, columns_ = 0, nnz_ = 0, value_size_ = 0, index_size_ = 0, offset_size_ = 0;
    std::shared_ptr<const detail::HalfStorage> half_;
    std::optional<anndatapp::SparseBytesView<const std::byte>> array_;
};

class NumericView {
public:
    NumericView() = default;
    NumericView(anndatapp::ArrayView array);
    static NumericView borrow_dense_f16(const std::uint16_t*, std::vector<std::size_t>, std::vector<std::ptrdiff_t> strides);
    static NumericView retain_dense_f16(const std::uint16_t*, std::vector<std::size_t>, std::vector<std::ptrdiff_t> strides, std::shared_ptr<const void> owner);
    static NumericView borrow_sparse_f16(anndatapp::Layout, std::size_t rows, std::size_t columns, std::size_t nnz, SparseF16);
    static NumericView retain_sparse_f16(anndatapp::Layout, std::size_t rows, std::size_t columns, std::size_t nnz, SparseF16, std::shared_ptr<const void> owner);
    std::size_t rank() const noexcept;
    std::size_t extent(std::size_t axis) const;
    StorageType dtype() const;
    anndatapp::Layout layout() const;
    anndatapp::DType index_dtype() const;
    anndatapp::DType offset_dtype() const;
    void validate() const;
    NumericView select(const std::vector<anndatapp::Axis>& axes) const;
    NumericView select(std::initializer_list<anndatapp::Axis> axes) const { return select(std::vector<anndatapp::Axis>(axes)); }
    NumericBytes<const std::byte> bytes() const;
    NumericSparseBytes sparse_bytes() const;
    template<class T> NumericDense<const T> dense() const {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, std::uint16_t>);
        auto expected = std::is_same_v<T, float> ? StorageType::float32 : std::is_same_v<T, double> ? StorageType::float64 : StorageType::float16;
        if (dtype() != expected) throw std::invalid_argument("numeric view dtype mismatch");
        if constexpr (!std::is_same_v<T, std::uint16_t>) if (!half_) (void)array_.dense<T>();
        return NumericDense<const T>(bytes());
    }
private:
    friend class NumericEdit;
    std::shared_ptr<const detail::HalfStorage> half_;
    std::vector<anndatapp::Axis> axes_;
    anndatapp::ArrayView array_;
};

class NumericEdit {
public:
    NumericEdit() = default;
    NumericEdit(anndatapp::ArrayEdit array);
    static NumericEdit borrow_dense_f16(std::uint16_t*, std::vector<std::size_t>, std::vector<std::ptrdiff_t> strides);
    static NumericEdit retain_dense_f16(std::uint16_t*, std::vector<std::size_t>, std::vector<std::ptrdiff_t> strides, std::shared_ptr<const void> owner);
    NumericView view() const;
    NumericBytes<std::byte> bytes() const;
    template<class T> NumericDense<T> dense() const {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, std::uint16_t>);
        auto expected = std::is_same_v<T, float> ? StorageType::float32 : std::is_same_v<T, double> ? StorageType::float64 : StorageType::float16;
        if (view().dtype() != expected) throw std::invalid_argument("numeric edit dtype mismatch");
        if constexpr (!std::is_same_v<T, std::uint16_t>) if (!half_) (void)array_.dense<T>();
        return NumericDense<T>(bytes());
    }
private:
    std::shared_ptr<const detail::HalfStorage> half_;
    anndatapp::ArrayEdit array_;
};

}
