module;
#include <Cellerator/ops/numeric/binary16.hh>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

module velocity;
import anndatapp.core;

namespace velocity::reference {
namespace {

using anndatapp::DType;
using anndatapp::Layout;
using ArrayView = NumericView;
using std::size_t;

struct Entry { size_t column, position; };

struct DenseInput {
    const std::byte* values;
    std::ptrdiff_t row_stride, column_stride;
    size_t n;
};

struct SparseInput {
    const std::byte* values;
    const std::byte* indices;
    const std::byte* offsets;
    StorageType value_type;
    DType index_type, offset_type;
    size_t nnz, n;
};

struct SparseRows {
    std::vector<size_t> offsets;
    std::vector<Entry> entries; // CSC is indexed by row; CSR stays in its input.
};

size_t checked_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::length_error("neighborhood size overflows size_t");
    return a + b;
}

size_t checked_extent(size_t n) {
    if (n == std::numeric_limits<size_t>::max() ||
        n + 1 > std::vector<size_t>().max_size())
        throw std::length_error("neighborhood row count exceeds addressable storage");
    return n + 1;
}

template<class T> T load(const std::byte* p) {
    T value;
    std::memcpy(&value, p, sizeof value);
    return value;
}

template<class T> auto numeric_value(const std::byte* p) {
    if constexpr(std::is_same_v<T,std::uint16_t>)return cellerator::numeric::load_f16(load<T>(p));
    else return load<T>(p);
}

template<class T> size_t as_size(const std::byte* p) {
    const T value = load<T>(p);
    if constexpr (std::is_signed_v<T>)
        if (value < 0) throw std::invalid_argument("sparse index or offset is negative");
    using U = std::make_unsigned_t<T>;
    const auto u = static_cast<U>(value);
    if constexpr (sizeof(U) > sizeof(size_t))
        if (u > std::numeric_limits<size_t>::max())
            throw std::length_error("sparse index or offset exceeds size_t");
    return static_cast<size_t>(u);
}

size_t integer_at(const std::byte* p, DType type, size_t i) {
    p += i * anndatapp::dtype_size(type);
    switch (type) {
    case DType::int8: return as_size<std::int8_t>(p);
    case DType::uint8: return as_size<std::uint8_t>(p);
    case DType::int16: return as_size<std::int16_t>(p);
    case DType::uint16: return as_size<std::uint16_t>(p);
    case DType::int32: return as_size<std::int32_t>(p);
    case DType::uint32: return as_size<std::uint32_t>(p);
    case DType::int64: return as_size<std::int64_t>(p);
    case DType::uint64: return as_size<std::uint64_t>(p);
    default: throw std::invalid_argument("sparse indices and offsets must be integers");
    }
}

std::ptrdiff_t scaled_stride(std::ptrdiff_t stride, size_t i) {
    if (!stride || !i) return 0;
    using U = std::make_unsigned_t<std::ptrdiff_t>;
    if (i > std::numeric_limits<U>::max())
        throw std::length_error("dense neighborhood stride overflows ptrdiff_t");
    const auto k = static_cast<U>(i);
    if (stride > 0) {
        const auto s = static_cast<U>(stride);
        if (k > static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max()) / s)
            throw std::length_error("dense neighborhood stride overflows ptrdiff_t");
        return static_cast<std::ptrdiff_t>(k * s);
    }
    const auto magnitude = static_cast<U>(-(stride + 1)) + 1;
    const auto limit = static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max()) + 1;
    if (k > limit / magnitude)
        throw std::length_error("dense neighborhood stride overflows ptrdiff_t");
    const auto product = k * magnitude;
    if (product == limit) return std::numeric_limits<std::ptrdiff_t>::min();
    return -static_cast<std::ptrdiff_t>(product);
}

std::ptrdiff_t checked_offset(std::ptrdiff_t a, std::ptrdiff_t b) {
    if ((b > 0 && a > std::numeric_limits<std::ptrdiff_t>::max() - b) ||
        (b < 0 && a < std::numeric_limits<std::ptrdiff_t>::min() - b))
        throw std::length_error("dense neighborhood offset overflows ptrdiff_t");
    return a + b;
}

template<class T> void check_dense(const DenseInput& input) {
    for (size_t i = 0; i < input.n; ++i) {
        const auto row = scaled_stride(input.row_stride, i);
        for (size_t j = 0; j < input.n; ++j) {
            const auto offset = checked_offset(row, scaled_stride(input.column_stride, j));
            if (!std::isfinite(numeric_value<T>(input.values + offset)))
                throw std::invalid_argument("connectivities contain a nonfinite value");
        }
    }
}

template<class T> void check_sparse(const SparseInput& input) {
    for (size_t p = 0; p < input.nnz; ++p)
        if (!std::isfinite(numeric_value<T>(input.values + p * sizeof(T))))
            throw std::invalid_argument("connectivities contain a nonfinite value");
}

template<class T> DenseInput dense_input(const ArrayView& view, size_t n) {
    const auto bytes = view.bytes();
    DenseInput input{bytes.data(), bytes.stride_bytes(0), bytes.stride_bytes(1), n};
    check_dense<T>(input);
    return input;
}

SparseInput sparse_input(const ArrayView& view, size_t n) {
    const auto bytes = view.sparse_bytes(); // Also rejects selected sparse views.
    SparseInput input{bytes.values(), bytes.indices(), bytes.offsets(), view.dtype(),
                      view.index_dtype(), view.offset_dtype(), bytes.nnz(), n};
    if(input.value_type==StorageType::float16)check_sparse<std::uint16_t>(input);
    else if (input.value_type == StorageType::float32) check_sparse<float>(input);
    else check_sparse<double>(input);
    return input;
}

std::vector<size_t> source_offsets(const SparseInput& input) {
    std::vector<size_t> offsets(checked_extent(input.n));
    for (size_t i = 0; i <= input.n; ++i) {
        offsets[i] = integer_at(input.offsets, input.offset_type, i);
        if (offsets[i] > input.nnz)
            throw std::invalid_argument("sparse offset exceeds stored entries");
    }
    if (offsets.front() != 0 || offsets.back() != input.nnz)
        throw std::invalid_argument("sparse offsets do not span stored entries");
    for (size_t i = 1; i <= input.n; ++i)
        if (offsets[i] < offsets[i - 1])
            throw std::invalid_argument("sparse offsets are not monotone");
    return offsets;
}

SparseRows csc_rows(const SparseInput& input) {
    SparseRows rows;
    rows.offsets.assign(checked_extent(input.n), 0);
    const auto columns = source_offsets(input);
    for (size_t p = 0; p < input.nnz; ++p) {
        const auto row = integer_at(input.indices, input.index_type, p);
        if (row >= input.n) throw std::invalid_argument("sparse row index is out of bounds");
        rows.offsets[row + 1] = checked_add(rows.offsets[row + 1], 1);
    }
    for (size_t i = 1; i <= input.n; ++i)
        rows.offsets[i] = checked_add(rows.offsets[i], rows.offsets[i - 1]);
    if (input.nnz > std::vector<Entry>().max_size())
        throw std::length_error("sparse structure exceeds addressable storage");
    rows.entries.resize(input.nnz);
    auto cursor = rows.offsets;
    for (size_t column = 0; column < input.n; ++column)
        for (size_t p = columns[column]; p < columns[column + 1]; ++p) {
            const auto row = integer_at(input.indices, input.index_type, p);
            rows.entries[cursor[row]++] = {column, p};
        }
    return rows;
}

template<class T, class Emit>
size_t grouped_row(const SparseInput& input, const Entry* entries, size_t count,
                   size_t row, Emit&& emit) {
    size_t output_count = 0, p = 0;
    bool diagonal = false;
    while (p < count) {
        const auto column = entries[p].column;
        using C=std::conditional_t<std::is_same_v<T,std::uint16_t>,float,T>;
        C sum = C(0);
        do {
            sum = C(sum + numeric_value<T>(input.values + entries[p].position * sizeof(T)));
            ++p;
        } while (p < count && entries[p].column == column);

        const bool positive = sum > 0;
        if (column == row) {
            if (positive || !diagonal) {
                emit(row);
                ++output_count;
            }
            diagonal = true;
        } else {
            if (column > row && !diagonal) {
                emit(row);
                ++output_count;
                diagonal = true;
            }
            if (positive) {
                emit(column);
                ++output_count;
            }
        }
    }
    if (!diagonal) {
        emit(row);
        ++output_count;
    }
    return output_count;
}

template<class T, class Emit>
size_t csr_row(const SparseInput& input, const std::vector<size_t>& offsets,
               size_t row, std::vector<Entry>& scratch, Emit&& emit) {
    scratch.clear();
    for (size_t p = offsets[row]; p < offsets[row + 1]; ++p) {
        const auto column = integer_at(input.indices, input.index_type, p);
        if (column >= input.n) throw std::invalid_argument("sparse column index is out of bounds");
        scratch.push_back({column, p});
    }
    std::sort(scratch.begin(), scratch.end(), [](const Entry& a, const Entry& b) {
        return a.column < b.column || (a.column == b.column && a.position < b.position);
    });
    return grouped_row<T>(input, scratch.data(), scratch.size(), row,
                          std::forward<Emit>(emit));
}

template<class T, class Emit>
size_t sparse_row(const SparseInput& input, Layout layout,
                  const std::vector<size_t>& offsets, const SparseRows* csc,
                  size_t row, std::vector<Entry>& scratch, Emit&& emit) {
    if (layout == Layout::csr)
        return csr_row<T>(input, offsets, row, scratch, std::forward<Emit>(emit));
    const auto first = csc->offsets[row], last = csc->offsets[row + 1];
    const auto* entries = first == 0 && csc->entries.empty()
        ? nullptr : csc->entries.data() + first;
    return grouped_row<T>(input, entries, last - first,
                          row, std::forward<Emit>(emit));
}

template<class T, class Emit>
size_t dense_row(const DenseInput& input, size_t row, Emit&& emit) {
    const auto row_offset = scaled_stride(input.row_stride, row);
    size_t count = 0;
    for (size_t column = 0; column < input.n; ++column) {
        const auto offset = checked_offset(row_offset,
            scaled_stride(input.column_stride, column));
        if (column == row || numeric_value<T>(input.values + offset) > 0) {
            emit(column);
            ++count;
        }
    }
    return count;
}

DType unsigned_type(size_t value) {
    if (value <= std::numeric_limits<std::uint8_t>::max()) return DType::uint8;
    if (value <= std::numeric_limits<std::uint16_t>::max()) return DType::uint16;
    if (value <= std::numeric_limits<std::uint32_t>::max()) return DType::uint32;
    return DType::uint64;
}

template<class I, class O> anndatapp::Array make_csr(size_t n, size_t nnz) {
    return anndatapp::Array::csr<float, I, O>(n, n, nnz);
}

anndatapp::Array make_csr(size_t n, size_t nnz, DType index, DType offset) {
#define VELOCITY_CSR_CASE(I, ID) \
    case DType::ID: \
        switch (offset) { \
        case DType::uint8: return make_csr<I, std::uint8_t>(n, nnz); \
        case DType::uint16: return make_csr<I, std::uint16_t>(n, nnz); \
        case DType::uint32: return make_csr<I, std::uint32_t>(n, nnz); \
        case DType::uint64: return make_csr<I, std::uint64_t>(n, nnz); \
        default: break; \
        } \
        break
    switch (index) {
    VELOCITY_CSR_CASE(std::uint8_t, uint8);
    VELOCITY_CSR_CASE(std::uint16_t, uint16);
    VELOCITY_CSR_CASE(std::uint32_t, uint32);
    VELOCITY_CSR_CASE(std::uint64_t, uint64);
    default: break;
    }
#undef VELOCITY_CSR_CASE
    throw std::invalid_argument("unsupported output sparse index type");
}

void store_unsigned(std::byte* p, size_t value, DType type) {
    switch (type) {
    case DType::uint8: { const auto v = static_cast<std::uint8_t>(value); std::memcpy(p, &v, sizeof v); return; }
    case DType::uint16: { const auto v = static_cast<std::uint16_t>(value); std::memcpy(p, &v, sizeof v); return; }
    case DType::uint32: { const auto v = static_cast<std::uint32_t>(value); std::memcpy(p, &v, sizeof v); return; }
    case DType::uint64: { const auto v = static_cast<std::uint64_t>(value); std::memcpy(p, &v, sizeof v); return; }
    default: throw std::invalid_argument("output sparse type must be unsigned");
    }
}

template<class T, class Row>
anndatapp::Array finish(size_t n, Row&& row, const std::vector<size_t>& counts) {
    std::vector<size_t> offsets(checked_extent(n));
    for (size_t i = 0; i < n; ++i) offsets[i + 1] = checked_add(offsets[i], counts[i]);
    const auto nnz = offsets[n];
    const auto index_type = unsigned_type(n ? n - 1 : 0);
    const auto offset_type = unsigned_type(nnz);
    auto result = make_csr(n, nnz, index_type, offset_type);
    auto edit = result.edit();
    auto output = edit.sparse_bytes();
    for (size_t i = 0; i <= n; ++i)
        store_unsigned(output.unsafe_offsets() + i * output.offset_size(), offsets[i], offset_type);

    auto* values = output.values();
    auto* indices = output.unsafe_indices();
    for (size_t i = 0; i < n; ++i) {
        const auto count = counts[i];
        const float weight = static_cast<float>(1.0 / static_cast<double>(count));
        size_t p = offsets[i];
        const auto written = row(i, [&](size_t column) {
            store_unsigned(indices + p * output.index_size(), column, index_type);
            std::memcpy(values + p * sizeof(float), &weight, sizeof weight);
            ++p;
        });
        if (written != count || p != offsets[i + 1])
            throw std::logic_error("neighborhood structure changed during preparation");
    }
    return result;
}

template<class T> anndatapp::Array prepare_dense(DenseInput input) {
    std::vector<size_t> counts(checked_extent(input.n) - 1);
    for (size_t i = 0; i < input.n; ++i)
        counts[i] = dense_row<T>(input, i, [](size_t) {});
    return finish<T>(input.n, [&](size_t i, auto&& emit) {
        return dense_row<T>(input, i, std::forward<decltype(emit)>(emit));
    }, counts);
}

template<class T> anndatapp::Array prepare_sparse(SparseInput input, Layout layout) {
    const auto offsets = source_offsets(input);
    SparseRows csc;
    const SparseRows* csc_ptr = nullptr;
    if (layout == Layout::csc) { csc = csc_rows(input); csc_ptr = &csc; }
    std::vector<Entry> scratch;
    std::vector<size_t> counts(input.n);
    for (size_t i = 0; i < input.n; ++i)
        counts[i] = sparse_row<T>(input, layout, offsets, csc_ptr, i, scratch,
                                  [](size_t) {});
    return finish<T>(input.n, [&](size_t i, auto&& emit) {
        return sparse_row<T>(input, layout, offsets, csc_ptr, i, scratch,
                             std::forward<decltype(emit)>(emit));
    }, counts);
}

} // namespace

anndatapp::Array prepare_neighborhood(ArrayView connectivities) {
    if (connectivities.rank() != 2)
        throw std::invalid_argument("connectivities must have rank two");
    const auto rows = connectivities.extent(0), columns = connectivities.extent(1);
    if (rows != columns) throw std::invalid_argument("connectivities must be square");
    if (connectivities.dtype() != StorageType::float16 && connectivities.dtype() != StorageType::float32 && connectivities.dtype() != StorageType::float64)
        throw std::invalid_argument("connectivities must be float16, float32 or float64");
    const auto layout = connectivities.layout();
    if (layout != Layout::dense && layout != Layout::csr && layout != Layout::csc)
        throw std::invalid_argument("connectivities must be dense, CSR, or CSC");
    connectivities.validate();

    if (layout == Layout::dense) {
        if(connectivities.dtype()==StorageType::float16)return prepare_dense<std::uint16_t>(dense_input<std::uint16_t>(connectivities,rows));
        if (connectivities.dtype() == StorageType::float32)
            return prepare_dense<float>(dense_input<float>(connectivities, rows));
        return prepare_dense<double>(dense_input<double>(connectivities, rows));
    }
    auto input = sparse_input(connectivities, rows);
    if(input.value_type==StorageType::float16)return prepare_sparse<std::uint16_t>(input,layout);
    if (input.value_type == StorageType::float32)
        return prepare_sparse<float>(input, layout);
    return prepare_sparse<double>(input, layout);
}

} // namespace velocity::reference
