module;
#include <float.h>
#include <stddef.h>
#include <stdint.h>

module velocitypp;
import anndatapp.core;

namespace velocitypp::reference {
namespace {

using anndatapp::DType;
using anndatapp::Layout;

struct MatrixData {
    const unsigned char* values;
    const unsigned char* indices;
    const unsigned char* offsets;
    ptrdiff_t row_stride, column_stride;
    size_t n, nnz;
    DType dtype, index_type, offset_type;
    Layout layout;
};

Status status(Code code, size_t location = 0, size_t required = 0) {
    return {code, location, required};
}

size_t value_width(DType type) {
    if (type == DType::float32) return sizeof(float);
    if (type == DType::float64) return sizeof(double);
    return 0;
}

size_t integer_width(DType type) {
    switch (type) {
    case DType::int8: case DType::uint8: return 1;
    case DType::int16: case DType::uint16: return 2;
    case DType::int32: case DType::uint32: return 4;
    case DType::int64: case DType::uint64: return 8;
    default: return 0;
    }
}

bool add_size(size_t a, size_t b, size_t& out) {
    if (b > SIZE_MAX - a) return false;
    out = a + b;
    return true;
}

bool multiply_size(size_t a, size_t b, size_t& out) {
    if (a && b > SIZE_MAX / a) return false;
    out = a * b;
    return true;
}

bool scaled_stride(ptrdiff_t stride, size_t i, ptrdiff_t& out) {
    if (!stride || !i) { out = 0; return true; }
    if (stride > 0) {
        const size_t s = static_cast<size_t>(stride);
        if (i > static_cast<size_t>(PTRDIFF_MAX) / s) return false;
        out = static_cast<ptrdiff_t>(i * s);
        return true;
    }
    const size_t magnitude = static_cast<size_t>(-(stride + 1)) + 1;
    const size_t limit = static_cast<size_t>(PTRDIFF_MAX) + 1;
    if (i > limit / magnitude) return false;
    const size_t product = i * magnitude;
    if (product == limit) out = PTRDIFF_MIN;
    else out = -static_cast<ptrdiff_t>(product);
    return true;
}

bool add_offset(ptrdiff_t a, ptrdiff_t b, ptrdiff_t& out) {
    if ((b > 0 && a > PTRDIFF_MAX - b) ||
        (b < 0 && a < PTRDIFF_MIN - b)) return false;
    out = a + b;
    return true;
}

bool dense_offset(const MatrixData& input, size_t row, size_t column, ptrdiff_t& out) {
    ptrdiff_t row_offset, column_offset;
    return scaled_stride(input.row_stride, row, row_offset) &&
        scaled_stride(input.column_stride, column, column_offset) &&
        add_offset(row_offset, column_offset, out);
}

template<class T> T load_value(const unsigned char* p) {
    T value;
    __builtin_memcpy(&value, p, sizeof value);
    return value;
}

bool finite_value(float x) {
    return x == x && x <= FLT_MAX && x >= -FLT_MAX;
}

bool finite_value(double x) {
    return x == x && x <= DBL_MAX && x >= -DBL_MAX;
}

Status to_size(uint64_t value, size_t location, size_t& out) {
    const size_t converted = static_cast<size_t>(value);
    if (static_cast<uint64_t>(converted) != value)
        return status(Code::overflow, location);
    out = converted;
    return status(Code::ok);
}

template<class T> Status read_signed(const unsigned char* p, size_t location, size_t& out) {
    T value;
    __builtin_memcpy(&value, p, sizeof value);
    if (value < 0) return status(Code::bounds, location);
    return to_size(static_cast<uint64_t>(value), location, out);
}

template<class T> Status read_unsigned(const unsigned char* p, size_t location, size_t& out) {
    T value;
    __builtin_memcpy(&value, p, sizeof value);
    return to_size(static_cast<uint64_t>(value), location, out);
}

Status read_integer(const unsigned char* bytes, DType type, size_t width,
                    size_t i, size_t& out) {
    if (i && width > SIZE_MAX / i) return status(Code::overflow, i);
    const auto* p = bytes + i * width;
    switch (type) {
    case DType::int8: return read_signed<int8_t>(p, i, out);
    case DType::uint8: return read_unsigned<uint8_t>(p, i, out);
    case DType::int16: return read_signed<int16_t>(p, i, out);
    case DType::uint16: return read_unsigned<uint16_t>(p, i, out);
    case DType::int32: return read_signed<int32_t>(p, i, out);
    case DType::uint32: return read_unsigned<uint32_t>(p, i, out);
    case DType::int64: return read_signed<int64_t>(p, i, out);
    case DType::uint64: return read_unsigned<uint64_t>(p, i, out);
    default: return status(Code::dtype, i);
    }
}

size_t integer_at(const unsigned char* bytes, DType type, size_t width, size_t i) {
    const auto* p = bytes + i * width;
    switch (type) {
    case DType::int8: return static_cast<size_t>(load_value<int8_t>(p));
    case DType::uint8: return static_cast<size_t>(load_value<uint8_t>(p));
    case DType::int16: return static_cast<size_t>(load_value<int16_t>(p));
    case DType::uint16: return static_cast<size_t>(load_value<uint16_t>(p));
    case DType::int32: return static_cast<size_t>(load_value<int32_t>(p));
    case DType::uint32: return static_cast<size_t>(load_value<uint32_t>(p));
    case DType::int64: return static_cast<size_t>(load_value<int64_t>(p));
    case DType::uint64: return static_cast<size_t>(load_value<uint64_t>(p));
    default: return 0;
    }
}

Status acquire(const AcquiredMatrix& matrix, MatrixData& input) {
    if (matrix.layout == Layout::dense) {
        if (!matrix.dense || matrix.sparse) return status(Code::argument);
        if (matrix.dtype != DType::float32 && matrix.dtype != DType::float64)
            return status(Code::dtype);
        if (matrix.dense->rank() != 2) return status(Code::shape);
        const size_t rows = matrix.dense->extent(0);
        if (rows != matrix.dense->extent(1)) return status(Code::shape);
        if (matrix.dense->element_size() != value_width(matrix.dtype))
            return status(Code::dtype);
        input.values = reinterpret_cast<const unsigned char*>(matrix.dense->data());
        input.indices = nullptr;
        input.offsets = nullptr;
        input.n = rows;
        input.nnz = 0;
        input.dtype = matrix.dtype;
        input.index_type = DType::uint8;
        input.offset_type = DType::uint8;
        input.layout = Layout::dense;
        if (rows && !input.values) return status(Code::argument);
        input.row_stride = matrix.dense->stride_bytes(0);
        input.column_stride = matrix.dense->stride_bytes(1);
        return status(Code::ok);
    }

    if (matrix.layout != Layout::csr && matrix.layout != Layout::csc)
        return status(Code::layout);
    if (!matrix.sparse || matrix.dense) return status(Code::argument);
    if (matrix.dtype != DType::float32 && matrix.dtype != DType::float64)
        return status(Code::dtype);
    if (!integer_width(matrix.index_type) || !integer_width(matrix.offset_type))
        return status(Code::dtype);

    const auto& sparse = *matrix.sparse;
    if (sparse.rows() != sparse.columns()) return status(Code::shape);
    if (sparse.value_size() != value_width(matrix.dtype) ||
        sparse.index_size() != integer_width(matrix.index_type) ||
        sparse.offset_size() != integer_width(matrix.offset_type))
        return status(Code::dtype);
    input.values = reinterpret_cast<const unsigned char*>(sparse.values());
    input.indices = reinterpret_cast<const unsigned char*>(sparse.indices());
    input.offsets = reinterpret_cast<const unsigned char*>(sparse.offsets());
    input.row_stride = 0;
    input.column_stride = 0;
    input.n = sparse.rows();
    input.nnz = sparse.nnz();
    input.dtype = matrix.dtype;
    input.index_type = matrix.index_type;
    input.offset_type = matrix.offset_type;
    input.layout = matrix.layout;
    if (!input.offsets || (input.nnz && (!input.values || !input.indices)))
        return status(Code::argument);
    return status(Code::ok);
}

Status read_sparse_offset(const MatrixData& input, size_t i, size_t& out) {
    return read_integer(input.offsets, input.offset_type,
                        integer_width(input.offset_type), i, out);
}

Status read_sparse_index(const MatrixData& input, size_t i, size_t& out) {
    return read_integer(input.indices, input.index_type,
                        integer_width(input.index_type), i, out);
}

size_t sparse_offset_at(const MatrixData& input, size_t i) {
    return integer_at(input.offsets, input.offset_type,
                      integer_width(input.offset_type), i);
}

size_t sparse_index_at(const MatrixData& input, size_t i) {
    return integer_at(input.indices, input.index_type,
                      integer_width(input.index_type), i);
}

template<class T> bool finite_sparse_values(const MatrixData& input, size_t& bad) {
    for (size_t p = 0; p < input.nnz; ++p)
        if (!finite_value(load_value<T>(input.values + p * sizeof(T)))) {
            bad = p;
            return false;
        }
    return true;
}

size_t sparse_entries_bound(size_t n, size_t nnz, bool& ok) {
    // The output has at most one edge per stored edge plus one diagonal per row.
    // When representable, n*n is a tighter bound even with duplicate input edges.
    size_t sum = 0, square = 0;
    const bool sum_ok = add_size(nnz, n, sum);
    const bool square_ok = multiply_size(n, n, square);
    if (square_ok) {
        ok = true;
        return sum_ok && sum < square ? sum : square;
    }
    ok = sum_ok;
    return sum;
}

Status analyze(const MatrixData& input, NeighborhoodSize& result) {
    NeighborhoodSize required{};
    size_t offset_count;
    if (!add_size(input.n, 1, offset_count)) return status(Code::overflow);

    if (input.layout == Layout::dense) {
        size_t total = 0;
        for (size_t row = 0; row < input.n; ++row) {
            size_t row_count = 0;
            for (size_t column = 0; column < input.n; ++column) {
                ptrdiff_t offset;
                if (!dense_offset(input, row, column, offset))
                    return status(Code::overflow, row);
                const auto* p = input.values + offset;
                bool positive;
                bool finite;
                if (input.dtype == DType::float32) {
                    const float value = load_value<float>(p);
                    finite = finite_value(value);
                    positive = value > 0;
                } else {
                    const double value = load_value<double>(p);
                    finite = finite_value(value);
                    positive = value > 0;
                }
                if (!finite) return status(Code::nonfinite, row);
                if (column == row || positive) {
                    if (!add_size(row_count, 1, row_count) ||
                        !add_size(total, 1, total)) return status(Code::overflow, row);
                }
            }
        }
        required.entries = total;
        result = required;
        return status(Code::ok);
    }

    const size_t offset_width = integer_width(input.offset_type);
    const size_t index_width = integer_width(input.index_type);
    size_t byte_count;
    if (!multiply_size(offset_count, offset_width, byte_count) ||
        !multiply_size(input.nnz, index_width, byte_count) ||
        !multiply_size(input.nnz, value_width(input.dtype), byte_count))
        return status(Code::overflow);

    size_t first;
    Status s = read_sparse_offset(input, 0, first);
    if (!s) return s;
    if (first != 0) return status(Code::bounds, 0);
    size_t previous = 0, max_row = 0;
    for (size_t row = 0; row < input.n; ++row) {
        size_t next;
        s = read_sparse_offset(input, row + 1, next);
        if (!s) return s;
        if (next < previous || next > input.nnz)
            return status(Code::bounds, row + 1);
        if (input.layout == Layout::csr && next - previous > max_row)
            max_row = next - previous;
        previous = next;
    }
    if (previous != input.nnz) return status(Code::bounds, input.n);

    for (size_t p = 0; p < input.nnz; ++p) {
        size_t index;
        s = read_sparse_index(input, p, index);
        if (!s) return s;
        if (index >= input.n) return status(Code::bounds, p);
    }
    size_t bad;
    if (input.dtype == DType::float32) {
        if (!finite_sparse_values<float>(input, bad))
            return status(Code::nonfinite, bad);
    } else if (!finite_sparse_values<double>(input, bad)) {
        return status(Code::nonfinite, bad);
    }

    bool bound_ok;
    required.entries = sparse_entries_bound(input.n, input.nnz, bound_ok);
    if (!bound_ok) return status(Code::overflow);
    if (!add_size(input.n, 1, required.indices)) return status(Code::overflow);
    if (input.layout == Layout::csc &&
        !multiply_size(required.indices, 2, required.indices))
        return status(Code::overflow);
    required.scratch_entries = input.layout == Layout::csr ? max_row : input.nnz;
    result = required;
    return status(Code::ok);
}

bool entry_before(const Entry& a, const Entry& b) {
    return a.index < b.index || (a.index == b.index && a.position < b.position);
}

void swap_entry(Entry& a, Entry& b) {
    const Entry value = a;
    a = b;
    b = value;
}

void sift_down(Entry* entries, size_t root, size_t count) {
    while (root < count / 2) {
        size_t child = root * 2 + 1;
        if (child + 1 < count && entry_before(entries[child], entries[child + 1]))
            ++child;
        if (!entry_before(entries[root], entries[child])) return;
        swap_entry(entries[root], entries[child]);
        root = child;
    }
}

void sort_entries(Entry* entries, size_t count) {
    for (size_t start = count / 2; start > 0; --start)
        sift_down(entries, start - 1, count);
    for (size_t end = count; end > 1; --end) {
        swap_entry(entries[0], entries[end - 1]);
        sift_down(entries, 0, end - 1);
    }
}

template<class T, class Emit>
size_t grouped_row(const MatrixData& input, const Entry* entries, size_t count,
                   size_t row, Emit emit) {
    // Source-position order preserves duplicate summation before thresholding.
    size_t written = 0, p = 0;
    bool diagonal = false;
    while (p < count) {
        const size_t column = entries[p].index;
        T sum = T(0);
        do {
            sum = T(sum + load_value<T>(input.values + entries[p].position * sizeof(T)));
            ++p;
        } while (p < count && entries[p].index == column);

        if (column == row) {
            emit(row);
            ++written;
            diagonal = true;
        } else {
            if (column > row && !diagonal) {
                emit(row);
                ++written;
                diagonal = true;
            }
            if (sum > T(0)) {
                emit(column);
                ++written;
            }
        }
    }
    if (!diagonal) {
        emit(row);
        ++written;
    }
    return written;
}

template<class T> size_t dense_row_count(const MatrixData& input, size_t row) {
    size_t count = 0;
    for (size_t column = 0; column < input.n; ++column) {
        ptrdiff_t offset;
        dense_offset(input, row, column, offset);
        const T value = load_value<T>(input.values + offset);
        if (column == row || value > T(0)) ++count;
    }
    return count;
}

template<class T> void write_dense(const MatrixData& input,
                                   const NeighborhoodBuffers& buffers,
                                   size_t& output) {
    buffers.offsets[0] = 0;
    for (size_t row = 0; row < input.n; ++row) {
        const size_t count = dense_row_count<T>(input, row);
        const float weight = static_cast<float>(1.0 / static_cast<double>(count));
        for (size_t column = 0; column < input.n; ++column) {
            ptrdiff_t offset;
            dense_offset(input, row, column, offset);
            const T value = load_value<T>(input.values + offset);
            if (column == row || value > T(0)) {
                buffers.columns[output] = column;
                buffers.values[output] = weight;
                ++output;
            }
        }
        buffers.offsets[row + 1] = output;
    }
}

template<class T> void write_sparse_row(const MatrixData& input,
                                        const Entry* entries, size_t count,
                                        size_t row,
                                        const NeighborhoodBuffers& buffers,
                                        size_t& output) {
    const size_t written = grouped_row<T>(input, entries, count, row,
        [](size_t) {});
    const float weight = static_cast<float>(1.0 / static_cast<double>(written));
    grouped_row<T>(input, entries, count, row, [&](size_t column) {
        buffers.columns[output] = column;
        buffers.values[output] = weight;
        ++output;
    });
}

template<class T> void write_csr(const MatrixData& input,
                                 const NeighborhoodBuffers& buffers,
                                 size_t& output) {
    for (size_t i = 0; i <= input.n; ++i)
        buffers.indices[i] = sparse_offset_at(input, i);
    buffers.offsets[0] = 0;
    for (size_t row = 0; row < input.n; ++row) {
        const size_t first = buffers.indices[row];
        const size_t last = buffers.indices[row + 1];
        const size_t count = last - first;
        for (size_t p = 0; p < count; ++p)
            buffers.scratch[p] = {sparse_index_at(input, first + p), first + p};
        sort_entries(buffers.scratch, count);
        write_sparse_row<T>(input, buffers.scratch, count, row, buffers, output);
        buffers.offsets[row + 1] = output;
    }
}

template<class T> void write_csc(const MatrixData& input,
                                 const NeighborhoodBuffers& buffers,
                                 size_t& output) {
    const size_t offset_count = input.n + 1;
    size_t* row_offsets = buffers.indices;
    size_t* cursor = buffers.indices + offset_count;
    for (size_t i = 0; i <= input.n; ++i) row_offsets[i] = 0;
    for (size_t p = 0; p < input.nnz; ++p)
        ++row_offsets[sparse_index_at(input, p) + 1];
    for (size_t row = 1; row <= input.n; ++row)
        row_offsets[row] += row_offsets[row - 1];
    for (size_t row = 0; row <= input.n; ++row) cursor[row] = row_offsets[row];

    for (size_t column = 0; column < input.n; ++column) {
        const size_t first = sparse_offset_at(input, column);
        const size_t last = sparse_offset_at(input, column + 1);
        for (size_t p = first; p < last; ++p) {
            const size_t row = sparse_index_at(input, p);
            buffers.scratch[cursor[row]++] = {column, p};
        }
    }

    // The CSC scatter visits columns and source positions in ascending order.
    buffers.offsets[0] = 0;
    for (size_t row = 0; row < input.n; ++row) {
        const size_t first = row_offsets[row];
        const size_t count = row_offsets[row + 1] - first;
        write_sparse_row<T>(input, buffers.scratch ? buffers.scratch + first : nullptr,
                            count, row, buffers, output);
        buffers.offsets[row + 1] = output;
    }
}

} // namespace

Status neighborhood_size(const AcquiredMatrix& matrix, NeighborhoodSize& size) {
    MatrixData input{};
    Status s = acquire(matrix, input);
    if (!s) return s;
    NeighborhoodSize required{};
    s = analyze(input, required);
    if (!s) return s;
    size = required;
    return status(Code::ok);
}

Status prepare_neighborhood(const AcquiredMatrix& matrix,
                            const NeighborhoodBuffers& buffers, size_t& used) {
    MatrixData input{};
    Status s = acquire(matrix, input);
    if (!s) return s;
    NeighborhoodSize required{};
    s = analyze(input, required);
    if (!s) return s;

    if (buffers.index_capacity < required.indices ||
        (required.indices && !buffers.indices))
        return status(Code::capacity, 1, required.indices);
    if (buffers.scratch_capacity < required.scratch_entries ||
        (required.scratch_entries && !buffers.scratch))
        return status(Code::capacity, 2, required.scratch_entries);
    if (buffers.capacity < required.entries ||
        (required.entries && (!buffers.values || !buffers.columns)))
        return status(Code::capacity, 3, required.entries);
    size_t required_offsets;
    if (!add_size(input.n, 1, required_offsets)) return status(Code::overflow);
    if (buffers.offset_capacity < required_offsets || !buffers.offsets)
        return status(Code::capacity, 4, required_offsets);

    size_t output = 0;
    if (input.layout == Layout::dense) {
        if (input.dtype == DType::float32) write_dense<float>(input, buffers, output);
        else write_dense<double>(input, buffers, output);
    } else if (input.layout == Layout::csr) {
        if (input.dtype == DType::float32) write_csr<float>(input, buffers, output);
        else write_csr<double>(input, buffers, output);
    } else {
        if (input.dtype == DType::float32) write_csc<float>(input, buffers, output);
        else write_csc<double>(input, buffers, output);
    }
    used = output;
    return status(Code::ok);
}

} // namespace velocitypp::reference
