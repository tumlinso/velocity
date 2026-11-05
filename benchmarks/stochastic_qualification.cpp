#include <algorithm>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <Cellerator/ops/numeric/binary16.hh>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

import velocity;

namespace {
constinit std::atomic<bool> count_allocations{false};
constinit std::atomic<std::uint64_t> allocation_calls{0}, allocation_bytes{0};

void* allocate(std::size_t bytes, std::size_t alignment) {
    void* p = nullptr;
    for (;;) {
        if (alignment <= alignof(std::max_align_t)) p = std::malloc(bytes ? bytes : 1);
        else {
            const auto size = bytes ? bytes : alignment;
            if (size > std::numeric_limits<std::size_t>::max() - (alignment - 1)) throw std::bad_alloc();
            const auto rounded = (size + alignment - 1) & ~(alignment - 1);
            p = std::aligned_alloc(alignment, rounded);
        }
        if (p) break;
        auto handler = std::get_new_handler();
        if (!handler) throw std::bad_alloc();
        handler();
    }
    if (count_allocations.load(std::memory_order_relaxed)) {
        allocation_calls.fetch_add(1, std::memory_order_relaxed);
        allocation_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    return p;
}
}

void* operator new(std::size_t n) { return allocate(n, alignof(std::max_align_t)); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) { return allocate(n, static_cast<std::size_t>(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try { return ::operator new(n); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try { return ::operator new[](n); } catch (...) { return nullptr; }
}
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return ::operator new(n, a); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return ::operator new[](n, a); } catch (...) { return nullptr; }
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t max_fixture_bytes = 512ull * 1024 * 1024;
constexpr std::size_t max_workspace_bytes = 64ull * 1024 * 1024;
constexpr std::uint64_t max_accounted_bytes = 1536ull * 1024 * 1024;

struct Csr {
    std::size_t rows = 0, columns = 0;
    std::vector<std::int64_t> offsets;
    std::vector<std::int32_t> indices;
    std::vector<float> values;

    std::size_t bytes() const {
        return offsets.capacity()*sizeof(std::int64_t)
            + indices.capacity()*sizeof(std::int32_t) + values.capacity()*sizeof(float);
    }
};

struct Fixture {
    std::shared_ptr<Csr> spliced, unspliced, weights;
    std::uint64_t file_bytes = 0;
    std::uint64_t payload_bytes = 0;
};

std::size_t checked_size(std::uint64_t x, const char* what) {
    if (x > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error(std::string(what) + " exceeds host size_t");
    return static_cast<std::size_t>(x);
}

std::uint64_t add_bytes(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::runtime_error("fixture size overflows uint64");
    return a + b;
}

std::uint64_t mul_bytes(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a)
        throw std::runtime_error("fixture size overflows uint64");
    return a*b;
}

void read_exact(std::ifstream& f, void* dst, std::size_t bytes) {
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("fixture section exceeds stream limits");
    if (bytes && !f.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("truncated fixture");
}

void validate_csr(const Csr& x, bool normalized_rows) {
    if (!x.rows || !x.columns || x.offsets.size() != x.rows + 1
        || x.indices.size() != x.values.size() || x.offsets.front() != 0
        || x.offsets.back() != static_cast<std::int64_t>(x.values.size()))
        throw std::runtime_error("invalid CSR dimensions or offsets");
    for (std::size_t i = 0; i < x.rows; ++i) {
        if (x.offsets[i] < 0 || x.offsets[i] > x.offsets[i + 1]
            || x.offsets[i + 1] > static_cast<std::int64_t>(x.values.size()))
            throw std::runtime_error("CSR offsets are not monotone");
        std::int32_t previous = -1;
        double row_sum = 0;
        for (auto p = static_cast<std::size_t>(x.offsets[i]); p < static_cast<std::size_t>(x.offsets[i + 1]); ++p) {
            const auto j = x.indices[p];
            const float v = x.values[p];
            if (j < 0 || static_cast<std::size_t>(j) >= x.columns || j <= previous
                || !std::isfinite(v) || v <= 0)
                throw std::runtime_error("CSR entries must be finite, positive, sorted, and unique");
            previous = j;
            row_sum += v;
        }
        if (normalized_rows && (x.offsets[i] == x.offsets[i + 1] || std::abs(row_sum - 1) > 2e-5))
            throw std::runtime_error("graph rows must be nonempty and normalized");
    }
}

std::shared_ptr<Csr> read_matrix(std::ifstream& f, std::size_t rows, std::size_t columns,
                                 std::uint64_t nnz64, bool normalized_rows) {
    auto x = std::make_shared<Csr>();
    x->rows = rows;
    x->columns = columns;
    const auto nnz = checked_size(nnz64, "nnz");
    if (rows == std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("row count overflows offsets");
    x->offsets.resize(rows + 1);
    x->indices.resize(nnz);
    x->values.resize(nnz);
    read_exact(f, x->offsets.data(), x->offsets.size()*sizeof(std::int64_t));
    read_exact(f, x->indices.data(), x->indices.size()*sizeof(std::int32_t));
    read_exact(f, x->values.data(), x->values.size()*sizeof(float));
    validate_csr(*x, normalized_rows);
    return x;
}

Fixture load_fixture(const std::filesystem::path& path) {
    if constexpr (std::endian::native != std::endian::little)
        throw std::runtime_error("fixture reader currently requires a little-endian host");
    if (!std::numeric_limits<float>::is_iec559 || sizeof(float) != 4)
        throw std::runtime_error("fixture requires IEEE-754 binary32");
    std::error_code ec;
    const auto file_bytes = std::filesystem::file_size(path, ec);
    if (ec) throw std::runtime_error("cannot stat input fixture: " + ec.message());
    if (file_bytes < 40 || file_bytes > max_fixture_bytes)
        throw std::runtime_error("fixture size is outside the 40-byte to 512-MiB bound");
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open input fixture");
    std::uint64_t h[5]{};
    read_exact(f, h, sizeof h);
    const auto n = checked_size(h[0], "N"), g = checked_size(h[1], "G");
    if (!n || !g || n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())
        || g > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error("fixture dimensions are outside int32 sparse-index bounds");
    std::uint64_t expected = 40;
    const auto ptr_bytes = mul_bytes(add_bytes(n, 1), sizeof(std::int64_t));
    for (auto nnz : {h[2], h[3], h[4]})
        expected = add_bytes(expected, add_bytes(ptr_bytes,
            mul_bytes(nnz, sizeof(std::int32_t) + sizeof(float))));
    if (expected != file_bytes) throw std::runtime_error("fixture byte count does not match its header");
    Fixture out;
    out.file_bytes = file_bytes;
    out.payload_bytes = file_bytes - 40;
    out.spliced = read_matrix(f, n, g, h[2], false);
    out.unspliced = read_matrix(f, n, g, h[3], false);
    out.weights = read_matrix(f, n, n, h[4], true);
    if (f.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing fixture bytes");
    return out;
}

std::shared_ptr<Csr> to_csc(const std::shared_ptr<Csr>& source) {
    auto out = std::make_shared<Csr>();
    out->rows = source->rows;
    out->columns = source->columns;
    out->offsets.assign(out->columns + 1, 0);
    for (auto i : source->indices) ++out->offsets[static_cast<std::size_t>(i) + 1];
    for (std::size_t j = 1; j < out->offsets.size(); ++j) out->offsets[j] += out->offsets[j - 1];
    out->indices.resize(source->indices.size());
    out->values.resize(source->values.size());
    auto cursor = out->offsets;
    for (std::size_t i = 0; i < source->rows; ++i)
        for (auto p = static_cast<std::size_t>(source->offsets[i]); p < static_cast<std::size_t>(source->offsets[i + 1]); ++p) {
            const auto j = static_cast<std::size_t>(source->indices[p]);
            const auto q = static_cast<std::size_t>(cursor[j]++);
            out->indices[q] = static_cast<std::int32_t>(i);
            out->values[q] = source->values[p];
        }
    return out;
}

struct Dense {
    std::size_t rows = 0, columns = 0;
    std::vector<float> values;
    std::size_t bytes() const { return values.capacity()*sizeof(float); }
};

std::shared_ptr<Dense> to_dense(const Csr& source) {
    const auto count = mul_bytes(source.rows, source.columns);
    const auto bytes = mul_bytes(count, sizeof(float));
    if (bytes > max_accounted_bytes) throw std::runtime_error("dense control exceeds memory budget");
    auto out = std::make_shared<Dense>();
    out->rows = source.rows;
    out->columns = source.columns;
    out->values.assign(checked_size(count, "dense element count"), 0.0f);
    for (std::size_t i = 0; i < source.rows; ++i)
        for (auto p = static_cast<std::size_t>(source.offsets[i]); p < static_cast<std::size_t>(source.offsets[i + 1]); ++p)
            out->values[i*source.columns + static_cast<std::size_t>(source.indices[p])] = source.values[p];
    return out;
}

std::uint64_t sparse_storage_bytes(std::uint64_t nnz, std::uint64_t major) {
    return add_bytes(mul_bytes(add_bytes(major, 1), sizeof(std::int64_t)),
                     mul_bytes(nnz, sizeof(std::int32_t) + sizeof(float)));
}

enum class Format { csr, csc, dense };

void check_probe_bound(const Fixture& f, Format expression, Format graph, bool retain,
                       std::size_t n, std::size_t g) {
    const auto expr_bytes = expression == Format::dense
        ? mul_bytes(mul_bytes(mul_bytes(n, g), sizeof(float)), 2)
        : add_bytes(sparse_storage_bytes(f.spliced->values.size(), expression == Format::csc ? g : n),
                    sparse_storage_bytes(f.unspliced->values.size(), expression == Format::csc ? g : n));
    const auto graph_bytes = sparse_storage_bytes(f.weights->values.size(), n);
    const auto selected_bytes = add_bytes(expr_bytes, graph_bytes);
    const auto entry_bytes = 2*sizeof(std::size_t);
    auto prepared_bytes = mul_bytes(mul_bytes(f.weights->values.size(), 2), entry_bytes);
    if (expression == Format::csr)
        prepared_bytes = add_bytes(prepared_bytes,
            mul_bytes(add_bytes(f.spliced->values.size(), f.unspliced->values.size()), entry_bytes));
    prepared_bytes = add_bytes(prepared_bytes, mul_bytes(mul_bytes(add_bytes(n, 1), 3), sizeof(std::size_t)));
    prepared_bytes = add_bytes(prepared_bytes, mul_bytes(g, sizeof(std::size_t)));
    if (expression != Format::dense)
        prepared_bytes = add_bytes(prepared_bytes, mul_bytes(mul_bytes(add_bytes(g, 1), 2), sizeof(std::size_t)));
    const auto output_bytes = retain ? mul_bytes(mul_bytes(n, g), sizeof(float)) : 0;
    auto estimate = mul_bytes(f.file_bytes, 2);
    estimate = add_bytes(estimate, selected_bytes);
    estimate = add_bytes(estimate, prepared_bytes);
    estimate = add_bytes(estimate, max_workspace_bytes);
    estimate = add_bytes(estimate, output_bytes);
    estimate = add_bytes(estimate, 128ull*1024*1024);
    if (estimate > max_accounted_bytes)
        throw std::runtime_error("fixture and estimated live buffers exceed the 1.5-GiB probe bound");
}

Format parse_expression_format(std::string_view s) {
    if (s == "csr") return Format::csr;
    if (s == "csc") return Format::csc;
    if (s == "dense") return Format::dense;
    throw std::runtime_error("expression format must be csr, csc, or dense");
}
Format parse_graph_format(std::string_view s) {
    if (s == "csr") return Format::csr;
    if (s == "csc") return Format::csc;
    throw std::runtime_error("graph format must be csr or csc");
}
std::string_view name(Format x) {
    return x == Format::csr ? "csr" : x == Format::csc ? "csc" : "dense";
}

struct OwnedArray {
    anndatapp::Array array;
    std::size_t bytes = 0;
};

OwnedArray make_array(const std::shared_ptr<Csr>& source, Format format) {
    if (format == Format::dense) {
        auto dense = to_dense(*source);
        auto owner = std::static_pointer_cast<const void>(dense);
        auto array = anndatapp::Array::share_dense<float>(dense->values.data(),
            {dense->rows, dense->columns},
            {static_cast<std::ptrdiff_t>(dense->columns), 1}, std::move(owner), anndatapp::Validation::none);
        return {std::move(array), dense->bytes()};
    }
    auto storage = format == Format::csr ? source : to_csc(source);
    const auto layout = format == Format::csr ? anndatapp::Layout::csr : anndatapp::Layout::csc;
    auto owner = std::static_pointer_cast<const void>(storage);
    auto array = anndatapp::Array::share_sparse<float, std::int32_t, std::int64_t>(layout,
        storage->rows, storage->columns, storage->values.size(),
        storage->values.data(), storage->indices.data(), storage->offsets.data(), std::move(owner), anndatapp::Validation::none);
    return {std::move(array), storage->bytes()};
}

std::size_t scalar_bytes(anndatapp::DType type) {
    switch (type) {
    case anndatapp::DType::boolean: case anndatapp::DType::int8: case anndatapp::DType::uint8: return 1;
    case anndatapp::DType::int16: case anndatapp::DType::uint16: return 2;
    case anndatapp::DType::int32: case anndatapp::DType::uint32: case anndatapp::DType::float32: return 4;
    case anndatapp::DType::int64: case anndatapp::DType::uint64: case anndatapp::DType::float64: return 8;
    }
    throw std::runtime_error("unsupported output dtype");
}

std::size_t array_bytes(const anndatapp::Array& a) {
    std::size_t count = 1;
    for (std::size_t i = 0; i < a.rank(); ++i) {
        if (a.extent(i) && count > std::numeric_limits<std::size_t>::max()/a.extent(i))
            throw std::runtime_error("array extent product overflows");
        count *= a.extent(i);
    }
    return count*scalar_bytes(a.dtype());
}

std::size_t fit_result_bytes(const velocity::StochasticFitResult& fit) {
    const auto& init = fit.initialization;
    return array_bytes(init.gamma) + array_bytes(init.offset) + array_bytes(init.r2)
        + array_bytes(init.genes) + array_bytes(init.status) + array_bytes(fit.gamma)
        + array_bytes(fit.offset) + array_bytes(fit.offset2) + array_bytes(fit.first_scale)
        + array_bytes(fit.second_scale) + array_bytes(fit.status) + array_bytes(fit.objective);
}

struct Checksum {
    std::uint64_t values = 0;
    double sum = 0, absolute_sum = 0;
    bool finite = true;

    template<class T> void add(T x) {
        const double d = static_cast<double>(x);
        sum += d;
        absolute_sum += std::abs(d);
        finite = finite && std::isfinite(d);
        ++values;
    }
};

void checksum_add(Checksum& checksum, std::uint16_t bits) {
    checksum.add(cellerator::numeric::load_f16(bits));
}
template<class T> void checksum_add(Checksum& checksum, T value) { checksum.add(value); }

template<class T>
void checksum_values(Checksum& checksum, const T* data, std::size_t rows, std::size_t columns,
                     std::ptrdiff_t row_stride, std::ptrdiff_t column_stride) {
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < columns; ++j)
            checksum_add(checksum, data[std::ptrdiff_t(i)*row_stride + std::ptrdiff_t(j)*column_stride]);
}

void checksum_block(Checksum& checksum, const velocity::NumericView& output,
                    std::size_t count, std::size_t rows) {
    if (output.layout() != anndatapp::Layout::dense || output.rank() != 2
        || output.extent(0) != rows || output.extent(1) != count)
        throw std::runtime_error("velocity sink received an unexpected output view");
    if (output.dtype() == velocity::StorageType::float16) {
        const auto view = output.dense<std::uint16_t>();
        checksum_values(checksum, view.data(), rows, count, view.stride(0), view.stride(1));
    } else if (output.dtype() == velocity::StorageType::float32) {
        const auto view = output.dense<float>();
        checksum_values(checksum, view.data(), rows, count, view.stride(0), view.stride(1));
    } else if (output.dtype() == velocity::StorageType::float64) {
        const auto view = output.dense<double>();
        checksum_values(checksum, view.data(), rows, count, view.stride(0), view.stride(1));
    } else throw std::runtime_error("velocity sink dtype must be float16, float32, or float64");
}

struct SinkState {
    Checksum velocity, second_residual;
    std::size_t next_gene = 0, rows = 0;
};

void stream_sink(void* context, const velocity::StochasticOutputBlock& block) {
    auto& state = *static_cast<SinkState*>(context);
    if (block.first != state.next_gene || !block.count)
        throw std::runtime_error("velocity sink blocks are not contiguous");
    checksum_block(state.velocity, block.velocity, block.count, state.rows);
    checksum_block(state.second_residual, block.second_residual, block.count, state.rows);
    state.next_gene += block.count;
}

void checksum_retained(Checksum& checksum, const velocity::NumericView& output,
                       std::size_t rows, std::size_t genes, std::size_t block_genes) {
    if (output.layout() != anndatapp::Layout::dense || output.rank() != 2
        || output.extent(0) != rows || output.extent(1) != genes)
        throw std::runtime_error("retained velocity output has an unexpected shape");
    if (output.dtype() == velocity::StorageType::float16) {
        const auto view = output.dense<std::uint16_t>();
        for (std::size_t first = 0; first < genes; first += block_genes)
            checksum_values(checksum, view.data() + std::ptrdiff_t(first)*view.stride(1), rows,
                std::min(block_genes, genes - first), view.stride(0), view.stride(1));
    } else if (output.dtype() == velocity::StorageType::float32) {
        const auto view = output.dense<float>();
        for (std::size_t first = 0; first < genes; first += block_genes)
            checksum_values(checksum, view.data() + std::ptrdiff_t(first)*view.stride(1), rows,
                std::min(block_genes, genes - first), view.stride(0), view.stride(1));
    } else if (output.dtype() == velocity::StorageType::float64) {
        const auto view = output.dense<double>();
        for (std::size_t first = 0; first < genes; first += block_genes)
            checksum_values(checksum, view.data() + std::ptrdiff_t(first)*view.stride(1), rows,
                std::min(block_genes, genes - first), view.stride(0), view.stride(1));
    } else throw std::runtime_error("retained velocity dtype must be float16, float32, or float64");
}

struct Memory { std::uint64_t rss = 0, hwm = 0; };
Memory memory() {
    std::ifstream f("/proc/self/status");
    Memory out;
    std::string line;
    while (std::getline(f, line)) {
        auto read_kib = [&line](std::string_view key) {
            if (line.compare(0, key.size(), key) != 0) return std::uint64_t{};
            const char* first = line.data() + key.size();
            while (first != line.data() + line.size() && (*first == ' ' || *first == '\t')) ++first;
            std::uint64_t kib = 0;
            const auto [last, error] = std::from_chars(first, line.data() + line.size(), kib);
            return error == std::errc{} && last != first ? kib*1024 : std::uint64_t{};
        };
        if (line.starts_with("VmRSS:")) out.rss = read_kib("VmRSS:");
        else if (line.starts_with("VmHWM:")) out.hwm = read_kib("VmHWM:");
    }
    return out;
}

double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

struct AllocationStats { std::uint64_t calls = 0, bytes = 0; };
AllocationStats allocation_stats() {
    return {allocation_calls.load(std::memory_order_relaxed), allocation_bytes.load(std::memory_order_relaxed)};
}

std::string json_escape(std::string_view s) {
    std::string out;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(static_cast<char>(c)); }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) out += '?';
        else out.push_back(static_cast<char>(c));
    }
    return out;
}

struct Config { std::filesystem::path input; std::size_t block = 64; };
void help() { std::cout << "usage: velocity-stochastic-qualification --input FILE [--block 1..64]\n"; }
std::optional<Config> parse_args(int argc, char** argv) {
    Config c; bool have_input = false;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") { help(); return std::nullopt; }
        if (++i == argc) throw std::runtime_error("option requires a value");
        std::string_view value = argv[i];
        if (arg == "--input") { c.input = value; have_input = true; }
        else if (arg == "--block") {
            auto [end, error] = std::from_chars(value.data(), value.data()+value.size(), c.block);
            if (error != std::errc{} || end != value.data()+value.size()) throw std::runtime_error("invalid block size");
        } else throw std::runtime_error("unknown option: " + std::string(arg));
    }
    if (!have_input || !c.block || c.block > 64) throw std::runtime_error("input and block [1,64] required");
    return c;
}

std::pair<double, bool> sum_array(const anndatapp::Array& a) {
    double sum = 0; bool finite = true;
    if (a.dtype() == anndatapp::DType::float64) {
        auto x = a.view().dense<double>();
        for (std::size_t i = 0; i < a.extent(0); ++i) { sum += x.data()[std::ptrdiff_t(i)*x.stride(0)]; finite &= std::isfinite(x.data()[std::ptrdiff_t(i)*x.stride(0)]); }
    } else throw std::runtime_error("expected float64 fit output");
    return {sum, finite};
}

int run(const Config& config) {
    const auto total_start = Clock::now();
    const auto before = memory();
    const auto load_start = Clock::now();
    auto fixture = load_fixture(config.input);
    const auto load_end = Clock::now();
    const auto loaded = memory();
    const auto n = fixture.spliced->rows, g = fixture.spliced->columns;
    const auto prep_start = load_end;
    const auto prep_alloc_start = allocation_stats();
    count_allocations.store(true, std::memory_order_relaxed);
    auto s = make_array(fixture.spliced, Format::csr);
    auto u = make_array(fixture.unspliced, Format::csr);
    auto w = make_array(fixture.weights, Format::csr);
    fixture.spliced.reset(); fixture.unspliced.reset(); fixture.weights.reset();
    velocity::Input input{s.array.view(), u.array.view(), w.array.view()};
    velocity::PreparedInput prepared(std::move(input), anndatapp::Validation::none);
    velocity::StochasticRunOptions run;
    run.workspace_bytes = max_workspace_bytes; run.block_genes = config.block;
    run.output_dtype = velocity::StorageType::float32; run.second_residual = true;
    velocity::StochasticWorkspace workspace(prepared, run);
    const auto input_bytes = s.bytes + u.bytes + w.bytes;
    const auto prep_accounted = input_bytes + prepared.storage_bytes();
    const auto mem_prepared = memory();
    const auto prep_end = Clock::now();
    count_allocations.store(false, std::memory_order_relaxed);
    const auto prep_alloc_end = allocation_stats();
    const AllocationStats prep_alloc{prep_alloc_end.calls-prep_alloc_start.calls, prep_alloc_end.bytes-prep_alloc_start.bytes};

    velocity::StochasticOptions options;
    options.initialization.min_r2 = -100; options.initialization.min_ratio = -100;
    velocity::StochasticFitResult result;
    SinkState sink{{}, {}, 0, n};
    const auto exec_alloc_start = allocation_stats();
    count_allocations.store(true, std::memory_order_relaxed);
    const auto execute_start = Clock::now();
    result = velocity::reference::stochastic(prepared, workspace, options, {&sink, stream_sink});
    const auto execute_end = Clock::now();
    count_allocations.store(false, std::memory_order_relaxed);
    const auto exec_alloc_end = allocation_stats();
    const AllocationStats exec_alloc{exec_alloc_end.calls-exec_alloc_start.calls, exec_alloc_end.bytes-exec_alloc_start.bytes};
    const auto after = memory();
    const auto fit_bytes = fit_result_bytes(result);
    if (sink.next_gene != g || sink.velocity.values != n*g || sink.second_residual.values != n*g
        || !sink.velocity.finite || !sink.second_residual.finite)
        throw std::runtime_error("stochastic sink missed, duplicated, or found nonfinite output");

    auto mask = result.initialization.genes.view().dense<bool>();
    auto status = result.status.view().dense<std::uint8_t>();
    auto gamma = result.gamma.view().dense<double>();
    auto offset = result.offset.view().dense<double>();
    auto offset2 = result.offset2.view().dense<double>();
    auto first_scale = result.first_scale.view().dense<double>();
    auto second_scale = result.second_scale.view().dense<double>();
    auto objective = result.objective.view().dense<double>();
    std::size_t selected = 0, successful = 0;
    std::string bits; bits.reserve(g);
    double gamma_sum = 0, offset_sum = 0, offset2_sum = 0;
    for (std::size_t j = 0; j < g; ++j) {
        const bool active = mask.data()[std::ptrdiff_t(j)*mask.stride(0)];
        selected += active; successful += active && status.data()[std::ptrdiff_t(j)*status.stride(0)] == 0;
        bits.push_back(active ? '1' : '0');
        gamma_sum += gamma.data()[std::ptrdiff_t(j)*gamma.stride(0)];
        offset_sum += offset.data()[std::ptrdiff_t(j)*offset.stride(0)];
        offset2_sum += offset2.data()[std::ptrdiff_t(j)*offset2.stride(0)];
        if (active && (!std::isfinite(first_scale.data()[std::ptrdiff_t(j)*first_scale.stride(0)])
            || !std::isfinite(second_scale.data()[std::ptrdiff_t(j)*second_scale.stride(0)])
            || !std::isfinite(objective.data()[std::ptrdiff_t(j)*objective.stride(0)])))
            throw std::runtime_error("selected stochastic diagnostics are nonfinite");
    }
    std::size_t success_counts[7]{};
    for (std::size_t j=0;j<g;++j) if (status.data()[std::ptrdiff_t(j)*status.stride(0)] < 7) ++success_counts[status.data()[std::ptrdiff_t(j)*status.stride(0)]];
    const auto output = mul_bytes(mul_bytes(n, g), sizeof(float));
    const auto streamed_live = mul_bytes(mul_bytes(n, std::min(g, config.block)), 2*sizeof(float));
    std::cout << std::setprecision(17)
        << "{\"schema\":\"velocity.stochastic-qualification.v1\",\"input\":\"" << json_escape(config.input.string())
        << "\",\"input_file_bytes\":" << fixture.file_bytes << ",\"shape\":[" << n << ',' << g
        << "],\"block_genes\":" << workspace.block_genes() << ",\"threads\":1,\"input_storage_bytes\":" << input_bytes
        << ",\"prepared_bytes\":" << prepared.storage_bytes() << ",\"workspace_scratch_bytes\":" << workspace.scratch_bytes()
        << ",\"prep_global_new_calls\":" << prep_alloc.calls << ",\"prep_global_new_requested_bytes\":" << prep_alloc.bytes
        << ",\"execution_global_new_calls\":" << exec_alloc.calls << ",\"execution_global_new_requested_bytes\":" << exec_alloc.bytes
        << ",\"fit_result_bytes\":" << fit_bytes << ",\"logical_velocity_bytes\":" << output
        << ",\"logical_second_residual_bytes\":" << output << ",\"stream_block_bytes\":" << streamed_live
        << ",\"selected_genes\":" << selected << ",\"successful_refinements\":" << successful
        << ",\"status_counts\":[" << success_counts[0] << ',' << success_counts[1] << ',' << success_counts[2] << ','
        << success_counts[3] << ',' << success_counts[4] << ',' << success_counts[5] << ',' << success_counts[6] << ']'
        << ",\"selected_mask_bits\":\"" << bits << "\",\"velocity_checksum\":" << sink.velocity.sum
        << ",\"velocity_absolute_checksum\":" << sink.velocity.absolute_sum
        << ",\"second_residual_checksum\":" << sink.second_residual.sum
        << ",\"second_residual_absolute_checksum\":" << sink.second_residual.absolute_sum
        << ",\"gamma_checksum\":" << gamma_sum
        << ",\"offset_checksum\":" << offset_sum << ",\"offset2_checksum\":" << offset2_sum
        << ",\"finite\":true,\"load_s\":" << seconds(load_start, load_end) << ",\"prepare_s\":" << seconds(prep_start, prep_end)
        << ",\"execute_sink_s\":" << seconds(execute_start, execute_end) << ",\"cold_total_s\":" << seconds(total_start, execute_end)
        << ",\"rss_before_bytes\":" << before.rss << ",\"rss_loaded_bytes\":" << loaded.rss
        << ",\"rss_prepared_bytes\":" << mem_prepared.rss << ",\"process_peak_bytes\":" << after.hwm
        << ",\"incremental_peak_including_preparation_bytes\":" << (after.hwm>loaded.rss?after.hwm-loaded.rss:0) << "}\n";
    return 0;
}

}

int main(int argc, char** argv) {
    try {
        auto config = parse_args(argc, argv);
        return config ? run(*config) : 0;
    } catch (const std::exception& e) {
        std::cerr << "velocity-qualification: " << e.what() << '\n';
        return 2;
    }
}
