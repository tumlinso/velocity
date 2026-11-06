#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

import velocitypp;
#include "support.hh"

namespace {
struct Matrix {
    std::vector<std::int64_t> offsets;
    std::vector<std::int32_t> indices;
    std::vector<float> values;
    anndatapp::Array array(std::size_t rows, std::size_t columns) {
        return anndatapp::Array::borrow_sparse(anndatapp::Layout::csr, rows, columns,
            values.size(), values.data(), indices.data(), offsets.data());
    }
};

struct InputOwner {
    Acquired spliced, unspliced, weights;
    velocitypp::Input input;
    InputOwner(anndatapp::ArrayView s, anndatapp::ArrayView u, anndatapp::ArrayView w)
        : spliced(s), unspliced(u), weights(w),
          input{spliced.matrix, unspliced.matrix, weights.matrix} {}
};

void require(velocitypp::Status status, const char* phase) {
    if (!status) throw std::runtime_error(phase);
}

template<class T> void read(std::istream& in, T* data, std::size_t n) {
    in.read(reinterpret_cast<char*>(data), std::streamsize(n * sizeof(T)));
    if (!in) throw std::runtime_error("truncated input");
}
template<class T> void write(std::ostream& out, const T* data, std::size_t n) {
    out.write(reinterpret_cast<const char*>(data), std::streamsize(n * sizeof(T)));
    if (!out) throw std::runtime_error("output write failed");
}
Matrix matrix(std::istream& in, std::size_t rows, std::size_t nnz) {
    Matrix m;
    m.offsets.resize(rows + 1);
    m.indices.resize(nnz);
    m.values.resize(nnz);
    read(in, m.offsets.data(), rows + 1);
    read(in, m.indices.data(), nnz);
    read(in, m.values.data(), nnz);
    return m;
}

struct PreparedStorage {
    std::vector<std::size_t> indices;
    std::vector<velocitypp::Entry> entries;
    velocitypp::PreparedInput value;
    explicit PreparedStorage(const velocitypp::Input& input) {
        velocitypp::PreparationSize size;
        require(velocitypp::preparation_size(input, size), "preparation_size failed");
        indices.resize(size.indices);
        entries.resize(size.entries);
        velocitypp::PreparationBuffers buffers{
            indices.data(), indices.size(), entries.data(), entries.size()};
        require(velocitypp::prepare(input, buffers, value), "prepare failed");
    }
};

template<class T> struct WorkspaceStorage {
    std::vector<T> scalars;
    std::vector<std::uint8_t> masks;
    velocitypp::Workspace<T> value;
    WorkspaceStorage(const velocitypp::PreparedInput& input, velocitypp::RunOptions options) {
        velocitypp::WorkspaceSize size;
        require(velocitypp::workspace_size<T>(input, options, false, size), "workspace_size failed");
        scalars.resize(size.scalars);
        masks.resize(size.masks);
        require(velocitypp::bind_workspace(input, options, false, scalars.data(), scalars.size(),
            masks.data(), masks.size(), value), "bind_workspace failed");
    }
};
}

// Input: common 5-u64 header plus CSR S/U/W, then a separate file containing
// byte fit-observation mask followed by byte highly-variable-gene mask.
// Output: row-major FP64 Ms/Mu, FP32 V, FP64 gamma/offset/r2, u8 gene/status.
int main(int argc, char** argv) try {
    if (argc != 4) throw std::runtime_error("usage: interop_driver input masks output");
    std::ifstream in(argv[1], std::ios::binary), masks(argv[2], std::ios::binary);
    if (!in || !masks) throw std::runtime_error("cannot open input");
    std::uint64_t h[5];
    read(in, h, 5);
    const auto [n, g, sn, un, wn] = h;
    if (!n || !g || n > 100000 || g > 100000 || sn > n*g || un > n*g || wn > n*n)
        throw std::runtime_error("invalid input dimensions");
    auto s = matrix(in, n, sn), u = matrix(in, n, un), w = matrix(in, n, wn);
    if (in.peek() != EOF) throw std::runtime_error("trailing input");
    std::vector<std::uint8_t> fit(n), genes(g);
    read(masks, fit.data(), n);
    read(masks, genes.data(), g);
    if (masks.peek() != EOF) throw std::runtime_error("trailing mask input");

    auto sa = s.array(n, g), ua = u.array(n, g), wa = w.array(n, n);
    InputOwner owner(sa.view(), ua.view(), wa.view());
    owner.input.fit_mask = fit.data();
    owner.input.gene_mask = genes.data();
    PreparedStorage prepared(owner.input);

    velocitypp::RunOptions options;
    options.block_genes = 16;
    WorkspaceStorage<double> workspace(prepared.value, options);

    std::vector<double> ms(n*g), mu(n*g);
    std::vector<float> velocity(n*g);
    std::vector<double> gamma(g), offset(g), r2(g);
    std::vector<std::uint8_t> status(g);
    std::unique_ptr<bool[]> selected(new bool[g]{});
    velocitypp::FitResult result{
        gamma.data(), offset.data(), r2.data(), selected.get(), status.data(), g};
    velocitypp::reference::moments(prepared.value, 0, g, ms.data(), mu.data());
    require(velocitypp::reference::steady_state_into<double, float>(prepared.value, workspace.value,
        result, velocity.data(), velocity.size(), std::ptrdiff_t(g)), "steady_state_into failed");
    std::ofstream out(argv[3], std::ios::binary);
    if (!out) throw std::runtime_error("cannot open output");
    write(out, ms.data(), ms.size());
    write(out, mu.data(), mu.size());
    write(out, velocity.data(), velocity.size());
    write(out, gamma.data(), gamma.size());
    write(out, offset.data(), offset.size());
    write(out, r2.data(), r2.size());
    write(out, selected.get(), g);
    write(out, status.data(), status.size());
    return 0;
} catch (const std::exception& e) {
    std::cerr << "interop: " << e.what() << '\n';
    return 1;
}
