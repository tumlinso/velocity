#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

import velocitypp;

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
template<class T> void read(std::istream& in, T* data, std::size_t n) {
    in.read(reinterpret_cast<char*>(data), n * sizeof(T));
    if (!in) throw std::runtime_error("truncated input");
}
template<class T> void write(std::ostream& out, const T* data, std::size_t n) {
    out.write(reinterpret_cast<const char*>(data), n * sizeof(T));
    if (!out) throw std::runtime_error("output write failed");
}
Matrix matrix(std::istream& in, std::size_t rows, std::size_t nnz) {
    Matrix m;
    m.offsets.resize(rows+1); m.indices.resize(nnz); m.values.resize(nnz);
    read(in, m.offsets.data(), rows+1);
    read(in, m.indices.data(), nnz);
    read(in, m.values.data(), nnz);
    return m;
}
}

// Input: common 5-u64 header plus CSR S/U/W, then a separate file containing
// byte fit-observation mask followed by byte highly-variable-gene mask.
// Output: row-major FP64 Ms/Mu, FP32 V, FP64 gamma/offset/r2, u8 gene/status.
int main(int argc, char** argv) try {
    if (argc != 4) throw std::runtime_error("usage: interop_driver input masks output");
    std::ifstream in(argv[1], std::ios::binary), masks(argv[2], std::ios::binary);
    if (!in || !masks) throw std::runtime_error("cannot open input");
    std::uint64_t h[5]; read(in, h, 5);
    const auto [n, g, sn, un, wn] = h;
    if (!n || !g || n > 100000 || g > 100000 || sn > n*g || un > n*g || wn > n*n)
        throw std::runtime_error("invalid input dimensions");
    auto s = matrix(in, n, sn), u = matrix(in, n, un), w = matrix(in, n, wn);
    if (in.peek() != EOF) throw std::runtime_error("trailing input");
    std::vector<std::uint8_t> fit(n), genes(g);
    read(masks, fit.data(), n); read(masks, genes.data(), g);
    if (masks.peek() != EOF) throw std::runtime_error("trailing mask input");
    auto sa = s.array(n, g), ua = u.array(n, g), wa = w.array(n, n);
    auto fa = anndatapp::Array::borrow_dense(fit.data(), {n}, {1});
    auto ga = anndatapp::Array::borrow_dense(genes.data(), {g}, {1});
    velocitypp::Input input{sa.view(), ua.view(), wa.view()};
    input.fit_mask = fa.view(); input.gene_mask = ga.view();
    velocitypp::PreparedInput prepared(input);
    velocitypp::RunOptions run;
    run.block_genes = 16;
    velocitypp::Workspace workspace(prepared, run);
    std::vector<double> ms(n*g), mu(n*g);
    velocitypp::reference::moments(prepared, 0, g, ms.data(), mu.data());
    auto v = anndatapp::Array::empty<float>({n, g});
    auto result = velocitypp::reference::steady_state_into(prepared, workspace, v.edit());
    std::ofstream out(argv[3], std::ios::binary);
    if (!out) throw std::runtime_error("cannot open output");
    write(out, ms.data(), ms.size()); write(out, mu.data(), mu.size());
    write(out, v.view().dense<float>().data(), n*g);
    write(out, result.gamma.view().dense<double>().data(), g);
    write(out, result.offset.view().dense<double>().data(), g);
    write(out, result.r2.view().dense<double>().data(), g);
    write(out, result.genes.view().dense<bool>().data(), g);
    write(out, result.status.view().dense<std::uint8_t>().data(), g);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "interop: " << e.what() << '\n';
    return 1;
}
