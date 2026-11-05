#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

import velocity;

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
    m.offsets.resize(rows + 1); m.indices.resize(nnz); m.values.resize(nnz);
    read(in, m.offsets.data(), rows + 1);
    read(in, m.indices.data(), nnz);
    read(in, m.values.data(), nnz);
    return m;
}
velocity::Trimming trim(const std::string& value) {
    if (value == "none") return velocity::Trimming::none;
    if (value == "upper") return velocity::Trimming::upper;
    if (value == "tails") return velocity::Trimming::tails;
    throw std::runtime_error("trimming must be none, upper, or tails");
}
}

// Input: five u64 dimensions/nnz, CSR S/U/W, then byte cell/gene masks.
// Output: Ms, Mu, Mss, Mus, V, residual2 (FP32 V/residual; FP64 otherwise),
// stochastic coefficients/scales/diagnostics, selected/refined/status bytes.
int main(int argc, char** argv) try {
    if (argc != 9) throw std::runtime_error(
        "usage: stochastic_interop_driver input masks output fit_offset fit_offset2 trimming min_r2 min_ratio");
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
    velocity::Input input{sa.view(), ua.view(), wa.view()};
    input.fit_mask = fa.view(); input.gene_mask = ga.view();
    velocity::PreparedInput prepared(input);

    velocity::StochasticOptions options;
    options.initialization.fit_intercept = std::stoi(argv[4]) != 0;
    options.fit_second_intercept = std::stoi(argv[5]) != 0;
    options.initialization.trimming = trim(argv[6]);
    options.initialization.min_r2 = std::stod(argv[7]);
    options.initialization.min_ratio = std::stod(argv[8]);
    velocity::StochasticRunOptions run;
    run.block_genes = 16;
    run.second_residual = true;
    velocity::StochasticWorkspace workspace(prepared, run);

    std::vector<double> ms(n*g), mu(n*g), mss(n*g), mus(n*g);
    std::vector<double> source_s(n), source_u(n);
    velocity::reference::second_moments(prepared, 0, g, ms.data(), mu.data(),
        mss.data(), mus.data(), nullptr, source_s.data(), source_u.data());
    auto v = anndatapp::Array::empty<float>({n, g});
    auto residual2 = anndatapp::Array::empty<float>({n, g});
    auto result = velocity::reference::stochastic_into(prepared, workspace, v.edit(),
        options, residual2.edit());

    std::ofstream out(argv[3], std::ios::binary);
    if (!out) throw std::runtime_error("cannot open output");
    write(out, ms.data(), ms.size()); write(out, mu.data(), mu.size());
    write(out, mss.data(), mss.size()); write(out, mus.data(), mus.size());
    write(out, v.view().dense<float>().data(), n*g);
    write(out, residual2.view().dense<float>().data(), n*g);
    for (const auto* array : {&result.gamma, &result.offset, &result.offset2,
                               &result.first_scale, &result.second_scale,
                               &result.initialization.r2, &result.objective})
        write(out, array->view().dense<double>().data(), g);
    write(out, result.initialization.gamma.view().dense<double>().data(), g);
    write(out, result.initialization.offset.view().dense<double>().data(), g);
    write(out, result.initialization.genes.view().dense<bool>().data(), g);
    std::vector<std::uint8_t> refined(g);
    const auto* status = result.status.view().dense<std::uint8_t>().data();
    for (std::size_t i = 0; i < g; ++i) refined[i] = status[i] == std::uint8_t(velocity::StochasticStatus::ok);
    write(out, refined.data(), g); write(out, status, g);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "stochastic interop: " << e.what() << '\n';
    return 1;
}
