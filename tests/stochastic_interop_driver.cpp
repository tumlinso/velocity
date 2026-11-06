#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
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
velocitypp::Trimming trim(const std::string& value) {
    if (value == "none") return velocitypp::Trimming::none;
    if (value == "upper") return velocitypp::Trimming::upper;
    if (value == "tails") return velocitypp::Trimming::tails;
    throw std::runtime_error("trimming must be none, upper, or tails");
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
        require(velocitypp::workspace_size<T>(input, options, true, size), "workspace_size failed");
        scalars.resize(size.scalars);
        masks.resize(size.masks);
        require(velocitypp::bind_workspace(input, options, true, scalars.data(), scalars.size(),
            masks.data(), masks.size(), value), "bind_workspace failed");
    }
};
}

// Input: five u64 dimensions/nnz, CSR S/U/W, then byte cell/gene masks.
// Output: Ms, Mu, Mss, Mus, V, residual2 (FP32 V/residual; FP64 otherwise),
// stochastic coefficients/scales/diagnostics, selected/refined/status bytes.
int main(int argc, char** argv) try {
    if (argc != 9) throw std::runtime_error(
        "usage: stochastic_interop_driver input masks output fit_offset fit_offset2 trimming min_r2 min_ratio");
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

    velocitypp::StochasticOptions options;
    options.initialization.fit_intercept = std::stoi(argv[4]) != 0;
    options.fit_second_intercept = std::stoi(argv[5]) != 0;
    options.initialization.trimming = trim(argv[6]);
    options.initialization.min_r2 = std::stod(argv[7]);
    options.initialization.min_ratio = std::stod(argv[8]);
    velocitypp::RunOptions run;
    run.block_genes = 16;
    WorkspaceStorage<double> workspace(prepared.value, run);

    std::vector<double> ms(n*g), mu(n*g), mss(n*g), mus(n*g);
    std::vector<double> source_s(n), source_u(n);
    std::vector<float> velocity(n*g), residual2(n*g);
    std::vector<double> initial_gamma(g), initial_offset(g), initial_r2(g);
    std::vector<double> gamma(g), offset(g), offset2(g), first_scale(g), second_scale(g), objective(g);
    std::vector<std::uint8_t> initial_status(g), status(g), refined(g);
    std::unique_ptr<bool[]> selected(new bool[g]{});
    velocitypp::StochasticFitResult result{};
    result.initialization = {initial_gamma.data(), initial_offset.data(), initial_r2.data(),
        selected.get(), initial_status.data(), g};
    result.gamma = gamma.data();
    result.offset = offset.data();
    result.offset2 = offset2.data();
    result.first_scale = first_scale.data();
    result.second_scale = second_scale.data();
    result.objective = objective.data();
    result.status = status.data();
    result.capacity = g;

    velocitypp::reference::second_moments<double>(prepared.value, 0, g, ms.data(), mu.data(),
        mss.data(), mus.data(), nullptr, source_s.data(), source_u.data());
    require(velocitypp::reference::stochastic_into<double, float>(prepared.value, workspace.value,
        result, velocity.data(), velocity.size(), residual2.data(), residual2.size(),
        std::ptrdiff_t(g), options), "stochastic_into failed");
    for (std::size_t i = 0; i < g; ++i)
        refined[i] = status[i] == std::uint8_t(velocitypp::StochasticStatus::ok);

    std::ofstream out(argv[3], std::ios::binary);
    if (!out) throw std::runtime_error("cannot open output");
    write(out, ms.data(), ms.size());
    write(out, mu.data(), mu.size());
    write(out, mss.data(), mss.size());
    write(out, mus.data(), mus.size());
    write(out, velocity.data(), velocity.size());
    write(out, residual2.data(), residual2.size());
    write(out, gamma.data(), gamma.size());
    write(out, offset.data(), offset.size());
    write(out, offset2.data(), offset2.size());
    write(out, first_scale.data(), first_scale.size());
    write(out, second_scale.data(), second_scale.size());
    write(out, initial_r2.data(), initial_r2.size());
    write(out, objective.data(), objective.size());
    write(out, initial_gamma.data(), initial_gamma.size());
    write(out, initial_offset.data(), initial_offset.size());
    write(out, selected.get(), g);
    write(out, refined.data(), refined.size());
    write(out, status.data(), status.size());
    return 0;
} catch (const std::exception& e) {
    std::cerr << "stochastic interop: " << e.what() << '\n';
    return 1;
}
