#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

import velocitypp;
#include "support.hh"
#include "precision_test_support.hh"

namespace {
enum class Storage : std::uint8_t { f32 = 2, f64 = 3 };
enum class Method : std::uint8_t { deterministic = 1, stochastic = 2 };

struct InputData {
    std::size_t n = 0, g = 0;
    struct Matrix {
        Storage type{};
        std::vector<float> f32;
        std::vector<double> f64;
        anndatapp::Array array(std::size_t rows, std::size_t columns) {
            const auto stride = std::ptrdiff_t(columns);
            if (type == Storage::f32) {
                return anndatapp::Array::borrow_dense(f32.data(), {rows, columns}, {stride, 1});
            }
            return anndatapp::Array::borrow_dense(f64.data(), {rows, columns}, {stride, 1});
        }
    } s, u, w;
    std::vector<std::uint8_t> fit, genes;
};

template<class T> void read(std::istream& in, T* p, std::size_t n) {
    in.read(reinterpret_cast<char*>(p), std::streamsize(n * sizeof(T)));
    if (!in) throw std::runtime_error("truncated precision input");
}
template<class T> void write(std::ostream& out, const T* p, std::size_t n) {
    out.write(reinterpret_cast<const char*>(p), std::streamsize(n * sizeof(T)));
    if (!out) throw std::runtime_error("precision output write failed");
}

Storage decode_storage(std::uint8_t tag) {
    if (tag == 1) throw std::runtime_error("FP16 input storage is unsupported");
    if (tag == static_cast<std::uint8_t>(Storage::f32)) return Storage::f32;
    if (tag == static_cast<std::uint8_t>(Storage::f64)) return Storage::f64;
    throw std::runtime_error("unsupported input storage dtype");
}

InputData::Matrix read_matrix(std::istream& in, Storage type, std::size_t count) {
    InputData::Matrix matrix;
    matrix.type = type;
    if (type == Storage::f32) {
        matrix.f32.resize(count);
        read(in, matrix.f32.data(), count);
    } else {
        matrix.f64.resize(count);
        read(in, matrix.f64.data(), count);
    }
    return matrix;
}

InputData load_input(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open precision input");
    char magic[8];
    read(in, magic, sizeof magic);
    if (std::memcmp(magic, "VPRCIN1\0", 8) != 0) throw std::runtime_error("bad precision input magic");
    std::uint64_t dims[2];
    read(in, dims, 2);
    if (!dims[0] || !dims[1] || dims[0] > 100000 || dims[1] > 100000 || dims[0] > 100000000 / dims[1])
        throw std::runtime_error("invalid precision input dimensions");
    std::uint8_t types[3];
    read(in, types, 3);
    std::uint8_t reserved[5];
    read(in, reserved, 5);
    const Storage s_type = decode_storage(types[0]);
    const Storage u_type = decode_storage(types[1]);
    const Storage w_type = decode_storage(types[2]);
    InputData data;
    data.n = std::size_t(dims[0]);
    data.g = std::size_t(dims[1]);
    data.s = read_matrix(in, s_type, data.n * data.g);
    data.u = read_matrix(in, u_type, data.n * data.g);
    data.w = read_matrix(in, w_type, data.n * data.n);
    data.fit.resize(data.n);
    data.genes.resize(data.g);
    read(in, data.fit.data(), data.fit.size());
    read(in, data.genes.data(), data.genes.size());
    if (in.peek() != EOF) throw std::runtime_error("trailing precision input bytes");
    return data;
}

velocitypp::Trimming trimming(const std::string& value) {
    if (value == "none") return velocitypp::Trimming::none;
    if (value == "upper") return velocitypp::Trimming::upper;
    if (value == "tails") return velocitypp::Trimming::tails;
    throw std::runtime_error("trimming must be none, upper, or tails");
}

template<class T> struct SteadyOutput {
    std::size_t rows, genes;
    std::vector<T>* velocity;
    std::vector<T>* spliced;
    std::vector<T>* unspliced;
};
template<class T> void sink_steady(void* context, const velocitypp::BasicMomentBlock<T>& moments,
                                   const velocitypp::BasicFitBlock<T>& fit) {
    auto& out = *static_cast<SteadyOutput<T>*>(context);
    for (std::size_t i = 0; i < moments.rows; ++i) for (std::size_t j = 0; j < moments.count; ++j) {
        const auto source = std::ptrdiff_t(i) * moments.stride + std::ptrdiff_t(j);
        const auto destination = i * out.genes + moments.first + j;
        (*out.spliced)[destination] = moments.spliced[source];
        (*out.unspliced)[destination] = moments.unspliced[source];
    }
    velocitypp::reference::residuals(moments, fit, out.velocity->data() + moments.first,
                                     std::ptrdiff_t(out.genes), 1);
}

template<class T> struct StochasticOutput {
    std::size_t rows, genes;
    std::vector<T>* velocity;
    std::vector<T>* residual2;
    std::vector<T>* spliced;
    std::vector<T>* unspliced;
    std::vector<T>* spliced_second;
    std::vector<T>* cross;
    std::vector<T>* unspliced_second;
};
template<class T> void sink_stochastic(void* context,
    const velocitypp::BasicStochasticMomentBlock<T>& moments,
    const velocitypp::BasicStochasticFitBlock<T>& fit) {
    auto& out = *static_cast<StochasticOutput<T>*>(context);
    for (std::size_t i = 0; i < moments.rows; ++i) for (std::size_t j = 0; j < moments.count; ++j) {
        const auto source = std::ptrdiff_t(i) * moments.stride + std::ptrdiff_t(j);
        const auto destination = i * out.genes + moments.first + j;
        (*out.spliced)[destination] = moments.spliced[source];
        (*out.unspliced)[destination] = moments.unspliced[source];
        (*out.spliced_second)[destination] = moments.spliced_second[source];
        (*out.cross)[destination] = moments.cross[source];
        (*out.unspliced_second)[destination] = moments.unspliced_second[source];
    }
    velocitypp::reference::stochastic_residuals(moments, fit,
        out.velocity->data() + moments.first, out.residual2->data() + moments.first,
        std::ptrdiff_t(out.genes), 1);
}

template<class T> void write_result(std::ostream& out, const InputData& data, Method method,
    const std::vector<T>& ms, const std::vector<T>& mu, const std::vector<T>& mss,
    const std::vector<T>& mus, const std::vector<T>& velocity, const std::vector<T>& residual2,
    const std::vector<T>& gamma, const std::vector<T>& offset, const std::vector<T>& offset2,
    const std::vector<T>& first_scale, const std::vector<T>& second_scale, const std::vector<T>& r2,
    const std::vector<T>& objective, const std::vector<T>& initial_gamma,
    const std::vector<T>& initial_offset, const std::vector<std::uint8_t>& selected,
    const std::vector<std::uint8_t>& initial_status, const std::vector<std::uint8_t>& status,
    bool low_signal, bool refit_all) {
    out.write("VPRCOT1\0", 8);
    const std::uint8_t backend = 0;
    const std::uint8_t tags[5] = {sizeof(T) == 4 ? std::uint8_t(1) : std::uint8_t(2),
        std::uint8_t(method), std::uint8_t(low_signal), std::uint8_t(refit_all), backend};
    write(out, tags, 5);
    const std::uint64_t dims[2] = {data.n, data.g};
    write(out, dims, 2);
    for (const auto* values : {&ms, &mu, &mss, &mus, &velocity, &residual2}) write(out, values->data(), values->size());
    for (const auto* values : {&gamma, &offset, &offset2, &first_scale, &second_scale, &r2,
                               &objective, &initial_gamma, &initial_offset}) write(out, values->data(), values->size());
    write(out, selected.data(), selected.size());
    write(out, initial_status.data(), initial_status.size());
    write(out, status.data(), status.size());
}

template<class T> void copy_fit(const velocitypp_test::FitStorage<T>& source,
                                std::vector<T>& gamma, std::vector<T>& offset,
                                std::vector<T>& r2, std::vector<std::uint8_t>& selected,
                                std::vector<std::uint8_t>& status) {
    for (std::size_t j = 0; j < gamma.size(); ++j) {
        gamma[j] = source.gamma[j];
        offset[j] = source.offset[j];
        r2[j] = source.r2[j];
        selected[j] = source.selected[j];
        status[j] = source.status[j];
    }
}

template<class T> void run_cpu(const InputData& data, Method method, velocitypp::FitOptions fit,
                               bool fit_second_intercept, const char* output_path) {
    auto sa = const_cast<InputData::Matrix&>(data.s).array(data.n, data.g);
    auto ua = const_cast<InputData::Matrix&>(data.u).array(data.n, data.g);
    auto wa = const_cast<InputData::Matrix&>(data.w).array(data.n, data.n);
    Acquired acquired_s{sa.view()}, acquired_u{ua.view()}, acquired_w{wa.view()};
    velocitypp::Input input{acquired_s.matrix, acquired_u.matrix, acquired_w.matrix,
        nullptr, 0, data.fit.data(), data.genes.data()};
    velocitypp::PreparedInput prepared{};
    std::vector<std::size_t> indices;
    std::vector<velocitypp::Entry> entries;
    if (!velocitypp_test::prepare(input, indices, entries, prepared))
        throw std::runtime_error("precision input preparation failed");

    const std::size_t output_count = data.n * data.g;
    std::vector<T> ms(output_count), mu(output_count), mss(output_count), mus(output_count);
    std::vector<T> velocity(output_count), residual2(output_count);
    std::vector<T> gamma(data.g), offset(data.g), offset2(data.g), scale1(data.g), scale2(data.g), r2(data.g), objective(data.g);
    std::vector<T> initial_gamma(data.g), initial_offset(data.g);
    std::vector<std::uint8_t> selected(data.g), initial_status(data.g), status(data.g);
    bool low_signal = false, refit_all = false;

    velocitypp::RunOptions run;
    run.workspace_bytes = 64u * 1024u * 1024u;
    run.block_genes = 16;
    if (method == Method::deterministic) {
        velocitypp_test::WorkspaceStorage<T> workspace;
        velocitypp_test::FitStorage<T> result(data.g);
        if (!workspace.bind(prepared, run.block_genes, false, run.workspace_bytes))
            throw std::runtime_error("insufficient deterministic caller workspace");
        const auto check = velocitypp::reference::steady_state_into<T, T>(prepared, workspace.workspace,
            result.result, velocity.data(), velocity.size(), std::ptrdiff_t(data.g), fit);
        if (!check) throw std::runtime_error("deterministic reference call failed");
        velocitypp::reference::moments<T>(prepared, 0, data.g, ms.data(), mu.data());
        copy_fit(result, gamma, offset, r2, selected, status);
        initial_gamma = gamma;
        initial_offset = offset;
        initial_status = status;
        low_signal = result.result.low_signal;
    } else {
        run.unspliced_second = true;
        velocitypp_test::WorkspaceStorage<T> workspace;
        velocitypp_test::StochasticStorage<T> result(data.g);
        if (!workspace.bind(prepared, run.block_genes, true, run.workspace_bytes, run.unspliced_second))
            throw std::runtime_error("insufficient stochastic caller workspace");
        velocitypp::StochasticOptions options;
        options.initialization = fit;
        options.fit_second_intercept = fit_second_intercept;
        std::vector<T> source_s(data.n), source_u(data.n);
        velocitypp::reference::second_moments<T>(prepared, 0, data.g, ms.data(), mu.data(),
            mss.data(), mus.data(), nullptr, source_s.data(), source_u.data());
        const auto check = velocitypp::reference::stochastic_into<T, T>(prepared, workspace.workspace,
            result.result, velocity.data(), velocity.size(), residual2.data(), residual2.size(),
            std::ptrdiff_t(data.g), options);
        if (!check) throw std::runtime_error("stochastic reference call failed");
        copy_fit(result.initialization, initial_gamma, initial_offset, r2, selected, initial_status);
        for (std::size_t j = 0; j < data.g; ++j) {
            gamma[j] = result.gamma[j]; offset[j] = result.offset[j]; offset2[j] = result.offset2[j];
            scale1[j] = result.first_scale[j]; scale2[j] = result.second_scale[j];
            objective[j] = result.objective[j]; status[j] = result.status[j];
        }
        low_signal = result.result.initialization.low_signal;
        refit_all = result.result.refit_all;
    }

    std::ofstream out(output_path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open precision output");
    write_result(out, data, method, ms, mu, mss, mus, velocity, residual2, gamma, offset, offset2,
        scale1, scale2, r2, objective, initial_gamma, initial_offset, selected, initial_status,
        status, low_signal, refit_all);
}

}

int main(int argc, char** argv) try {
    if (argc < 10) throw std::runtime_error("usage: precision_interop_driver INPUT OUTPUT deterministic|stochastic f32|f64 FIT_OFFSET FIT_OFFSET2 none|upper|tails MIN_R2 MIN_RATIO");
    const auto data = load_input(argv[1]);
    const std::string method_name = argv[3];
    const auto method = method_name == "deterministic" ? Method::deterministic :
        method_name == "stochastic" ? Method::stochastic : throw std::runtime_error("unknown model");
    const std::string precision = argv[4];
    velocitypp::FitOptions fit;
    fit.fit_intercept = std::stoi(argv[5]) != 0;
    fit.positive_intercept = true;
    fit.trimming = trimming(argv[7]);
    fit.percentile_low = 5;
    fit.percentile_high = 95;
    fit.min_r2 = std::stod(argv[8]);
    fit.min_ratio = std::stod(argv[9]);
    if (argc != 10) throw std::runtime_error("CPU precision driver accepts exactly nine arguments after program name");
    if (precision == "f32") run_cpu<float>(data, method, fit, std::stoi(argv[6]) != 0, argv[2]);
    else if (precision == "f64") run_cpu<double>(data, method, fit, std::stoi(argv[6]) != 0, argv[2]);
    else throw std::runtime_error("compute precision must be f32 or f64");
    return 0;
} catch (const std::exception& error) {
    std::cerr << "precision interop: " << error.what() << '\n';
    return 1;
}
