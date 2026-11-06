#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

import velocitypp;

namespace {
enum class Storage : std::uint8_t { f32 = 2, f64 = 3 };
enum class Method : std::uint8_t { deterministic = 1, stochastic = 2 };
struct InputData {
    std::size_t n = 0, g = 0;
    struct Matrix {
        Storage type{};
        std::vector<float> f32;
        std::vector<double> f64;
        anndatapp::ArrayView array_view(std::size_t rows, std::size_t columns) {
            const auto stride = std::ptrdiff_t(columns);
            if (type == Storage::f32) {
                auto array = anndatapp::Array::borrow_dense(f32.data(), {rows, columns}, {stride, 1});
                return array.view();
            }
            auto array = anndatapp::Array::borrow_dense(f64.data(), {rows, columns}, {stride, 1});
            return array.view();
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
    InputData::Matrix m; m.type = type;
    if (type == Storage::f32) read(in, (m.f32.resize(count), m.f32.data()), count);
    else read(in, (m.f64.resize(count), m.f64.data()), count);
    return m;
}

InputData load_input(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open precision input");
    char magic[8]; read(in, magic, sizeof magic);
    if (std::memcmp(magic, "VPRCIN1\0", 8) != 0) throw std::runtime_error("bad precision input magic");
    std::uint64_t dims[2]; read(in, dims, 2);
    if (!dims[0] || !dims[1] || dims[0] > 100000 || dims[1] > 100000 || dims[0] > 100000000 / dims[1])
        throw std::runtime_error("invalid precision input dimensions");
    std::uint8_t types[3]; read(in, types, 3);
    std::uint8_t reserved[5]; read(in, reserved, 5);
    const Storage s_type = decode_storage(types[0]);
    const Storage u_type = decode_storage(types[1]);
    const Storage w_type = decode_storage(types[2]);
    InputData d; d.n = std::size_t(dims[0]); d.g = std::size_t(dims[1]);
    d.s = read_matrix(in, s_type, d.n*d.g);
    d.u = read_matrix(in, u_type, d.n*d.g);
    d.w = read_matrix(in, w_type, d.n*d.n);
    d.fit.resize(d.n); d.genes.resize(d.g); read(in, d.fit.data(), d.fit.size()); read(in, d.genes.data(), d.genes.size());
    if (in.peek() != EOF) throw std::runtime_error("trailing precision input bytes");
    return d;
}

velocitypp::Trimming trimming(const std::string& s) {
    if (s == "none") return velocitypp::Trimming::none;
    if (s == "upper") return velocitypp::Trimming::upper;
    if (s == "tails") return velocitypp::Trimming::tails;
    throw std::runtime_error("trimming must be none, upper, or tails");
}

template<class T> void copy_view(anndatapp::ArrayView view, std::vector<T>& destination,
                                 std::size_t g, std::size_t first) {
    if (!view.rank()) return;
    auto bytes = view.bytes();
    for (std::size_t i = 0; i < view.extent(0); ++i) for (std::size_t j = 0; j < view.extent(1); ++j) {
        T value;
        std::memcpy(&value, bytes.data() + std::ptrdiff_t(i)*bytes.stride_bytes(0) + std::ptrdiff_t(j)*bytes.stride_bytes(1), sizeof(T));
        destination[i*g + first+j] = value;
    }
}
template<class T> struct SteadyOutput { std::size_t g; std::vector<T> *v,*ms,*mu; };
template<class T> struct StochasticOutput { std::size_t g; std::vector<T> *v,*r2,*ms,*mu,*mss,*mus; };
template<class T> void sink_steady(void* p, const velocitypp::OutputBlock& block) {
    auto& out=*static_cast<SteadyOutput<T>*>(p);
    copy_view<T>(block.velocity,*out.v,out.g,block.first);
    copy_view<T>(block.spliced,*out.ms,out.g,block.first);
    copy_view<T>(block.unspliced,*out.mu,out.g,block.first);
}
template<class T> void sink_stochastic(void* p, const velocitypp::StochasticOutputBlock& block) {
    auto& out=*static_cast<StochasticOutput<T>*>(p);
    copy_view<T>(block.velocity,*out.v,out.g,block.first);
    copy_view<T>(block.second_residual,*out.r2,out.g,block.first);
    copy_view<T>(block.spliced,*out.ms,out.g,block.first);
    copy_view<T>(block.unspliced,*out.mu,out.g,block.first);
    copy_view<T>(block.spliced_second,*out.mss,out.g,block.first);
    copy_view<T>(block.cross,*out.mus,out.g,block.first);
}

template<class T> void write_result(std::ostream& out, const InputData& d, Method method,
    std::vector<T>& ms, std::vector<T>& mu, std::vector<T>& mss, std::vector<T>& mus,
    std::vector<T>& velocity, std::vector<T>& residual2,
    std::vector<T>& gamma, std::vector<T>& offset, std::vector<T>& offset2,
    std::vector<T>& first_scale, std::vector<T>& second_scale, std::vector<T>& r2,
    std::vector<T>& objective, std::vector<T>& initial_gamma, std::vector<T>& initial_offset,
    std::vector<std::uint8_t>& selected, std::vector<std::uint8_t>& initial_status,
    std::vector<std::uint8_t>& status, bool low_signal, bool refit_all) {
    out.write("VPRCOT1\0", 8);
    const std::uint8_t backend=0;
    const std::uint8_t tags[5] = {sizeof(T) == 4 ? std::uint8_t(1) : std::uint8_t(2), std::uint8_t(method), std::uint8_t(low_signal), std::uint8_t(refit_all),backend};
    write(out, tags, 5);
    const std::uint64_t dims[2] = {d.n, d.g}; write(out, dims, 2);
    for (const auto* v : {&ms, &mu, &mss, &mus, &velocity, &residual2}) write(out, v->data(), v->size());
    for (const auto* v : {&gamma, &offset, &offset2, &first_scale, &second_scale, &r2,
                           &objective, &initial_gamma, &initial_offset}) write(out, v->data(), v->size());
    write(out, selected.data(), selected.size());
    write(out, initial_status.data(), initial_status.size());
    write(out, status.data(), status.size());
}

template<class T> void run_cpu(const InputData& d, Method method, velocitypp::FitOptions fit,
    bool fit_second_intercept, const char* output_path) {
    auto sa = const_cast<InputData::Matrix&>(d.s).array_view(d.n, d.g);
    auto ua = const_cast<InputData::Matrix&>(d.u).array_view(d.n, d.g);
    auto wa = const_cast<InputData::Matrix&>(d.w).array_view(d.n, d.n);
    auto fa = anndatapp::Array::borrow_dense(const_cast<std::uint8_t*>(d.fit.data()), {d.n}, {1});
    auto ga = anndatapp::Array::borrow_dense(const_cast<std::uint8_t*>(d.genes.data()), {d.g}, {1});
    velocitypp::Input input; input.spliced=sa; input.unspliced=ua; input.weights=wa;
    input.fit_mask=fa.view(); input.gene_mask=ga.view();
    velocitypp::PreparedInput prepared(input);
    const auto precision = std::is_same_v<T,float> ? velocitypp::ComputePrecision::float32 : velocitypp::ComputePrecision::float64;
    std::vector<T> ms(d.n*d.g), mu(d.n*d.g), mss(d.n*d.g), mus(d.n*d.g), v(d.n*d.g), residual2(d.n*d.g);
    std::vector<T> gamma(d.g), offset(d.g), offset2(d.g), scale1(d.g), scale2(d.g), r2(d.g), objective(d.g), initial_gamma(d.g), initial_offset(d.g);
    std::vector<std::uint8_t> selected(d.g), initial_status(d.g), status(d.g, 255);
    bool low_signal = false, refit_all = false;
    if (method == Method::deterministic) {
        velocitypp::RunOptions run; run.block_genes=16; run.compute_precision=precision;
        run.output_dtype=std::is_same_v<T,float>?anndatapp::DType::float32:anndatapp::DType::float64;
        run.spliced_moments=true;run.unspliced_moments=true;
        velocitypp::Workspace workspace(prepared, run);
        SteadyOutput<T> capture{d.g,&v,&ms,&mu};
        velocitypp::BlockSink sink; sink.context=&capture; sink.write=&sink_steady<T>;
        auto result=velocitypp::cpu::steady_state(prepared,workspace,fit,sink);
        auto rg=result.gamma.view().template dense<T>().data();
        auto ro=result.offset.view().template dense<T>().data();
        auto rr=result.r2.view().template dense<T>().data();
        auto rs=result.genes.view().template dense<bool>().data();
        auto rst=result.status.view().template dense<std::uint8_t>().data();
        for(std::size_t j=0;j<d.g;++j){gamma[j]=initial_gamma[j]=rg[j];offset[j]=initial_offset[j]=ro[j];r2[j]=rr[j];selected[j]=rs[j];initial_status[j]=status[j]=rst[j];}
        low_signal=result.low_signal;
    } else {
        velocitypp::StochasticRunOptions run; run.block_genes=16; run.compute_precision=precision; run.second_residual=true;
        run.output_dtype=std::is_same_v<T,float>?anndatapp::DType::float32:anndatapp::DType::float64;
        run.spliced_moments=true;run.unspliced_moments=true;run.spliced_second=true;run.cross=true;
        velocitypp::StochasticWorkspace workspace(prepared,run);
        velocitypp::StochasticOptions options; options.initialization=fit; options.fit_second_intercept=fit_second_intercept;
        StochasticOutput<T> capture{d.g,&v,&residual2,&ms,&mu,&mss,&mus};
        velocitypp::StochasticBlockSink sink; sink.context=&capture; sink.write=&sink_stochastic<T>;
        auto result=velocitypp::cpu::stochastic(prepared,workspace,options,sink);
        auto copy=[&](anndatapp::ArrayView view,std::vector<T>& target){auto p=view.template dense<T>().data();std::copy_n(p,d.g,target.data());};
        copy(result.gamma.view(),gamma);copy(result.offset.view(),offset);copy(result.offset2.view(),offset2);copy(result.first_scale.view(),scale1);copy(result.second_scale.view(),scale2);copy(result.objective.view(),objective);
        copy(result.initialization.gamma.view(),initial_gamma);copy(result.initialization.offset.view(),initial_offset);copy(result.initialization.r2.view(),r2);
        auto sel=result.initialization.genes.view().template dense<bool>().data();
        auto ist=result.initialization.status.view().template dense<std::uint8_t>().data();
        auto st=result.status.view().template dense<std::uint8_t>().data();
        for(std::size_t j=0;j<d.g;++j){selected[j]=sel[j];initial_status[j]=ist[j];status[j]=st[j];}
        refit_all=result.refit_all;
    }
    std::ofstream out(output_path,std::ios::binary); if(!out) throw std::runtime_error("cannot open precision output");
    write_result(out,d,method,ms,mu,mss,mus,v,residual2,gamma,offset,offset2,scale1,scale2,r2,objective,initial_gamma,initial_offset,selected,initial_status,status,low_signal,refit_all);
}

}

int main(int argc,char** argv) try {
    if(argc<10) throw std::runtime_error("usage: precision_interop_driver INPUT OUTPUT deterministic|stochastic f32|f64 FIT_OFFSET FIT_OFFSET2 none|upper|tails MIN_R2 MIN_RATIO");
    const auto d=load_input(argv[1]);
    const std::string method_name=argv[3];
    const auto method=method_name=="deterministic"?Method::deterministic:method_name=="stochastic"?Method::stochastic:throw std::runtime_error("unknown model");
    const std::string precision=argv[4];
    velocitypp::FitOptions fit;fit.fit_intercept=std::stoi(argv[5])!=0;fit.positive_intercept=true;
    fit.trimming=trimming(argv[7]);fit.percentile_low=5;fit.percentile_high=95;
    fit.min_r2=std::stod(argv[8]);fit.min_ratio=std::stod(argv[9]);
    if(argc!=10)throw std::runtime_error("CPU precision driver accepts exactly nine arguments after program name");
    if(precision=="f32")run_cpu<float>(d,method,fit,std::stoi(argv[6])!=0,argv[2]);
    else if(precision=="f64")run_cpu<double>(d,method,fit,std::stoi(argv[6])!=0,argv[2]);
    else throw std::runtime_error("compute precision must be f32 or f64");
    return 0;
} catch(const std::exception& e) { std::cerr<<"precision interop: "<<e.what()<<'\n';return 1; }
