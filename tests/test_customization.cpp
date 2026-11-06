#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
import velocitypp;
#include "support.hh"

void check(bool condition) { if (!condition) throw std::runtime_error("customization check failed"); }

// Consumer data is gene-major and genes appear in reverse logical order.
// Its state and intermediates never use reference buffers or result types.
struct Foreign {
    double packed[8]{4,8,3,6,2,4,1,2};
    double sums[4]{}, coefficients[4]{};
    double output[8]{};
    int blocks=0;
    struct Packet { std::size_t first, count; double* packed; };
    Packet moments(std::size_t first,std::size_t count) { return {first,count,packed}; }
    Packet fit(Packet p) {
        for(std::size_t j=0;j<p.count;++j) { auto g=p.first+j; auto k=3-g; sums[g]=p.packed[2*k]+p.packed[2*k+1]; }
        return p;
    }
    void emit(Packet p,Packet f) {
        check(p.first==f.first && p.count==f.count); ++blocks;
        for(std::size_t j=0;j<p.count;++j) { auto g=p.first+j; auto k=3-g;
            for(std::size_t r=0;r<2;++r) output[2*k+r]=packed[2*k+r]+sums[g]; }
    }
    int finish() { return blocks; }
};
struct Fused : Foreign {
    void process_block(std::size_t first,std::size_t count) { auto p=moments(first,count); auto f=fit(p);emit(p,f); }
};
struct Whole : Foreign {
    int run() { for(std::size_t g=0;g<4;++g){auto p=moments(g,1);auto f=fit(p);emit(p,f);}return 97; }
};
void check_foreign(const Foreign& f) {
    for(std::size_t g=0;g<4;++g) { auto k=3-g; auto sum=f.packed[2*k]+f.packed[2*k+1];
        check(f.output[2*k]==f.packed[2*k]+sum && f.output[2*k+1]==f.packed[2*k+1]+sum); }
}
velocitypp::Status rt_moments(void* p,std::size_t first,std::size_t count) {
    auto& x=*static_cast<Foreign*>(p);
    for(std::size_t j=0;j<count;++j) { auto g=first+j,k=3-g;x.sums[g]=x.packed[2*k]+x.packed[2*k+1]; }
    return {};
}
velocitypp::Status rt_fit(void* p,std::size_t first,std::size_t count) {
    auto& x=*static_cast<Foreign*>(p); for(std::size_t j=0;j<count;++j)x.sums[first+j]*=2;return {};
}
velocitypp::Status rt_fit_identity(void*,std::size_t,std::size_t) { return {}; }
velocitypp::Status rt_emit(void* p,std::size_t first,std::size_t count) {
    auto& x=*static_cast<Foreign*>(p);++x.blocks;
    for(std::size_t j=0;j<count;++j) {auto g=first+j,k=3-g;for(std::size_t r=0;r<2;++r)x.output[2*k+r]=x.packed[2*k+r]+x.sums[g];}
    return {};
}
velocitypp::Status rt_fused(void* p,std::size_t first,std::size_t count) {
    auto s=rt_moments(p,first,count);if(!s)return s;return rt_emit(p,first,count);
}
velocitypp::Status rt_whole(void* p) { auto& x=*static_cast<Foreign*>(p);x.blocks=77;x.output[0]=123;return {}; }
velocitypp::Status rt_fail(void*,std::size_t,std::size_t) { return {velocitypp::Code::argument}; }
velocitypp::Status rt_resolve_fail(void*) { return {velocitypp::Code::argument}; }
velocitypp::Status rt_initialize_foreign(void* p) {
    auto& x=*static_cast<Foreign*>(p);
    for(std::size_t g=0;g<4;++g) x.coefficients[g]=x.packed[2*(3-g)];
    return {};
}
velocitypp::Status rt_scale_foreign(void* p,std::size_t first,std::size_t count) {
    auto& x=*static_cast<Foreign*>(p);
    for(std::size_t j=0;j<count;++j) {auto g=first+j;x.sums[g]*=x.coefficients[g];}
    return {};
}

struct CustomFit {
    velocitypp::ReferenceStages& reference;
    double gamma[1]{3}, offset[1]{0}, r2[1]{.9};
    bool selected[1]{true};
    uint8_t status[1]{uint8_t(velocitypp::FitStatus::ok)};
    velocitypp::MomentBlock moments(std::size_t first,std::size_t count) { return reference.moments(first,count); }
    velocitypp::FitBlock fit(velocitypp::MomentBlock m) { return {m.first,m.count,gamma,offset,r2,selected,status}; }
    void emit(velocitypp::MomentBlock m,velocitypp::FitBlock f) { reference.emit(m,f); }
    velocitypp::FitResult finish() { return reference.finish(); }
};
struct CustomMoments {
    velocitypp::ReferenceStages& reference;
    double spliced[3]{1,2,3},unspliced[3]{6,12,18};
    velocitypp::MomentBlock moments(std::size_t first,std::size_t count) { return {first,count,3,1,spliced,unspliced}; }
    velocitypp::FitBlock fit(velocitypp::MomentBlock m) { return reference.fit(m); }
    void emit(velocitypp::MomentBlock m,velocitypp::FitBlock f) { reference.emit(m,f); }
    velocitypp::FitResult finish() { return reference.finish(); }
};
void collect(void* context,const velocitypp::MomentBlock& m,const velocitypp::FitBlock& f) {
    velocitypp::reference::residuals(m,f,static_cast<double*>(context),ptrdiff_t(m.count));
}

int main() try {
    Foreign a;check(velocitypp::steady_state_with(a,4,2)==2);check_foreign(a);
    Fused b;check(velocitypp::steady_state_with(b,4,3)==2);check_foreign(b);
    Whole c;check(velocitypp::steady_state_with(c,4,2)==97);check_foreign(c);
    Foreign d;velocitypp::RuntimeOperations ops{};ops.context=&d;ops.moments=rt_moments;ops.fit=rt_fit;ops.emit=rt_emit;
    check(bool(velocitypp::steady_state_with(ops,4,2)) && d.blocks==2);
    for(std::size_t g=0;g<4;++g){auto k=3-g;double sum=d.packed[2*k]+d.packed[2*k+1];for(std::size_t r=0;r<2;++r)check(d.output[2*k+r]==d.packed[2*k+r]+2*sum);}
    ops.fit=rt_fit_identity;d.blocks=0;check(bool(velocitypp::steady_state_with(ops,4,3)));check_foreign(d);
    ops.process_block=rt_fused;ops.moments=rt_fail;ops.fit=rt_fail;ops.emit=rt_fail;d.blocks=0;
    check(bool(velocitypp::steady_state_with(ops,4,2)) && d.blocks==2);check_foreign(d);
    ops.run=rt_whole;d.blocks=0;check(bool(velocitypp::steady_state_with(ops,4,2)) && d.blocks==77 && d.output[0]==123);

    // The initialization region derives coefficients in the consumer's
    // reversed gene-major representation. Reference initialization and its
    // resolve callback are deliberately unavailable and must not be called.
    Foreign e;velocitypp::RuntimeOperations stochastic_ops{};
    stochastic_ops.context=&e;stochastic_ops.initialize=rt_initialize_foreign;
    stochastic_ops.initial_block=rt_fail;stochastic_ops.resolve=rt_resolve_fail;
    stochastic_ops.moments=rt_moments;stochastic_ops.scales=rt_scale_foreign;
    stochastic_ops.fit=rt_fit;stochastic_ops.emit=rt_emit;
    check(bool(velocitypp::stochastic_with(stochastic_ops,4,2)) && e.blocks==2);
    for(std::size_t g=0;g<4;++g) {
        auto k=3-g;double coefficient=e.packed[2*k];
        double sum=e.packed[2*k]+e.packed[2*k+1];
        for(std::size_t r=0;r<2;++r) check(e.output[2*k+r]==e.packed[2*k+r]+2*coefficient*sum);
    }

    double s[3]{1,2,3},u[3]{2,4,6},w[9]{1,0,0,0,1,0,0,0,1};
    auto sa=anndatapp::Array::borrow_dense(s,{3,1},{1,1});
    auto ua=anndatapp::Array::borrow_dense(u,{3,1},{1,1});
    auto wa=anndatapp::Array::borrow_dense(w,{3,3},{3,1});
    Acquired sv(sa.view()),uv(ua.view()),wv(wa.view());
    velocitypp::Input input{sv.matrix,uv.matrix,wv.matrix};
    velocitypp::PreparationSize ps{};check(bool(velocitypp::preparation_size(input,ps)));
    size_t indices[128]{};velocitypp::Entry entries[64]{};
    check(ps.indices<=128 && ps.entries<=64);
    velocitypp::PreparedInput prepared{};check(bool(velocitypp::prepare(input,{indices,128,entries,64},prepared)));
    velocitypp::RunOptions ro{};velocitypp::WorkspaceSize ws{};
    check(bool(velocitypp::workspace_size<double>(prepared,ro,false,ws)));
    double scratch[256]{};uint8_t mask[16]{};check(ws.scalars<=256 && ws.masks<=16);
    velocitypp::Workspace<double> workspace{};
    check(bool(velocitypp::bind_workspace(prepared,ro,false,scratch,256,mask,16,workspace)));
    double gamma[1]{},offset[1]{},r2[1]{},velocity[3]{};bool selected[1]{};uint8_t status[1]{};
    velocitypp::FitResult result{gamma,offset,r2,selected,status,1};
    velocitypp::ReferenceStages stage{};
    check(bool(velocitypp::bind_stages(prepared,workspace,{},result,stage,{velocity,collect})));
    CustomFit custom{stage};auto replaced=velocitypp::steady_state_with(custom,prepared.genes,workspace.block);
    check(std::abs(replaced.gamma[0]-3)<1e-12);
    for(int i=0;i<3;++i)check(std::abs(velocity[i]+s[i])<1e-12);
    check(bool(velocitypp::bind_stages(prepared,workspace,{},result,stage,{velocity,collect})));
    CustomMoments custom_moments{stage};auto other=velocitypp::steady_state_with(custom_moments,prepared.genes,workspace.block);
    check(std::abs(other.gamma[0]-6)<1e-12);
    for(double x:velocity)check(std::abs(x)<1e-12);
    check(bool(velocitypp::reference::steady_state_into(prepared,workspace,result,velocity,3,1)));
    check(std::abs(result.gamma[0]-2)<1e-12);for(double x:velocity)check(std::abs(x)<1e-12);
    double x[4]{1e8,1e8+1,1e8+2,1e8+3},y[4]{2e8+1,2e8+3,2e8+5,2e8+7},fit_scratch[4];
    velocitypp::FitOptions centered;centered.trimming=velocitypp::Trimming::none;centered.fit_intercept=true;
    auto stable=velocitypp::reference::fit_gene(x,y,4,1,centered,fit_scratch);
    check(std::abs(stable.gamma-2)<1e-12 && std::abs(stable.offset-1)<1e-12);
    std::puts("foreign layout, runtime swap, fused/full replacement and reference stages passed");
    return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
