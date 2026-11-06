// Linux/glibc allocation interposition for the standalone reference. Build and
// execute only without sanitizers: ASan owns allocator interception itself.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <new>
#include <stdexcept>

import velocitypp;
import velocitypp.precision;
#include "support.hh"

#if !defined(__GLIBC__)
int main() { return 77; } // CTest marks non-glibc allocator instrumentation unavailable.
#else

extern "C" void* __libc_malloc(std::size_t) noexcept;
extern "C" void* __libc_calloc(std::size_t,std::size_t) noexcept;
extern "C" void* __libc_realloc(void*,std::size_t) noexcept;
extern "C" void __libc_free(void*) noexcept;
extern "C" void* __libc_memalign(std::size_t,std::size_t) noexcept;

namespace {
bool tracking=false;
struct Counts { std::size_t malloc=0,calloc=0,realloc=0,free=0,aligned=0,news=0,deletes=0; } counts;
}
extern "C" void* malloc(std::size_t n) noexcept { if(tracking)++counts.malloc;return __libc_malloc(n); }
extern "C" void* calloc(std::size_t n,std::size_t w) noexcept { if(tracking)++counts.calloc;return __libc_calloc(n,w); }
extern "C" void* realloc(void* p,std::size_t n) noexcept { if(tracking)++counts.realloc;return __libc_realloc(p,n); }
extern "C" void free(void* p) noexcept { if(tracking)++counts.free;__libc_free(p); }
extern "C" void* aligned_alloc(std::size_t a,std::size_t n) noexcept { if(tracking)++counts.aligned;return __libc_memalign(a,n); }
extern "C" int posix_memalign(void** p,std::size_t a,std::size_t n) noexcept {
    if(tracking)++counts.aligned;
    if(a<sizeof(void*) || (a&(a-1))) return 22;
    void* q=__libc_memalign(a,n);if(!q)return 12;*p=q;return 0;
}
void* operator new(std::size_t n) { if(tracking)++counts.news;if(void* p=__libc_malloc(n?n:1))return p;throw std::bad_alloc(); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n,std::align_val_t a) { if(tracking)++counts.news;if(void* p=__libc_memalign(std::size_t(a),n?n:1))return p;throw std::bad_alloc(); }
void* operator new[](std::size_t n,std::align_val_t a) { return ::operator new(n,a); }
void operator delete(void* p) noexcept { if(tracking)++counts.deletes;__libc_free(p); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete(void* p,std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p,std::size_t) noexcept { ::operator delete(p); }
void operator delete(void* p,std::align_val_t) noexcept { ::operator delete(p); }
void operator delete[](void* p,std::align_val_t) noexcept { ::operator delete(p); }
void operator delete(void* p,std::size_t,std::align_val_t) noexcept { ::operator delete(p); }
void operator delete[](void* p,std::size_t,std::align_val_t) noexcept { ::operator delete(p); }

namespace vp=velocitypp;
namespace pr=velocitypp::precision;

void require(bool value,const char* message) {
    if(!value) { std::fprintf(stderr,"allocation test: %s\n",message);throw std::runtime_error(message); }
}
template<class F> void phase(const char* name,F&& f) {
    counts={};tracking=true;
    try { f(); } catch(...) { tracking=false;throw; }
    tracking=false;
    if(counts.malloc||counts.calloc||counts.realloc||counts.free||counts.aligned||counts.news||counts.deletes) {
        std::fprintf(stderr,"%s: malloc=%zu calloc=%zu realloc=%zu free=%zu aligned=%zu new=%zu delete=%zu\n",
            name,counts.malloc,counts.calloc,counts.realloc,counts.free,counts.aligned,counts.news,counts.deletes);
        throw std::runtime_error("reference allocated or released storage");
    }
}
struct Dispatch {
    size_t calls=0, values=0;
    void process_block(size_t,size_t count) { ++calls;values+=count; }
    size_t finish() { return values; }
};
vp::Status dispatch_block(void* p,size_t,size_t count) {
    auto& d=*static_cast<Dispatch*>(p);++d.calls;d.values+=count;return {};
}
vp::Status dispatch_stage(void* p) { ++static_cast<Dispatch*>(p)->calls;return {}; }
struct Original {
    double* s; double* u; std::size_t rows,columns;
};
bool source(void* state,std::size_t gene,pr::OriginalColumn& c) {
    auto& v=*static_cast<Original*>(state);
    c={v.s+gene,v.u+gene,ptrdiff_t(v.columns*sizeof(double)),ptrdiff_t(v.columns*sizeof(double)),
       pr::OriginalType::float64,pr::OriginalType::float64,v.rows};
    return true;
}

int main() try {
    // Prove interposition is active, including direct libc calls and both
    // ordinary and over-aligned C++ allocation, before measuring reference.
    using Malloc = void* (*)(std::size_t) noexcept;
    using Calloc = void* (*)(std::size_t,std::size_t) noexcept;
    using Realloc = void* (*)(void*,std::size_t) noexcept;
    using Free = void (*)(void*) noexcept;
    Malloc volatile m=::malloc;Calloc volatile c=::calloc;Realloc volatile r=::realloc;Free volatile f=::free;
    counts={};tracking=true;
    void* p=m(32);void* q=c(2,32);q=r(q,96);f(p);f(q);
    void* a=::operator new(16);::operator delete(a);
    void* b=::operator new(32,std::align_val_t(16));::operator delete(b,std::align_val_t(16));
    tracking=false;
    require(counts.malloc && counts.calloc && counts.realloc && counts.free && counts.news>=2 && counts.deletes>=2,
            "allocation interception self-check");
    constexpr size_t n=4,g=2;
    double s[n*g]{1,2,2,3,3,5,4,8},u[n*g]{2,4,4,6,6,10,8,16};
    double w[n*n]{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    auto sa=anndatapp::Array::borrow_dense(s,{n,g},{ptrdiff_t(g),1});
    auto ua=anndatapp::Array::borrow_dense(u,{n,g},{ptrdiff_t(g),1});
    auto wa=anndatapp::Array::borrow_dense(w,{n,n},{ptrdiff_t(n),1});
    Acquired sv(sa.view()),uv(ua.view()),wv(wa.view());
    vp::Input input{sv.matrix,uv.matrix,wv.matrix};
    vp::PreparationSize ps{};vp::PreparedInput prepared{};vp::Status status{};
    size_t index[256]{};vp::Entry entries[256]{};
    vp::PreparationBuffers pb{index,256,entries,256};
    vp::RunOptions run{};run.block_genes=2;
    vp::WorkspaceSize need{};
    double storage[512]{};uint8_t mask[n]{};vp::Workspace<double> workspace{};
    double gamma[g]{},offset[g]{},r2[g]{},velocity[n*g]{};float velocity32[n*g]{},residual32[n*g]{};bool selected[g]{};uint8_t fit_status[g]{};
    vp::FitResult result{gamma,offset,r2,selected,fit_status,g};
    double sg[g]{},so[g]{},so2[g]{},scale1[g]{},scale2[g]{},objective[g]{},residual[n*g]{};uint8_t stochastic_status[g]{};
    double ig[g]{},io[g]{},ir[g]{};bool iselected[g]{};uint8_t istatus[g]{};
    vp::FitResult initial{ig,io,ir,iselected,istatus,g};
    vp::StochasticFitResult stochastic_result{initial,sg,so,so2,scale1,scale2,objective,stochastic_status,g};
    double ms[n*g]{},mu[n*g]{},mss[n*g]{},mus[n*g]{},muu[n*g]{},source_s[n]{},source_u[n]{};
    vp::reference::NeighborhoodSize ns{};size_t neighborhood_indices[256]{},columns[256]{},offsets[n+1]{};vp::Entry neighborhood_scratch[256]{};float values[256]{};
    vp::reference::NeighborhoodBuffers nb{neighborhood_indices,256,neighborhood_scratch,256,values,columns,offsets,256,n+1};size_t used=0;
    vp::FitOptions fit{};vp::StochasticOptions so_options{};
    pr::PrecisionCheck precision_status{};pr::SensitivityReport sensitivity{};
    double sensitivity_s[n]{1,2,3,4},sensitivity_u[n]{2,4,6,8},sensitivity_scratch[n]{};
    vp::GeneFit baseline{2,0,1,vp::FitStatus::ok,true};

    // The byte stride fits ptrdiff_t but its row extent does not.  Descriptor
    // acquisition is caller work; validation must reject before dereferencing.
    double placeholder=0;
    auto huge=anndatapp::Array::borrow_dense(&placeholder,{n,g},{PTRDIFF_MAX/ptrdiff_t(sizeof(double)),1},anndatapp::Validation::none);
    Acquired huge_view(huge.view());vp::Input huge_input=input;huge_input.spliced=huge_view.matrix;
    phase("dense_stride_overflow",[&]{status=vp::validate(huge_input);});
    require(status.code==vp::Code::overflow,"dense stride overflow");

    for(int iteration=0;iteration<2;++iteration) {
        run.unspliced_second=iteration!=0;
        Dispatch dispatched{};size_t static_result=0;
        phase("static_dispatch",[&]{static_result=vp::steady_state_with(dispatched,4,2);});
        require(static_result==4 && dispatched.calls==2,"static dispatch");
        vp::RuntimeOperations runtime{};runtime.context=&dispatched;runtime.process_block=dispatch_block;
        dispatched={};phase("runtime_dispatch",[&]{status=vp::steady_state_with(runtime,4,2);});
        require(bool(status)&&dispatched.values==4&&dispatched.calls==2,"runtime dispatch");
        runtime.initialize=dispatch_stage;
        dispatched={};phase("runtime_stochastic_dispatch",[&]{status=vp::stochastic_with(runtime,4,2);});
        require(bool(status)&&dispatched.values==4&&dispatched.calls==3,"runtime stochastic dispatch");
        phase("validate",[&]{status=vp::validate(input);});require(bool(status),"validate");
        vp::Input invalid_mask=input;uint8_t one=1;
        invalid_mask.fit_mask=&one;invalid_mask.fit_mask_stride=PTRDIFF_MAX;
        phase("invalid_mask_stride",[&]{status=vp::validate(invalid_mask);});
        require(status.code==vp::Code::overflow,"mask stride overflow");
        phase("preparation_size",[&]{status=vp::preparation_size(input,ps);});require(bool(status)&&ps.indices<=256&&ps.entries<=256,"preparation size");
        phase("prepare_capacity",[&]{vp::PreparedInput dummy;status=vp::prepare(input,{nullptr,0,nullptr,0},dummy);});
        require((ps.indices||ps.entries) ? status.code==vp::Code::capacity : bool(status),"prepare capacity/empty storage");
        phase("prepare",[&]{status=vp::prepare(input,pb,prepared);});require(bool(status),"prepare");
        phase("prepare_unchecked",[&]{status=vp::prepare_unchecked(input,pb,prepared);});require(bool(status),"prepare unchecked");
        phase("neighborhood_size",[&]{status=vp::reference::neighborhood_size(wv.matrix,ns);});require(bool(status),"neighborhood size");
        phase("neighborhood_capacity",[&]{status=vp::reference::prepare_neighborhood(wv.matrix,{},used);});
        require(status.code==vp::Code::capacity,"neighborhood capacity error");
        phase("neighborhood",[&]{status=vp::reference::prepare_neighborhood(wv.matrix,nb,used);});require(bool(status)&&used==n,"neighborhood");
        phase("workspace_size",[&]{status=vp::workspace_size<double>(prepared,run,false,need);});
        require(bool(status)&&need.scalars<=512&&need.masks<=n,"workspace size");
        phase("workspace_capacity",[&]{vp::Workspace<double> dummy;status=vp::bind_workspace(prepared,run,false,static_cast<double*>(nullptr),0,nullptr,0,dummy);});
        require(status.code==vp::Code::capacity,"workspace capacity error");
        phase("bind_workspace",[&]{status=vp::bind_workspace(prepared,run,false,storage,512,mask,n,workspace);});require(bool(status),"bind workspace");
        phase("validate_fit",[&]{status=vp::reference::validate(fit);});require(bool(status),"validate fit");
        auto bad=fit;bad.percentile_low=-1;
        phase("validate_invalid",[&]{status=vp::reference::validate(bad);});require(status.code!=vp::Code::ok,"validate invalid");
        phase("moments",[&]{vp::reference::moments(prepared,0,g,ms,mu);});
        phase("fit_gene",[&]{baseline=vp::reference::fit_gene(sensitivity_s,sensitivity_u,n,1,fit,sensitivity_scratch);});
        for(double& value:velocity)value=123;
        double old_gamma=gamma[0];
        phase("steady_capacity",[&]{status=vp::reference::steady_state_into(prepared,workspace,result,velocity,n*g-1,g,fit);});
        require(status.code==vp::Code::capacity && status.required==n*g && gamma[0]==old_gamma,"steady output capacity");
        for(double value:velocity)require(value==123,"steady capacity leaves output intact");
        phase("steady_state",[&]{status=vp::reference::steady_state_into(prepared,workspace,result,velocity,n*g,g,fit);});require(bool(status),"steady state");
        phase("steady_mixed",[&]{status=vp::reference::steady_state_into(prepared,workspace,result,velocity32,n*g,g,fit);});require(bool(status),"steady mixed output");
        vp::ReferenceStages stages{};
        phase("bind_stages",[&]{status=vp::bind_stages(prepared,workspace,fit,result,stages);});require(bool(status),"bind stages");
        vp::MomentBlock mb{};vp::FitBlock fb{};
        phase("stage_moments",[&]{mb=stages.moments(0,g);});
        phase("stage_fit",[&]{fb=stages.fit(mb);});
        phase("stage_emit",[&]{stages.emit(mb,fb);});
        phase("stage_finish",[&]{(void)stages.finish();});
        phase("sensitivity_validate",[&]{precision_status=pr::validate_sensitivity(n,1,sensitivity_s,sensitivity_u,sensitivity_scratch,n,fit);});
        require(precision_status.code==pr::PrecisionCode::ok,"sensitivity validation");
        phase("sensitivity_capacity",[&]{precision_status=pr::validate_sensitivity(n,1,sensitivity_s,sensitivity_u,nullptr,0,fit);});
        require(precision_status.code==pr::PrecisionCode::insufficient_capacity,"sensitivity capacity");
        phase("sensitivity",[&]{sensitivity=pr::sensitivity_gene(sensitivity_s,sensitivity_u,n,1,baseline,fit,sensitivity_scratch);});
        require(sensitivity.state!=pr::SensitivityState::unscoreable,"sensitivity");
        phase("stochastic_workspace_size",[&]{status=vp::workspace_size<double>(prepared,run,true,need);});
        require(bool(status)&&need.scalars<=512&&need.masks<=n,"stochastic workspace size");
        phase("bind_stochastic",[&]{status=vp::bind_workspace(prepared,run,true,storage,512,mask,n,workspace);});require(bool(status),"bind stochastic");
        phase("second_moments",[&]{vp::reference::second_moments(prepared,0,g,ms,mu,mss,mus,muu,source_s,source_u);});
        phase("validate_stochastic",[&]{status=vp::reference::validate(so_options);});require(bool(status),"validate stochastic");
        for(double& value:velocity)value=123;for(double& value:residual)value=456;
        double old_stochastic_gamma=sg[0];
        phase("stochastic_capacity",[&]{status=vp::reference::stochastic_into(prepared,workspace,stochastic_result,velocity,n*g-1,residual,n*g,g,so_options);});
        require(status.code==vp::Code::capacity && status.required==n*g && sg[0]==old_stochastic_gamma,"stochastic output capacity");
        for(double value:velocity)require(value==123,"stochastic capacity leaves output intact");
        for(double value:residual)require(value==456,"stochastic capacity leaves residual intact");
        phase("stochastic_second_capacity",[&]{status=vp::reference::stochastic_into(prepared,workspace,stochastic_result,velocity,n*g,residual,n*g-1,g,so_options);});
        require(status.code==vp::Code::capacity && status.required==n*g && sg[0]==old_stochastic_gamma,"stochastic second capacity");
        for(double value:velocity)require(value==123,"second capacity leaves output intact");
        for(double value:residual)require(value==456,"second capacity leaves residual intact");
        phase("stochastic",[&]{status=vp::reference::stochastic_into(prepared,workspace,stochastic_result,velocity,n*g,residual,n*g,g,so_options);});require(bool(status),"stochastic");
        phase("stochastic_mixed",[&]{status=vp::reference::stochastic_into(prepared,workspace,stochastic_result,velocity32,n*g,residual32,n*g,g,so_options);});require(bool(status),"stochastic mixed output");
        vp::StochasticReferenceStages ss{};
        phase("bind_stochastic_stages",[&]{status=vp::bind_stages(prepared,workspace,so_options,stochastic_result,ss);});require(bool(status),"bind stochastic stages");
        vp::MomentBlock im{};vp::FitBlock ib{};vp::StochasticMomentBlock sm{};vp::StochasticScaleBlock sb{};vp::StochasticFitBlock sf{};
        phase("initial_moments",[&]{im=ss.initial_moments(0,g);});
        phase("initial_fit",[&]{ib=ss.initial_fit(im);});
        phase("accept_initialization",[&]{ss.accept_initialization(im,ib);});
        phase("resolve_selection",[&]{ss.resolve_selection();});
        phase("stochastic_moments",[&]{sm=ss.moments(0,g);});
        phase("stochastic_scales",[&]{sb=ss.scales(sm);});
        phase("stochastic_fit",[&]{sf=ss.fit(sm,sb);});
        phase("stochastic_emit",[&]{ss.emit(sm,sf);});
        phase("stochastic_finish",[&]{(void)ss.finish();});

        Original original{s,u,n,g};uint8_t promote_mask[g]{1,1},promotion_status[g]{};
        vp::MomentBlock original_initial{0,g,n,ptrdiff_t(g),ms,mu};
        vp::StochasticMomentBlock original_second{0,g,n,ptrdiff_t(g),ms,mu,mss,mus,muu};
        pr::ReferenceMoments<double> replacement{&prepared,&original_initial,&original_second};
        pr::Promotion promotion{nullptr,g,g,n,promote_mask,g,promotion_status,g,0,{&original,source},{&replacement,pr::replace_reference64}};
        phase("promotion_validate",[&]{precision_status=pr::validate_promotion(promotion);});require(precision_status.code==pr::PrecisionCode::ok,"promotion validate");
        phase("promotion_initialize",[&]{precision_status=pr::initialize_promotion(promotion);});require(precision_status.code==pr::PrecisionCode::ok,"promotion initialize");
        phase("promotion_initial",[&]{precision_status=pr::promote_initial(promotion,0,g);});require(precision_status.code==pr::PrecisionCode::ok,"promotion initial");
        phase("promotion_second",[&]{precision_status=pr::promote_second(promotion,0,g);});require(precision_status.code==pr::PrecisionCode::ok,"promotion second");
        promotion.status_capacity=0;
        phase("promotion_capacity",[&]{precision_status=pr::validate_promotion(promotion);});require(precision_status.code==pr::PrecisionCode::insufficient_capacity,"promotion capacity");
    }
    // Separate CSR/CSC inputs exercise sparse topology, duplicate coordinates,
    // preparation scratch and graph traversal under the same allocator probe.
    double sparse_s[]{1,2,3,4},sparse_u[]{2,6,4,10,16},sparse_w[]{1,1,1,1};
    int32_t si[]{0,1,0,1},ui[]{0,2,1,1,3},wi[]{0,1,2,3};
    int64_t sparse_so[]{0,1,2,3,4},uo[]{0,2,5},wo[]{0,1,2,3,4};
    auto ssa=anndatapp::Array::borrow_sparse(anndatapp::Layout::csr,n,g,4,sparse_s,si,sparse_so);
    auto sua=anndatapp::Array::borrow_sparse(anndatapp::Layout::csc,n,g,5,sparse_u,ui,uo);
    auto swa=anndatapp::Array::borrow_sparse(anndatapp::Layout::csr,n,n,4,sparse_w,wi,wo);
    Acquired ssv(ssa.view()),suv(sua.view()),swv(swa.view());
    vp::Input sparse_input{ssv.matrix,suv.matrix,swv.matrix};
    for(int iteration=0;iteration<2;++iteration) {
        phase("sparse_validate",[&]{status=vp::validate(sparse_input);});require(bool(status),"sparse validate");
        phase("sparse_size",[&]{status=vp::preparation_size(sparse_input,ps);});
        require(bool(status)&&ps.indices<=256&&ps.entries<=256,"sparse preparation size");
        phase("sparse_prepare_capacity",[&]{vp::PreparedInput dummy;status=vp::prepare(sparse_input,{nullptr,0,nullptr,0},dummy);});
        require(status.code==vp::Code::capacity,"sparse preparation capacity error");
        phase("sparse_prepare",[&]{status=vp::prepare(sparse_input,pb,prepared);});require(bool(status),"sparse prepare");
        phase("sparse_neighborhood_size",[&]{status=vp::reference::neighborhood_size(swv.matrix,ns);});require(bool(status),"sparse neighborhood size");
        phase("sparse_neighborhood",[&]{status=vp::reference::prepare_neighborhood(swv.matrix,nb,used);});
        require(bool(status)&&used==n,"sparse neighborhood");
        phase("sparse_bind",[&]{status=vp::bind_workspace(prepared,run,true,storage,512,mask,n,workspace);});require(bool(status),"sparse bind");
        phase("sparse_second_moments",[&]{vp::reference::second_moments(prepared,0,g,ms,mu,mss,mus,muu,source_s,source_u);});
        phase("sparse_stochastic",[&]{status=vp::reference::stochastic_into(prepared,workspace,stochastic_result,velocity,n*g,residual,n*g,g,so_options);});
        require(bool(status),"sparse stochastic");
    }
    // Empty observations still size, bind and run without allocating.
    auto esa=anndatapp::Array::borrow_dense(static_cast<double*>(nullptr),{0,g},{ptrdiff_t(g),1});
    auto eua=anndatapp::Array::borrow_dense(static_cast<double*>(nullptr),{0,g},{ptrdiff_t(g),1});
    auto ewa=anndatapp::Array::borrow_dense(static_cast<double*>(nullptr),{0,0},{0,1});
    Acquired esv(esa.view()),euv(eua.view()),ewv(ewa.view());
    vp::Input empty_input{esv.matrix,euv.matrix,ewv.matrix};
    for(int iteration=0;iteration<2;++iteration) {
        phase("empty_validate",[&]{status=vp::validate(empty_input);});require(bool(status),"empty validate");
        phase("empty_size",[&]{status=vp::preparation_size(empty_input,ps);});require(bool(status),"empty preparation size");
        phase("empty_prepare",[&]{status=vp::prepare(empty_input,pb,prepared);});require(bool(status),"empty prepare");
        phase("empty_bind",[&]{status=vp::bind_workspace(prepared,run,false,storage,512,mask,n,workspace);});require(bool(status),"empty bind");
        phase("empty_moments",[&]{vp::reference::moments(prepared,0,g,static_cast<double*>(nullptr),static_cast<double*>(nullptr));});
        phase("empty_steady",[&]{status=vp::reference::steady_state_into(prepared,workspace,result,static_cast<double*>(nullptr),0,g,fit);});
        require(bool(status),"empty steady");
        phase("empty_stochastic_bind",[&]{status=vp::bind_workspace(prepared,run,true,storage,512,mask,n,workspace);});require(bool(status),"empty stochastic bind");
        phase("empty_stochastic",[&]{status=vp::reference::stochastic_into(prepared,workspace,stochastic_result,static_cast<double*>(nullptr),0,static_cast<double*>(nullptr),0,g,so_options);});
        require(bool(status),"empty stochastic");
    }
    std::puts("malloc/calloc/realloc/free, aligned allocation, new/delete: zero across fresh and repeated reference phases");
    return 0;
} catch(const std::exception& e) { tracking=false;std::fprintf(stderr,"%s\n",e.what());return 1; }

#endif // __GLIBC__
