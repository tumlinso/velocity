module;
#include <stddef.h>
#include <stdint.h>
#include "detail/scientific.hh"

module velocitypp;

namespace velocitypp {
namespace {
size_t maximum(size_t a,size_t b) {return a<b?b:a;}
bool mul(size_t a,size_t b,size_t& r) {if(a && b>SIZE_MAX/a)return false;r=a*b;return true;}
bool add(size_t a,size_t b,size_t& r) {if(b>SIZE_MAX-a)return false;r=a+b;return true;}
Status check_output(const PreparedInput& p,const void* out,size_t capacity,ptrdiff_t stride,size_t field) {
    if(!p.rows || !p.genes)return {};
    size_t pitch=stride<0?size_t(-(stride+1))+1:size_t(stride),span=0;
    if(p.rows>1 && pitch<p.genes)return {Code::argument,field};
    if(p.genes-1>size_t(PTRDIFF_MAX) || !mul(p.rows-1,pitch,span) || span>size_t(PTRDIFF_MAX)-(p.genes-1) || !add(span,p.genes,span))return {Code::overflow,field};
    if(!out || capacity<span)return {Code::capacity,field,span};
    return {};
}
size_t gene(const PreparedInput& p,size_t k) {return p.input.genes?p.input.genes[k]:k;}
bool enabled(const PreparedInput& p,size_t k) {return !p.input.gene_mask || p.input.gene_mask[ptrdiff_t(gene(p,k))*p.input.gene_mask_stride]!=0;}
template<class T> bool result_valid(const PreparedInput& p,const BasicFitResult<T>& r) {
    return r.capacity>=p.genes && (!p.genes || (r.gamma && r.offset && r.r2 && r.genes && r.status));
}
template<class T> bool result_valid(const PreparedInput& p,const BasicStochasticFitResult<T>& r) {
    return result_valid(p,r.initialization) && r.capacity>=p.genes &&
        (!p.genes || (r.gamma && r.offset && r.offset2 && r.first_scale && r.second_scale && r.objective && r.status));
}
template<class T> void initialize(const PreparedInput& p,Workspace<T>& w,BasicFitResult<T>& r) {
    r.low_signal=false;
    for(size_t i=0;i<p.rows;++i) w.mask[i]=p.input.fit_mask?uint8_t(p.input.fit_mask[ptrdiff_t(i)*p.input.fit_mask_stride]!=0):1;
    for(size_t j=0;j<p.genes;++j) {r.gamma[j]=r.offset[j]=r.r2[j]=0;r.genes[j]=false;r.status[j]=uint8_t(FitStatus::insufficient_data);}
}
template<class T> size_t select(const PreparedInput& p,const FitOptions& o,BasicFitResult<T>& r,T* score) {
    size_t selected=0;for(size_t k=0;k<p.genes;++k)selected+=r.genes[k];
    if(o.low_signal==LowSignal::scvelo_percentile && selected<2) {
        size_t n=0;
        for(size_t k=0;k<p.genes;++k)if(r.status[k]==uint8_t(FitStatus::ok) && enabled(p,k))score[n++]=r.r2[k];
        if(n) {
            detail::scientific::HeapSort{}(score,n);
            T q=T(.8)*T(n-1);size_t lo=size_t(q),hi=lo+1<n?lo+1:n-1;
            T bound=score[lo]+(q-T(lo))*(score[hi]-score[lo]);
            selected=0;
            for(size_t k=0;k<p.genes;++k) {r.genes[k]=r.status[k]==uint8_t(FitStatus::ok) && enabled(p,k) && r.r2[k]>bound;selected+=r.genes[k];}
        }
        r.low_signal=true;
    }
    return selected;
}
template<class T> BasicGeneFit<T> initial(const BasicStochasticFitResult<T>& r,size_t k) {
    auto& f=r.initialization;return {f.gamma[k],f.offset[k],f.r2[k],FitStatus(f.status[k]),f.genes[k]};
}
template<class T> bool refine(const BasicStochasticReferenceStages<T>& s,size_t k) {
    return (s.result.initialization.genes[k] || s.result.refit_all) && enabled(*s.input,k) && s.result.initialization.status[k]==uint8_t(FitStatus::ok);
}
}

namespace reference {
Status validate(const FitOptions& o) {
    if(__builtin_isnan(o.ratio_min) || __builtin_isnan(o.ratio_max) || o.ratio_min>o.ratio_max ||
       !__builtin_isfinite(o.min_r2) || !__builtin_isfinite(o.min_ratio) ||
       !__builtin_isfinite(o.percentile_low) || !__builtin_isfinite(o.percentile_high) ||
       o.percentile_low<0 || o.percentile_high>100 || o.percentile_low>o.percentile_high ||
       (o.trimming!=Trimming::none && o.trimming!=Trimming::upper && o.trimming!=Trimming::tails) ||
       (o.low_signal!=LowSignal::keep_filters && o.low_signal!=LowSignal::scvelo_percentile))return {Code::argument};
    return {};
}
Status validate(const StochasticOptions& o) {return validate(o.initialization);}
}

template<class T> Status workspace_size(const PreparedInput& p,RunOptions o,bool stochastic,WorkspaceSize& out) {
    out={};if(!o.block_genes)return {Code::argument};
    size_t fixed=stochastic?p.rows:maximum(p.rows,p.genes),extra=0,per_gene=0,bytes=0;
    if(stochastic && (!mul(p.rows,2,extra) || !add(fixed,extra,fixed)))return {Code::overflow};
    if(!mul(p.rows,stochastic?4+size_t(o.unspliced_second):2,per_gene) || !mul(per_gene,sizeof(T),bytes))return {Code::overflow};
    size_t fixed_bytes=0;if(!mul(fixed,sizeof(T),fixed_bytes) || !add(fixed_bytes,p.rows,fixed_bytes))return {Code::overflow};
    if(fixed_bytes>o.workspace_bytes)return {Code::capacity,0,fixed_bytes};
    size_t block=p.genes?p.genes:1;if(block>o.block_genes)block=o.block_genes;
    if(bytes && block>(o.workspace_bytes-fixed_bytes)/bytes)block=(o.workspace_bytes-fixed_bytes)/bytes;
    if(!block) {size_t need=0;if(!add(fixed_bytes,bytes,need))return {Code::overflow};return {Code::capacity,0,need};}
    size_t scalar_count=0;if(!mul(per_gene,block,scalar_count) || !add(scalar_count,fixed,scalar_count) ||
       !mul(scalar_count,sizeof(T),bytes) || !add(bytes,p.rows,bytes))return {Code::overflow};
    out={scalar_count,p.rows,block,bytes};return {};
}
template<class T> Status bind_workspace(const PreparedInput& p,RunOptions o,bool stochastic,T* storage,size_t capacity,uint8_t* mask,size_t mask_capacity,Workspace<T>& w) {
    WorkspaceSize n;auto e=workspace_size<T>(p,o,stochastic,n);if(!e)return e;
    if(capacity<n.scalars || (n.scalars && !storage))return {Code::capacity,0,n.scalars};
    if(mask_capacity<n.masks || (n.masks && !mask))return {Code::capacity,1,n.masks};
    size_t offset=0,block=p.rows*n.block;
    auto take=[&](size_t count){T* q=storage?storage+offset:nullptr;offset+=count;return q;};
    w={};w.ms=take(block);w.mu=take(block);w.score=take(stochastic?p.rows:maximum(p.rows,p.genes));
    if(stochastic) {w.mss=take(block);w.mus=take(block);if(o.unspliced_second)w.muu=take(block);w.source_s=take(p.rows);w.source_u=take(p.rows);}
    w.mask=mask;w.rows=p.rows;w.block=n.block;w.scratch_bytes=n.bytes;w.stochastic=stochastic;return {};
}
template<class T> Status bind_stages(const PreparedInput& p,Workspace<T>& w,FitOptions o,BasicFitResult<T> r,BasicReferenceStages<T>& s,BasicBlockSink<T> sink) {
    auto e=reference::validate(o);if(!e)return e;
    if(!w.block || w.rows!=p.rows || (p.rows && (!w.ms || !w.mu || !w.mask || !w.score)))return {Code::argument};
    if(!result_valid(p,r))return {Code::capacity,0,p.genes};
    if(sink.context && !sink.write)return {Code::argument};
    initialize(p,w,r);s={&p,&w,o,r,sink,0,false};return {};
}
template<class T> BasicMomentBlock<T> BasicReferenceStages<T>::moments(size_t first,size_t count) {
    reference::moments(*input,first,count,workspace->ms,workspace->mu);
    return {first,count,input->rows,ptrdiff_t(count),workspace->ms,workspace->mu};
}
template<class T> BasicFitBlock<T> BasicReferenceStages<T>::fit(BasicMomentBlock<T> m) {
    for(size_t j=0;j<m.count;++j) {
        size_t k=m.first+j;auto f=reference::fit_gene(m.spliced?m.spliced+j:nullptr,m.unspliced?m.unspliced+j:nullptr,
            m.rows,m.stride,options,workspace->score,workspace->mask,enabled(*input,k));
        result.gamma[k]=f.gamma;result.offset[k]=f.offset;result.r2[k]=f.r2;result.genes[k]=f.selected;result.status[k]=uint8_t(f.status);
    }
    return {m.first,m.count,result.gamma+m.first,result.offset+m.first,result.r2+m.first,result.genes+m.first,result.status+m.first};
}
template<class T> void BasicReferenceStages<T>::emit(BasicMomentBlock<T> m,BasicFitBlock<T> f) {
    for(size_t j=0;j<m.count;++j) {size_t k=m.first+j;
        result.gamma[k]=f.gamma[j];result.offset[k]=f.offset[j];result.r2[k]=f.r2[j];result.status[k]=f.status[j];
        result.genes[k]=f.selected[j] && f.status[j]==uint8_t(FitStatus::ok) && enabled(*input,k);}
    if(sink.write)sink.write(sink.context,m,f);
    next+=m.count;
}
template<class T> BasicFitResult<T> BasicReferenceStages<T>::finish() {
    select(*input,options,result,workspace->score);finished=true;return result;
}
template<class T> Status bind_stages(const PreparedInput& p,Workspace<T>& w,StochasticOptions o,BasicStochasticFitResult<T> r,BasicStochasticReferenceStages<T>& s,BasicStochasticBlockSink<T> sink) {
    auto e=reference::validate(o);if(!e)return e;
    if(!w.stochastic || !w.block || w.rows!=p.rows || (p.rows && (!w.ms || !w.mu || !w.mss || !w.mus || !w.source_s || !w.source_u || !w.score || !w.mask)))return {Code::argument};
    if(!result_valid(p,r))return {Code::capacity,0,p.genes};
    if(sink.context && !sink.write)return {Code::argument};
    initialize(p,w,r.initialization);r.refit_all=false;
    for(size_t k=0;k<p.genes;++k) {r.gamma[k]=r.offset[k]=r.offset2[k]=r.first_scale[k]=r.second_scale[k]=r.objective[k]=0;r.status[k]=uint8_t(StochasticStatus::not_selected);}
    s={&p,&w,o,r,sink,0,false,false};return {};
}
template<class T> BasicMomentBlock<T> BasicStochasticReferenceStages<T>::initial_moments(size_t f,size_t n) {
    reference::moments(*input,f,n,workspace->ms,workspace->mu);
    return {f,n,input->rows,ptrdiff_t(n),workspace->ms,workspace->mu};
}
template<class T> BasicFitBlock<T> BasicStochasticReferenceStages<T>::initial_fit(BasicMomentBlock<T> m) {
    auto& r=result.initialization;
    for(size_t j=0;j<m.count;++j) {size_t k=m.first+j;
        auto f=reference::fit_gene(m.spliced?m.spliced+j:nullptr,m.unspliced?m.unspliced+j:nullptr,m.rows,m.stride,
            options.initialization,workspace->score,workspace->mask,enabled(*input,k));
        r.gamma[k]=f.gamma;r.offset[k]=f.offset;r.r2[k]=f.r2;r.genes[k]=f.selected;r.status[k]=uint8_t(f.status);}
    return {m.first,m.count,r.gamma+m.first,r.offset+m.first,r.r2+m.first,r.genes+m.first,r.status+m.first};
}
template<class T> void BasicStochasticReferenceStages<T>::accept_initialization(BasicMomentBlock<T> m,BasicFitBlock<T> f) {
    auto& r=result.initialization;
    for(size_t j=0;j<m.count;++j) {size_t k=m.first+j;r.gamma[k]=f.gamma[j];r.offset[k]=f.offset[j];r.r2[k]=f.r2[j];r.status[k]=f.status[j];
        r.genes[k]=f.selected[j] && f.status[j]==uint8_t(FitStatus::ok) && enabled(*input,k);}
    next+=m.count;
}
template<class T> void BasicStochasticReferenceStages<T>::resolve_selection() {
    size_t selected=select(*input,options.initialization,result.initialization,result.objective);
    result.refit_all=options.refit_all_if_none && !selected;
    for(size_t k=0;k<input->genes;++k) {result.gamma[k]=result.initialization.gamma[k];result.offset[k]=result.initialization.offset[k];result.objective[k]=0;
        result.status[k]=uint8_t(!enabled(*input,k) || result.initialization.status[k]==uint8_t(FitStatus::ok)?StochasticStatus::not_selected:StochasticStatus::invalid_initialization);}
    resolved=true;next=0;
}
template<class T> BasicStochasticMomentBlock<T> BasicStochasticReferenceStages<T>::moments(size_t f,size_t n) {
    auto& w=*workspace;reference::second_moments(*input,f,n,w.ms,w.mu,w.mss,w.mus,w.muu,w.source_s,w.source_u);
    return {f,n,input->rows,ptrdiff_t(n),w.ms,w.mu,w.mss,w.mus,w.muu};
}
template<class T> BasicStochasticScaleBlock<T> BasicStochasticReferenceStages<T>::scales(BasicStochasticMomentBlock<T> m) {
    for(size_t j=0;j<m.count;++j) {size_t k=m.first+j;if(!refine(*this,k))continue;
        auto s=reference::stochastic_scales(m.spliced?m.spliced+j:nullptr,m.unspliced?m.unspliced+j:nullptr,
            m.spliced_second?m.spliced_second+j:nullptr,m.cross?m.cross+j:nullptr,m.rows,m.stride,initial(result,k),options,workspace->score,workspace->mask);
        result.first_scale[k]=s.first;result.second_scale[k]=s.second;result.status[k]=uint8_t(s.status);}
    return {m.first,m.count,result.first_scale+m.first,result.second_scale+m.first,result.status+m.first};
}
template<class T> BasicStochasticFitBlock<T> BasicStochasticReferenceStages<T>::fit(BasicStochasticMomentBlock<T> m,BasicStochasticScaleBlock<T> scales) {
    for(size_t j=0;j<m.count;++j) {size_t k=m.first+j;if(!refine(*this,k))continue;
        auto f=reference::stochastic_fit_gene(m.spliced?m.spliced+j:nullptr,m.unspliced?m.unspliced+j:nullptr,
            m.spliced_second?m.spliced_second+j:nullptr,m.cross?m.cross+j:nullptr,m.rows,m.stride,initial(result,k),
            {scales.first_scale[j],scales.second_scale[j],StochasticStatus(scales.status[j])},options,workspace->score,workspace->mask);
        result.gamma[k]=f.gamma;result.offset[k]=f.offset;result.offset2[k]=f.offset2;result.first_scale[k]=f.scales.first;result.second_scale[k]=f.scales.second;
        result.status[k]=uint8_t(f.status);result.objective[k]=f.objective;}
    return {m.first,m.count,result.gamma+m.first,result.offset+m.first,result.offset2+m.first,result.first_scale+m.first,result.second_scale+m.first,result.status+m.first,result.objective+m.first};
}
template<class T> void BasicStochasticReferenceStages<T>::emit(BasicStochasticMomentBlock<T> m,BasicStochasticFitBlock<T> f) {
    for(size_t j=0;j<m.count;++j) {size_t k=m.first+j;result.gamma[k]=f.gamma[j];result.offset[k]=f.offset[j];result.offset2[k]=f.offset2[j];
        result.first_scale[k]=f.first_scale[j];result.second_scale[k]=f.second_scale[j];result.status[k]=f.status[j];result.objective[k]=f.objective[j];}
    if(sink.write)sink.write(sink.context,m,f);
    next+=m.count;
}
template<class T> BasicStochasticFitResult<T> BasicStochasticReferenceStages<T>::finish() {finished=true;return result;}

namespace reference {
template<class T,class O> void residuals(const BasicMomentBlock<T>& m,const BasicFitBlock<T>& f,O* out,ptrdiff_t sr,ptrdiff_t sc) {
    for(size_t i=0;i<m.rows;++i)for(size_t j=0;j<m.count;++j) {ptrdiff_t k=ptrdiff_t(i)*m.stride+ptrdiff_t(j);out[ptrdiff_t(i)*sr+ptrdiff_t(j)*sc]=O(m.unspliced[k]-f.gamma[j]*m.spliced[k]-f.offset[j]);}
}
template<class T,class O> void stochastic_residuals(const BasicStochasticMomentBlock<T>& m,const BasicStochasticFitBlock<T>& f,O* out,O* second,ptrdiff_t sr,ptrdiff_t sc) {
    for(size_t i=0;i<m.rows;++i)for(size_t j=0;j<m.count;++j) {
        ptrdiff_t k=ptrdiff_t(i)*m.stride+ptrdiff_t(j),q=ptrdiff_t(i)*sr+ptrdiff_t(j)*sc;T x=m.spliced[k],y=m.unspliced[k];
        if(out)out[q]=O(y-f.gamma[j]*x-f.offset[j]);
        if(second)second[q]=O(f.status[j]==uint8_t(StochasticStatus::ok)?
            (2*m.cross[k]+y-2*x*y)-f.gamma[j]*(2*m.spliced_second[k]-x-2*x*x)+2*f.offset[j]*x-f.offset2[j]:T(0));
    }
}
template<class T> Status steady_state(const PreparedInput& p,Workspace<T>& w,BasicFitResult<T>& r,FitOptions o,BasicBlockSink<T> sink) {
    BasicReferenceStages<T> s;auto e=bind_stages(p,w,o,r,s,sink);if(!e)return e;
    r=steady_state_with(s,p.genes,w.block);return {};
}
template<class T,class O> Status steady_state_into(const PreparedInput& p,Workspace<T>& w,BasicFitResult<T>& r,O* out,size_t capacity,ptrdiff_t stride,FitOptions o) {
    auto output=check_output(p,out,capacity,stride,0);if(!output)return output;
    BasicReferenceStages<T> s;auto e=bind_stages(p,w,o,r,s);if(!e)return e;
    for(size_t first=0;first<p.genes;) {size_t count=p.genes-first<w.block?p.genes-first:w.block;
        auto m=s.moments(first,count);auto f=s.fit(m);residuals(m,f,out?out+first:nullptr,stride);s.emit(m,f);first+=count;}
    r=s.finish();return {};
}
template<class T> Status stochastic(const PreparedInput& p,Workspace<T>& w,BasicStochasticFitResult<T>& r,StochasticOptions o,BasicStochasticBlockSink<T> sink) {
    BasicStochasticReferenceStages<T> s;auto e=bind_stages(p,w,o,r,s,sink);if(!e)return e;
    r=stochastic_with(s,p.genes,w.block);return {};
}
template<class T,class O> Status stochastic_into(const PreparedInput& p,Workspace<T>& w,BasicStochasticFitResult<T>& r,O* out,size_t capacity,O* second,size_t second_capacity,ptrdiff_t stride,StochasticOptions o) {
    if(!out && !second && p.rows && p.genes)return {Code::argument};
    if(out){auto e=check_output(p,out,capacity,stride,0);if(!e)return e;}
    if(second){auto e=check_output(p,second,second_capacity,stride,1);if(!e)return e;}
    BasicStochasticReferenceStages<T> s;auto e=bind_stages(p,w,o,r,s);if(!e)return e;
    for(size_t first=0;first<p.genes;) {size_t count=p.genes-first<w.block?p.genes-first:w.block;
        auto m=s.initial_moments(first,count);auto f=s.initial_fit(m);s.accept_initialization(m,f);first+=count;}
    s.resolve_selection();
    for(size_t first=0;first<p.genes;) {size_t count=p.genes-first<w.block?p.genes-first:w.block;
        auto m=s.moments(first,count);auto scales=s.scales(m);auto f=s.fit(m,scales);
        stochastic_residuals(m,f,out?out+first:nullptr,second?second+first:nullptr,stride);s.emit(m,f);first+=count;}
    r=s.finish();return {};
}
}
namespace cpu {
template<class T> Status steady_state(const PreparedInput& p,Workspace<T>& w,BasicFitResult<T>& r,FitOptions o,BasicBlockSink<T> sink) {return reference::steady_state(p,w,r,o,sink);}
template<class T> Status stochastic(const PreparedInput& p,Workspace<T>& w,BasicStochasticFitResult<T>& r,StochasticOptions o,BasicStochasticBlockSink<T> sink) {return reference::stochastic(p,w,r,o,sink);}
}
namespace {
Status dispatch(const RuntimeOperations& o,size_t genes,size_t block,bool stochastic) {
    if(o.run)return o.run(o.context);
    if(!block || (!o.process_block && (!o.moments || !o.fit || !o.emit)))return {Code::argument};
    auto invoke=[&](auto f,size_t first,size_t count){return f?f(o.context,first,count):Status{};};
    if(o.begin){auto e=o.begin(o.context);if(!e)return e;}
    if(stochastic) {
        if(o.initialize) {auto e=o.initialize(o.context);if(!e)return e;}
        else {
            if(!o.initial_block || !o.resolve)return {Code::argument};
            for(size_t first=0;first<genes;) {size_t n=genes-first<block?genes-first:block;auto e=invoke(o.initial_block,first,n);if(!e)return e;first+=n;}
            auto e=o.resolve(o.context);if(!e)return e;
        }
    }
    for(size_t first=0;first<genes;) {size_t n=genes-first<block?genes-first:block;
        if(o.process_block){auto e=invoke(o.process_block,first,n);if(!e)return e;}
        else {auto e=invoke(o.moments,first,n);if(!e)return e;if(o.scales){e=invoke(o.scales,first,n);if(!e)return e;}e=invoke(o.fit,first,n);if(!e)return e;e=invoke(o.emit,first,n);if(!e)return e;}
        first+=n;}
    return o.finish?o.finish(o.context):Status{};
}
}
Status steady_state_with(const RuntimeOperations& o,size_t g,size_t b){return dispatch(o,g,b,false);}
Status stochastic_with(const RuntimeOperations& o,size_t g,size_t b){return dispatch(o,g,b,true);}

#define VELOCITYPP_INSTANTIATE(T) \
template Status workspace_size<T>(const PreparedInput&,RunOptions,bool,WorkspaceSize&); \
template Status bind_workspace<T>(const PreparedInput&,RunOptions,bool,T*,size_t,uint8_t*,size_t,Workspace<T>&); \
template Status bind_stages<T>(const PreparedInput&,Workspace<T>&,FitOptions,BasicFitResult<T>,BasicReferenceStages<T>&,BasicBlockSink<T>); \
template Status bind_stages<T>(const PreparedInput&,Workspace<T>&,StochasticOptions,BasicStochasticFitResult<T>,BasicStochasticReferenceStages<T>&,BasicStochasticBlockSink<T>); \
template struct BasicReferenceStages<T>;template struct BasicStochasticReferenceStages<T>; \
template Status reference::steady_state<T>(const PreparedInput&,Workspace<T>&,BasicFitResult<T>&,FitOptions,BasicBlockSink<T>); \
template Status reference::stochastic<T>(const PreparedInput&,Workspace<T>&,BasicStochasticFitResult<T>&,StochasticOptions,BasicStochasticBlockSink<T>); \
template Status cpu::steady_state<T>(const PreparedInput&,Workspace<T>&,BasicFitResult<T>&,FitOptions,BasicBlockSink<T>); \
template Status cpu::stochastic<T>(const PreparedInput&,Workspace<T>&,BasicStochasticFitResult<T>&,StochasticOptions,BasicStochasticBlockSink<T>);
VELOCITYPP_INSTANTIATE(float)
VELOCITYPP_INSTANTIATE(double)
#undef VELOCITYPP_INSTANTIATE
#define VELOCITYPP_OUTPUT(T,O) \
template void reference::residuals<T,O>(const BasicMomentBlock<T>&,const BasicFitBlock<T>&,O*,ptrdiff_t,ptrdiff_t); \
template void reference::stochastic_residuals<T,O>(const BasicStochasticMomentBlock<T>&,const BasicStochasticFitBlock<T>&,O*,O*,ptrdiff_t,ptrdiff_t); \
template Status reference::steady_state_into<T,O>(const PreparedInput&,Workspace<T>&,BasicFitResult<T>&,O*,size_t,ptrdiff_t,FitOptions); \
template Status reference::stochastic_into<T,O>(const PreparedInput&,Workspace<T>&,BasicStochasticFitResult<T>&,O*,size_t,O*,size_t,ptrdiff_t,StochasticOptions);
VELOCITYPP_OUTPUT(float,float)
VELOCITYPP_OUTPUT(float,double)
VELOCITYPP_OUTPUT(double,float)
VELOCITYPP_OUTPUT(double,double)
#undef VELOCITYPP_OUTPUT

}
