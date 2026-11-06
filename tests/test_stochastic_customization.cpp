#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <cstdio>
import velocitypp;
#include "stochastic_support.hh"
void check(bool x) { if(!x) throw std::runtime_error("stochastic customization check failed"); }
struct Token {std::size_t first,count;};
struct Opaque {
    int init=0,blocks=0; std::size_t values=0; bool resolved=false;
    Token initial_moments(std::size_t f,std::size_t n){return {f,n};}
    Token initial_fit(Token t){++init;return t;}
    void accept_initialization(Token,Token){}
    void resolve_selection(){resolved=true;}
    Token moments(std::size_t f,std::size_t n){check(resolved);return {f,n};}
    Token scales(Token t){return t;}
    Token fit(Token t,Token){++blocks;return t;}
    void emit(Token t,Token){values+=t.count;}
    std::size_t finish(){return values;}
};
struct Custom {
    velocitypp::StochasticReferenceStages ref;
    // Deliberately padded row stride, two-gene blocks, and five rows.
    double s[20]{},u[20]{},ss[20]{},su[20]{};
    double a[2]{1,1},b[2]{1,1},g[2]{3,3},off[2]{0,0},off2[2]{0,0},obj[2]{0,0};
    std::uint8_t status[2]{0,0};
    int replaced=0, scale_calls=0, fit_calls=0, mode=3;
    velocitypp::MomentBlock initial_moments(std::size_t f,std::size_t n){return ref.initial_moments(f,n);}
    velocitypp::FitBlock initial_fit(velocitypp::MomentBlock m){return ref.initial_fit(m);}
    void accept_initialization(velocitypp::MomentBlock m,velocitypp::FitBlock f){ref.accept_initialization(m,f);}
    void resolve_selection(){ref.resolve_selection();}
    velocitypp::StochasticMomentBlock moments(std::size_t f,std::size_t n){
        if(mode!=0 && mode!=3) return ref.moments(f,n);
        ++replaced;
        for(std::size_t i=0;i<5;++i)for(std::size_t j=0;j<n;++j){auto k=4*i+j;s[k]=double(i+1);u[k]=6*s[k];ss[k]=s[k]*s[k];su[k]=s[k]*u[k];}
        return {f,n,5,4,s,u,ss,su,nullptr};
    }
    velocitypp::StochasticScaleBlock scales(velocitypp::StochasticMomentBlock m){
        if(mode!=1 && mode!=3) return ref.scales(m);
        ++scale_calls; return {m.first,m.count,a,b,status};
    }
    velocitypp::StochasticFitBlock fit(velocitypp::StochasticMomentBlock m,velocitypp::StochasticScaleBlock scales){
        if(mode!=2 && mode!=3) return ref.fit(m,scales);
        ++fit_calls;
        if(mode==2) for(std::size_t i=0;i<m.rows;++i)for(std::size_t j=0;j<m.count;++j){
            auto q=std::ptrdiff_t(i)*m.stride+std::ptrdiff_t(j);
            check(std::abs(m.spliced_second[q]-m.spliced[q]*m.spliced[q])<1e-12);
            check(std::abs(m.cross[q]-m.spliced[q]*m.unspliced[q])<1e-12);
        }
        return {m.first,m.count,g,off,off2,a,b,status,obj};
    }
    void emit(velocitypp::StochasticMomentBlock m,velocitypp::StochasticFitBlock f){ref.emit(m,f);}
    velocitypp::StochasticFitResult finish(){return ref.finish();}
};
struct Collector {int blocks=0;std::size_t values=0;};
void collect(void* p,const velocitypp::StochasticMomentBlock& m,const velocitypp::StochasticFitBlock& f){
    auto& c=*static_cast<Collector*>(p);++c.blocks;
    for(std::size_t i=0;i<m.rows;++i)for(std::size_t j=0;j<m.count;++j)
        check(std::abs(m.unspliced[i*m.stride+j]-f.gamma[j]*m.spliced[i*m.stride+j]-3*double(i+1))<1e-12);
    c.values+=m.count;
}
int main(){try{
    Opaque opaque;check(velocitypp::stochastic_with(opaque,5,2)==5 && opaque.init==3 && opaque.blocks==3);
    auto s=anndatapp::Array::empty<double>({5,5}),u=anndatapp::Array::empty<double>({5,5}),w=anndatapp::Array::zeros<double>({5,5});
    for(int i=0;i<5;++i){w.edit().dense<double>().data()[6*i]=1;for(int j=0;j<5;++j){s.edit().dense<double>().data()[5*i+j]=i+1;u.edit().dense<double>().data()[5*i+j]=2*(i+1)+.1*(i%2);}}
    PreparedStorage storage(s.view(),u.view(),w.view()); auto& input=storage.value;
    velocitypp::RunOptions run;run.block_genes=2;
    StochasticStorage<double> workspace(input,run);Collector collector;
    Custom custom{}; require_status(velocitypp::bind_stages(input,workspace.workspace,{},workspace.result,custom.ref,{&collector,collect}));
    auto replaced=velocitypp::stochastic_with(custom,input.genes,workspace.workspace.block);
    check(custom.replaced==3 && collector.blocks==3 && collector.values==5);
    for(int j=0;j<5;++j)check(replaced.gamma[j]==3);
    for(int mode=0;mode<3;++mode){
        velocitypp::StochasticOptions opts;
        // The custom fit consumes real reference second moments even though
        // built-in initialization rejects every gene for refinement.
        if(mode==2) opts.initialization.min_r2=2;
        Custom part{}; require_status(velocitypp::bind_stages(input,workspace.workspace,opts,workspace.result,part.ref)); part.mode=mode;
        auto result=velocitypp::stochastic_with(part,input.genes,workspace.workspace.block);
        check((mode!=0 || part.replaced==3) && (mode!=1 || part.scale_calls==3) && (mode!=2 || part.fit_calls==3));
        if(mode==2)for(int j=0;j<5;++j){
            check(!result.initialization.genes[j]);
            check(result.gamma[j]==3);
        }
    }
    auto baseline=workspace.run(input);
    for(int j=0;j<5;++j)check(std::abs(baseline.gamma[j]-3)>.1);
    std::puts("stochastic opaque stages, multiblock padded custom values, explicit reference passed");
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
