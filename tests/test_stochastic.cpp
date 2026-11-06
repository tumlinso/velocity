#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

import velocitypp;
#include "stochastic_support.hh"

namespace {
int failures = 0;
void expect(bool ok, const char* what) {
    if (ok) return;
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
}
bool close(double a, double b, double tol = 2e-6) {
    return std::abs(a - b) <= tol*(1 + std::abs(b));
}
struct Sparse {
    anndatapp::Layout layout;
    std::size_t rows, cols;
    std::vector<double> values;
    std::vector<std::int32_t> indices;
    std::vector<std::int64_t> offsets;
    anndatapp::Array array() {
        return anndatapp::Array::borrow_sparse(layout, rows, cols, values.size(),
            values.data(), indices.data(), offsets.data());
    }
};
Sparse compressed(anndatapp::Layout layout, std::size_t rows, std::size_t cols,
                  const std::vector<double>& dense) {
    Sparse a{layout, rows, cols};
    auto major = layout == anndatapp::Layout::csr ? rows : cols;
    auto minor = layout == anndatapp::Layout::csr ? cols : rows;
    a.offsets.push_back(0);
    for (std::size_t i = 0; i < major; ++i) {
        for (std::size_t j = 0; j < minor; ++j) {
            auto r = layout == anndatapp::Layout::csr ? i : j;
            auto c = layout == anndatapp::Layout::csr ? j : i;
            auto x = dense[r*cols + c];
            if (x == 0) continue;
            a.values.push_back(x);
            a.indices.push_back(std::int32_t(j));
        }
        a.offsets.push_back(std::int64_t(a.values.size()));
    }
    return a;
}
velocitypp::FitOptions initialization() {
    velocitypp::FitOptions o;
    o.trimming = velocitypp::Trimming::none;
    o.fit_intercept = true;
    o.min_r2 = 0;
    o.min_ratio = 0;
    return o;
}
velocitypp::StochasticOptions options() {
    velocitypp::StochasticOptions o;
    o.initialization = initialization();
    return o;
}
struct Fixture {
    std::size_t n = 8, g = 2;
    std::vector<double> s, u, w;
    anndatapp::Array sa, ua, wa;
    PreparedStorage storage;
    velocitypp::PreparedInput& prepared;
    Fixture()
        : s{1,3, 2,4, 1,2, 3,7, 2,5, 5,8, 4,10, 7,11},
          u{2,5, 5,8, 1,5, 8,12, 3,13, 8,13, 10,17, 13,20},
          w(n*n), sa(anndatapp::Array::borrow_dense(s.data(), {n,g}, {std::ptrdiff_t(g),1})),
          ua(anndatapp::Array::borrow_dense(u.data(), {n,g}, {std::ptrdiff_t(g),1})),
          wa([&] {
              for (std::size_t i=0;i<n;++i) w[i*n+i]=1;
              return anndatapp::Array::borrow_dense(w.data(), {n,n}, {std::ptrdiff_t(n),1});
          }()), storage(sa.view(),ua.view(),wa.view()), prepared(storage.value) {}
};

auto dense_values(const anndatapp::Array& a) {
    auto view=a.view(); auto v=view.dense<double>();
    if(view.rank()==1) {
        std::vector<double> x(view.extent(0));
        for(std::size_t i=0;i<x.size();++i) x[i]=v.data()[std::ptrdiff_t(i)*v.stride(0)];
        return x;
    }
    std::vector<double> x(view.extent(0)*view.extent(1));
    for(std::size_t i=0;i<view.extent(0);++i) for(std::size_t j=0;j<view.extent(1);++j)
        x[i*view.extent(1)+j]=v.data()[std::ptrdiff_t(i)*v.stride(0)+std::ptrdiff_t(j)*v.stride(1)];
    return x;
}
void sparse_moments_coalesce_expression_duplicates() {
    constexpr std::size_t n=2,g=1;
    const double sv[]{1,2,2}, uv[]{4,1,4};
    const std::int32_t si[]{0,0,0}, ui[]{0,0,1};
    const std::int64_t so[]{0,2,3}, uo[]{0,3};
    auto s=anndatapp::Array::borrow_sparse(anndatapp::Layout::csr,n,g,3,sv,si,so);
    auto u=anndatapp::Array::borrow_sparse(anndatapp::Layout::csc,n,g,3,uv,ui,uo);
    auto w=anndatapp::Array::zeros<double>({n,n});
    auto* wp=w.edit().dense<double>().data(); wp[0]=1;wp[3]=1;
    PreparedStorage ps(s.view(),u.view(),w.view()); auto& p=ps.value;
    double ms[n]{},mu[n]{},mss[n]{},mus[n]{},muu[n]{},scratch_s[n]{},scratch_u[n]{};
    velocitypp::reference::second_moments(p,0,1,ms,mu,mss,mus,muu,scratch_s,scratch_u);
    expect(close(ms[0],3)&&close(mu[0],5)&&close(mss[0],9)&&close(mus[0],15)&&close(muu[0],25)&&
           close(ms[1],2)&&close(mu[1],4)&&close(mss[1],4)&&close(mus[1],8)&&close(muu[1],16),
           "CSR/CSC second moments combine duplicate logical expression entries before products");
}
void dense_csr_csc_and_signed_stride_match() {
    Fixture f;
    std::vector<double> ms(f.n*f.g),mu(f.n*f.g),mss(f.n*f.g),mus(f.n*f.g),muu(f.n*f.g),ss(f.n),su(f.n);
    velocitypp::reference::second_moments(f.prepared,0,f.g,ms.data(),mu.data(),mss.data(),mus.data(),muu.data(),ss.data(),su.data());
    bool ok=true;
    for(std::size_t i=0;i<f.n;++i) for(std::size_t j=0;j<f.g;++j) {
        auto k=i*f.g+j;
        ok &= close(ms[k],f.s[k])&&close(mu[k],f.u[k])&&close(mss[k],f.s[k]*f.s[k])&&
              close(mus[k],f.s[k]*f.u[k])&&close(muu[k],f.u[k]*f.u[k]);
    }
    expect(ok,"dense identity graph returns all five requested raw moments");

    auto sc=compressed(anndatapp::Layout::csr,f.n,f.g,f.s);
    auto uc=compressed(anndatapp::Layout::csc,f.n,f.g,f.u);
    auto wc=compressed(anndatapp::Layout::csc,f.n,f.n,f.w);
    // Fixture W is populated by construction in its array initializer.
    auto sa=sc.array(),ua=uc.array(),wa=wc.array();
    PreparedStorage sparse_storage(sa.view(),ua.view(),wa.view()); auto& sparse=sparse_storage.value;
    std::fill(ms.begin(),ms.end(),0);std::fill(mu.begin(),mu.end(),0);std::fill(mss.begin(),mss.end(),0);
    std::fill(mus.begin(),mus.end(),0);std::fill(muu.begin(),muu.end(),0);
    velocitypp::reference::second_moments(sparse,0,f.g,ms.data(),mu.data(),mss.data(),mus.data(),muu.data(),ss.data(),su.data());
    bool sparse_ok=true;
    for(std::size_t i=0;i<f.n*f.g;++i) sparse_ok &= close(ms[i],f.s[i])&&close(mu[i],f.u[i])&&close(mss[i],f.s[i]*f.s[i])&&close(mus[i],f.s[i]*f.u[i])&&close(muu[i],f.u[i]*f.u[i]);
    expect(sparse_ok,"CSR/CSC layers with opposite sparsity patterns match dense moments");

    std::vector<double> sr(f.s.size()),ur(f.u.size());
    for(std::size_t i=0;i<f.n;++i) for(std::size_t j=0;j<f.g;++j) {
        sr[i*f.g+j]=f.s[(f.n-1-i)*f.g+j]; ur[i*f.g+j]=f.u[(f.n-1-i)*f.g+j];
    }
    auto srev=anndatapp::Array::borrow_dense(sr.data()+(f.n-1)*f.g,{f.n,f.g},{-std::ptrdiff_t(f.g),1});
    auto urev=anndatapp::Array::borrow_dense(ur.data()+(f.n-1)*f.g,{f.n,f.g},{-std::ptrdiff_t(f.g),1});
    PreparedStorage reversed_storage(srev.view(),urev.view(),f.wa.view()); auto& reversed=reversed_storage.value;
    std::fill(ms.begin(),ms.end(),0);std::fill(mu.begin(),mu.end(),0);std::fill(mss.begin(),mss.end(),0);
    std::fill(mus.begin(),mus.end(),0);std::fill(muu.begin(),muu.end(),0);
    velocitypp::reference::second_moments(reversed,0,f.g,ms.data(),mu.data(),mss.data(),mus.data(),muu.data(),ss.data(),su.data());
    expect(ms==f.s&&mu==f.u,"signed dense row strides preserve logical second-moment row order");
}
void stochastic_path_and_outputs() {
    Fixture f;
    velocitypp::RunOptions ro; ro.block_genes=1;
    StochasticStorage<double> ws(f.prepared,ro);
    expect(ws.workspace.block==1&&ws.workspace.scratch_bytes<=ro.workspace_bytes,
           "stochastic workspace obeys one-gene blocks and budget");
    std::vector<double> velocity(f.n*f.g), residual2(f.n*f.g);
    require_status(velocitypp::reference::stochastic_into(f.prepared,ws.workspace,ws.result,velocity.data(),velocity.size(),residual2.data(),residual2.size(),f.g,options()));
    auto& result=ws.result; auto& gamma=ws.gamma; auto& offset=ws.offset; auto& offset2=ws.offset2; auto& status=ws.status;
    bool outputs_ok=true;
    for(std::size_t j=0;j<f.g;++j) {
        auto st=static_cast<velocitypp::StochasticStatus>(status[j]);
        outputs_ok &= std::isfinite(gamma[j])&&std::isfinite(offset[j])&&std::isfinite(offset2[j]);
        if(st==velocitypp::StochasticStatus::ok) {
            for(std::size_t i=0;i<f.n;++i) {
                auto k=i*f.g+j;
                outputs_ok &= close(velocity[k],f.u[k]-gamma[j]*f.s[k]-offset[j]);
                double A=2*f.s[k]*f.s[k]-f.s[k], B=2*f.s[k]*f.u[k]+f.u[k];
                outputs_ok &= close(residual2[k],B-2*f.s[k]*f.u[k]-gamma[j]*(A-2*f.s[k]*f.s[k])+2*offset[j]*f.s[k]-offset2[j]);
            }
        }
    }
    expect(outputs_ok,"stochastic caller-owned velocity and second residual follow documented equations and remain finite");
    expect(result.initialization.capacity==f.g,"result retains deterministic initialization diagnostics");

    velocitypp::RunOptions wide; wide.block_genes=64;
    StochasticStorage<double> ww(f.prepared,wide); ww.run(f.prepared,options());
    bool same=true;
    for(std::size_t j=0;j<f.g;++j) same &= close(gamma[j],ww.gamma[j]);
    expect(same,"stochastic results are independent of gene block width");
}
void typed_workspace_and_optional_residual() {
    Fixture f;
    StochasticStorage<double> d(f.prepared); d.run(f.prepared,options());
    StochasticStorage<float> x(f.prepared);
    std::vector<float> velocity(f.n*f.g);
    require_status(velocitypp::reference::stochastic_into(f.prepared,x.workspace,x.result,velocity.data(),velocity.size(),static_cast<float*>(nullptr),0,f.g,options()));
    bool same=true;
    for(std::size_t j=0;j<f.g;++j) {
        same &= x.status[j]==d.status[j] && close(x.gamma[j],d.gamma[j],2e-4) && close(x.offset[j],d.offset[j],2e-4);
        for(std::size_t i=0;i<f.n;++i) if(x.status[j]==std::uint8_t(velocitypp::StochasticStatus::ok))
            same &= close(velocity[i*f.g+j],f.u[i*f.g+j]-x.gamma[j]*f.s[i*f.g+j]-x.offset[j],2e-4);
    }
    expect(same,"float and double stochastic fits agree and second residual is optional");
    std::vector<float> mixed(f.n*f.g,-42);
    require_status(velocitypp::reference::stochastic_into(f.prepared,d.workspace,d.result,mixed.data(),mixed.size(),static_cast<float*>(nullptr),0,f.g,options()));
    for(std::size_t i=0;i<mixed.size();++i) same &= close(mixed[i],velocity[i],2e-4);
    expect(same,"double computation writes float output");
    std::fill(mixed.begin(),mixed.end(),-42);
    auto fail=velocitypp::reference::stochastic_into(f.prepared,d.workspace,d.result,mixed.data(),mixed.size()-1,static_cast<float*>(nullptr),0,f.g,options());
    bool untouched=fail.code==velocitypp::Code::capacity;
    for(float v:mixed) untouched &= v==-42;
    expect(untouched,"insufficient residual capacity reports failure without output writes");
}
void ratio_bounds_recompute_intercept_and_zero_scale_falls_back() {
    Fixture f;
    StochasticStorage<double> workspace(f.prepared);
    auto bound=options(); bound.initialization.ratio_max=1.;
    auto& result=workspace.run(f.prepared,bound);
    auto& gamma=workspace.gamma; auto& offset=workspace.offset; auto& status=workspace.status;
    bool ok=true;
    for(std::size_t j=0;j<f.g;++j) {
        auto st=static_cast<velocitypp::StochasticStatus>(status[j]);
        if(st==velocitypp::StochasticStatus::ok) {
            double mx=0,my=0;
            for(std::size_t i=0;i<f.n;++i) { auto k=i*f.g+j; mx+=f.s[k];my+=f.u[k]; }
            mx/=f.n;my/=f.n;
            ok &= close(gamma[j],1.)&&close(offset[j],my-mx);
        }
    }
    expect(ok,"active slope upper bound recomputes the enabled first-block intercept");

    constexpr std::size_t n=4;
    const double sdata[]{1,2,3,4}, udata[]{2,4,6,8};
    auto s=anndatapp::Array::borrow_dense(sdata,{n,1},{1,1});
    auto u=anndatapp::Array::borrow_dense(udata,{n,1},{1,1});
    auto w=anndatapp::Array::zeros<double>({n,n});
    auto* wp=w.edit().dense<double>().data(); for(std::size_t i=0;i<n;++i)wp[i*n+i]=1;
    PreparedStorage ps(s.view(),u.view(),w.view()); auto& p=ps.value;
    StochasticStorage<double> ws(p);
    std::vector<double> ev(n),er(n);
    require_status(velocitypp::reference::stochastic_into(p,ws.workspace,ws.result,ev.data(),ev.size(),er.data(),er.size(),1,options()));
    auto& exact_status=ws.status; auto& exact_gamma=ws.gamma; auto& exact_offset=ws.offset;
    bool fallback=static_cast<velocitypp::StochasticStatus>(exact_status[0])==velocitypp::StochasticStatus::zero_scale&&
        close(exact_gamma[0],2.)&&close(exact_offset[0],0.);
    for(double x:ev) fallback &= close(x,0.);
    for(double x:er) fallback &= close(x,0.);
    expect(fallback,"zero residual scale preserves deterministic parameters and zeroes second residual output");
}

void validation_reports_nonfinite_input() {
    Fixture f;
    auto old=f.s[0]; f.s[0]=std::numeric_limits<double>::quiet_NaN();
    expect(velocitypp::validate(f.prepared.input).code==velocitypp::Code::nonfinite,
           "explicit validation catches borrowed input mutation");
    f.s[0]=old;
}

void overflowed_primitive_stages_keep_finite_fallbacks() {
    const double ms[]{1e100,1e100},mu[]{1e100,1e100};
    const double mss[]{1e200,1e200},mus[]{2e200,2e200};
    double scratch[2]{};
    velocitypp::GeneFit initial{1.,0.,.9,velocitypp::FitStatus::ok,true};
    auto option=options();
    auto scales=velocitypp::reference::stochastic_scales(ms,mu,mss,mus,2,1,initial,option,scratch);
    bool ok=scales.status==velocitypp::StochasticStatus::nonfinite&&
        std::isfinite(scales.first)&&std::isfinite(scales.second);
    auto fit=velocitypp::reference::stochastic_fit_gene(ms,mu,mss,mus,2,1,initial,scales,option,scratch);
    ok &= std::isfinite(fit.gamma)&&std::isfinite(fit.offset)&&std::isfinite(fit.offset2)&&
          std::isfinite(fit.objective)&&close(fit.gamma,initial.gamma)&&close(fit.offset,initial.offset);
    velocitypp::StochasticScales bad{std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),velocitypp::StochasticStatus::nonfinite};
    auto direct=velocitypp::reference::stochastic_fit_gene(ms,mu,mss,mus,2,1,initial,bad,option,scratch);
    ok &= std::isfinite(direct.gamma)&&std::isfinite(direct.offset)&&std::isfinite(direct.offset2)&&
          std::isfinite(direct.objective)&&close(direct.gamma,initial.gamma);
    expect(ok,"overflowed public scale and fit primitives preserve finite deterministic fallback diagnostics");
}

void scratch_does_not_scale_with_total_gene_count() {
    Fixture f;
    PreparedStorage small(f.sa.view(),f.ua.view(),f.wa.view());
    size_t genes[]{0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1};
    PreparedStorage large(f.sa.view(),f.ua.view(),f.wa.view(),genes,20);
    velocitypp::RunOptions run; run.block_genes=2;
    StochasticStorage<double> a(small.value,run), b(large.value,run);
    expect(a.workspace.block==2&&b.workspace.block==2&&a.workspace.scratch_bytes==b.workspace.scratch_bytes,
           "stochastic numerical scratch depends on rows and gene block, not full gene count");
}

void signed_and_broadcast_mask_strides() {
    Fixture f;
    std::uint8_t rows[]{1,1,1,1,1,1,1,0}, reversed[]{0,1,1,1,1,1,1,1};
    PreparedStorage a(f.sa.view(),f.ua.view(),f.wa.view(),nullptr,0,rows);
    PreparedStorage b(f.sa.view(),f.ua.view(),f.wa.view(),nullptr,0,reversed+7,nullptr,-1);
    StochasticStorage<double> x(a.value),y(b.value); x.run(a.value,options()); y.run(b.value,options());
    bool same=true;
    for(size_t j=0;j<f.g;++j) same &= x.status[j]==y.status[j] && close(x.gamma[j],y.gamma[j]);
    expect(same,"negative row-mask stride preserves logical fitting rows");

    std::uint8_t one=1, gene_mask[]{1,0}, gene_reverse[]{0,1};
    PreparedStorage c(f.sa.view(),f.ua.view(),f.wa.view(),nullptr,0,&one,gene_mask,0,1);
    PreparedStorage d(f.sa.view(),f.ua.view(),f.wa.view(),nullptr,0,&one,gene_reverse+1,0,-1);
    StochasticStorage<double> z(c.value),w(d.value); z.run(c.value,options()); w.run(d.value,options());
    same=true;
    for(size_t j=0;j<f.g;++j) same &= z.status[j]==w.status[j] && close(z.gamma[j],w.gamma[j]);
    expect(same,"broadcast row mask and negative gene-mask stride preserve logical selection");
    PreparedStorage e(f.sa.view(),f.ua.view(),f.wa.view(),nullptr,0,nullptr,&one,1,0);
    StochasticStorage<double> all(e.value),unmasked(f.prepared); all.run(e.value,options()); unmasked.run(f.prepared,options());
    same=true;
    for(size_t j=0;j<f.g;++j) same &= all.status[j]==unmasked.status[j] && close(all.gamma[j],unmasked.gamma[j]);
    expect(same,"broadcast gene mask selects every gene");

    auto invalid=f.prepared.input; invalid.fit_mask=&one; invalid.fit_mask_stride=PTRDIFF_MAX;
    expect(velocitypp::validate(invalid).code==velocitypp::Code::overflow,"row-mask stride overflow is rejected before reading");
    double source[24]{}; auto matrix=anndatapp::Array::borrow_dense(source,{8,3},{3,1});
    Acquired acquired(matrix.view()); invalid=f.prepared.input;
    invalid.spliced=acquired.matrix; invalid.unspliced=acquired.matrix;
    invalid.gene_mask=&one; invalid.gene_mask_stride=PTRDIFF_MAX;
    expect(velocitypp::validate(invalid).code==velocitypp::Code::overflow,"gene-mask stride overflow is rejected before reading");
}
void masks_and_fallbacks_are_explicit() {
    Fixture f;
    std::array<std::uint8_t,8> rows{true,true,true,true,true,true,true,false};
    PreparedStorage ps(f.sa.view(),f.ua.view(),f.wa.view(),nullptr,0,rows.data()); auto& p=ps.value;
    StochasticStorage<double> w(p); w.run(p,options());
    auto& status=w.status; auto& scales1=w.first; auto& scales2=w.second;
    auto& initial_gamma=w.ig; auto& initial_offset=w.io;
    bool ok=true;
    for(std::size_t j=0;j<f.g;++j) {
        auto st=static_cast<velocitypp::StochasticStatus>(status[j]);
        double mean1=0,mean2=0, gamma2=0, xx=0, xy=0;
        for(std::size_t i=0;i<7;++i) {
            auto k=i*f.g+j;
            double a=2*f.s[k]*f.s[k]-f.s[k], b=2*f.s[k]*f.u[k]+f.u[k];
            xx+=a*a;xy+=a*b;
        }
        gamma2=xy/xx;
        for(std::size_t i=0;i<7;++i) {
            auto k=i*f.g+j;
            mean1 += f.u[k]-initial_gamma[j]*f.s[k]-initial_offset[j];
            mean2 += 2*f.s[k]*f.u[k]+f.u[k]-gamma2*(2*f.s[k]*f.s[k]-f.s[k]);
        }
        mean1/=7;mean2/=7;
        double var1=0,var2=0;
        for(std::size_t i=0;i<7;++i) {
            auto k=i*f.g+j;
            double r1=f.u[k]-initial_gamma[j]*f.s[k]-initial_offset[j]-mean1;
            double r2=2*f.s[k]*f.u[k]+f.u[k]-gamma2*(2*f.s[k]*f.s[k]-f.s[k])-mean2;
            var1+=r1*r1;var2+=r2*r2;
        }
        ok &= std::isfinite(scales1[j])&&std::isfinite(scales2[j]);
        if(st==velocitypp::StochasticStatus::ok)
            ok &= close(scales1[j],std::sqrt(var1/7))&&close(scales2[j],std::sqrt(var2/7));
    }
    expect(ok,"fit-row mask controls both population residual scales (ddof zero)");

    std::array<std::uint8_t,2> genes{false,false};
    PreparedStorage psn(f.sa.view(),f.ua.view(),f.wa.view(),nullptr,0,nullptr,genes.data()); auto& pn=psn.value; StochasticStorage<double> wn(pn);
    auto& retained=wn.run(pn,options());
    bool none_ok=!retained.refit_all;
    for(std::size_t j=0;j<f.g;++j) none_ok &= static_cast<velocitypp::StochasticStatus>(wn.status[j])==velocitypp::StochasticStatus::not_selected&&close(wn.gamma[j],wn.ig[j]);
    expect(none_ok,"no selected genes retain deterministic diagnostics by default");
    auto refit=options(); refit.refit_all_if_none=true;
    expect(wn.run(pn,refit).refit_all,"explicit option reports refit-all fallback");

    auto threshold=options(); threshold.initialization.min_r2=2.;
    StochasticStorage<double> wt(f.prepared);
    bool kept=wt.run(f.prepared,threshold).refit_all;
    auto attempted=threshold; attempted.refit_all_if_none=true;
    bool tried=wt.run(f.prepared,attempted).refit_all;
    for(std::size_t j=0;j<f.g;++j) tried &= static_cast<velocitypp::StochasticStatus>(wt.status[j])!=velocitypp::StochasticStatus::not_selected;
    expect(tried,"refit-all option attempts valid genes after deterministic selection thresholds reject every gene");
    expect(!kept,"default no-selection path does not silently refit all genes");
}
}
int main() {
    sparse_moments_coalesce_expression_duplicates();
    dense_csr_csc_and_signed_stride_match();
    stochastic_path_and_outputs();
    typed_workspace_and_optional_residual();
    ratio_bounds_recompute_intercept_and_zero_scale_falls_back();
    validation_reports_nonfinite_input();
    overflowed_primitive_stages_keep_finite_fallbacks();
    scratch_does_not_scale_with_total_gene_count();
    signed_and_broadcast_mask_strides();
    masks_and_fallbacks_are_explicit();
    if(!failures) std::cout<<"stochastic tests passed\n";
    return failures?1:0;
}
