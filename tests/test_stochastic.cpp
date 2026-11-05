#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

import velocity;

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
velocity::FitOptions initialization() {
    velocity::FitOptions o;
    o.trimming = velocity::Trimming::none;
    o.fit_intercept = true;
    o.min_r2 = 0;
    o.min_ratio = 0;
    return o;
}
velocity::StochasticOptions options() {
    velocity::StochasticOptions o;
    o.initialization = initialization();
    return o;
}
struct Fixture {
    std::size_t n = 8, g = 2;
    std::vector<double> s, u, w;
    anndatapp::Array sa, ua, wa;
    velocity::PreparedInput prepared;
    Fixture()
        : s{1,3, 2,4, 1,2, 3,7, 2,5, 5,8, 4,10, 7,11},
          u{2,5, 5,8, 1,5, 8,12, 3,13, 8,13, 10,17, 13,20},
          w(n*n), sa(anndatapp::Array::borrow_dense(s.data(), {n,g}, {std::ptrdiff_t(g),1})),
          ua(anndatapp::Array::borrow_dense(u.data(), {n,g}, {std::ptrdiff_t(g),1})),
          wa([&] {
              for (std::size_t i=0;i<n;++i) w[i*n+i]=1;
              return anndatapp::Array::borrow_dense(w.data(), {n,n}, {std::ptrdiff_t(n),1});
          }()), prepared({sa.view(),ua.view(),wa.view()}) {}
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
    velocity::PreparedInput p({s.view(),u.view(),w.view()});
    double ms[n]{},mu[n]{},mss[n]{},mus[n]{},muu[n]{},scratch_s[n]{},scratch_u[n]{};
    velocity::reference::second_moments(p,0,1,ms,mu,mss,mus,muu,scratch_s,scratch_u);
    expect(close(ms[0],3)&&close(mu[0],5)&&close(mss[0],9)&&close(mus[0],15)&&close(muu[0],25)&&
           close(ms[1],2)&&close(mu[1],4)&&close(mss[1],4)&&close(mus[1],8)&&close(muu[1],16),
           "CSR/CSC second moments combine duplicate logical expression entries before products");
}
void dense_csr_csc_and_signed_stride_match() {
    Fixture f;
    std::vector<double> ms(f.n*f.g),mu(f.n*f.g),mss(f.n*f.g),mus(f.n*f.g),muu(f.n*f.g),ss(f.n),su(f.n);
    velocity::reference::second_moments(f.prepared,0,f.g,ms.data(),mu.data(),mss.data(),mus.data(),muu.data(),ss.data(),su.data());
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
    velocity::PreparedInput sparse({sa.view(),ua.view(),wa.view()});
    std::fill(ms.begin(),ms.end(),0);std::fill(mu.begin(),mu.end(),0);std::fill(mss.begin(),mss.end(),0);
    std::fill(mus.begin(),mus.end(),0);std::fill(muu.begin(),muu.end(),0);
    velocity::reference::second_moments(sparse,0,f.g,ms.data(),mu.data(),mss.data(),mus.data(),muu.data(),ss.data(),su.data());
    bool sparse_ok=true;
    for(std::size_t i=0;i<f.n*f.g;++i) sparse_ok &= close(ms[i],f.s[i])&&close(mu[i],f.u[i])&&close(mss[i],f.s[i]*f.s[i])&&close(mus[i],f.s[i]*f.u[i])&&close(muu[i],f.u[i]*f.u[i]);
    expect(sparse_ok,"CSR/CSC layers with opposite sparsity patterns match dense moments");

    std::vector<double> sr(f.s.size()),ur(f.u.size());
    for(std::size_t i=0;i<f.n;++i) for(std::size_t j=0;j<f.g;++j) {
        sr[i*f.g+j]=f.s[(f.n-1-i)*f.g+j]; ur[i*f.g+j]=f.u[(f.n-1-i)*f.g+j];
    }
    auto srev=anndatapp::Array::borrow_dense(sr.data()+(f.n-1)*f.g,{f.n,f.g},{-std::ptrdiff_t(f.g),1});
    auto urev=anndatapp::Array::borrow_dense(ur.data()+(f.n-1)*f.g,{f.n,f.g},{-std::ptrdiff_t(f.g),1});
    velocity::PreparedInput reversed({srev.view(),urev.view(),f.wa.view()});
    std::fill(ms.begin(),ms.end(),0);std::fill(mu.begin(),mu.end(),0);std::fill(mss.begin(),mss.end(),0);
    std::fill(mus.begin(),mus.end(),0);std::fill(muu.begin(),muu.end(),0);
    velocity::reference::second_moments(reversed,0,f.g,ms.data(),mu.data(),mss.data(),mus.data(),muu.data(),ss.data(),su.data());
    expect(ms==f.s&&mu==f.u,"signed dense row strides preserve logical second-moment row order");
}
void stochastic_path_and_outputs() {
    Fixture f;
    velocity::StochasticRunOptions ro; ro.block_genes=1; ro.output_dtype=velocity::StorageType::float64;
    ro.second_residual=true; ro.spliced_second=true; ro.cross=true; ro.unspliced_second=true;
    velocity::StochasticWorkspace workspace(f.prepared,ro);
    expect(workspace.block_genes()==1&&workspace.scratch_bytes()<=ro.workspace_bytes,
           "stochastic workspace obeys one-gene blocks and budget");
    auto v=anndatapp::Array::zeros<double>({f.n,f.g}),r2=anndatapp::Array::zeros<double>({f.n,f.g});
    auto result=velocity::reference::stochastic_into(f.prepared,workspace,v.edit(),options(),r2.edit());
    auto gamma=dense_values(result.gamma),offset=dense_values(result.offset),offset2=dense_values(result.offset2);
    auto status=result.status.view().dense<std::uint8_t>();
    auto velocity=dense_values(v), residual2=dense_values(r2);
    bool outputs_ok=true;
    for(std::size_t j=0;j<f.g;++j) {
        auto st=static_cast<velocity::StochasticStatus>(status.data()[j]);
        outputs_ok &= std::isfinite(gamma[j])&&std::isfinite(offset[j])&&std::isfinite(offset2[j]);
        if(st==velocity::StochasticStatus::ok) {
            for(std::size_t i=0;i<f.n;++i) {
                auto k=i*f.g+j;
                outputs_ok &= close(velocity[k],f.u[k]-gamma[j]*f.s[k]-offset[j]);
                double A=2*f.s[k]*f.s[k]-f.s[k], B=2*f.s[k]*f.u[k]+f.u[k];
                outputs_ok &= close(residual2[k],B-2*f.s[k]*f.u[k]-gamma[j]*(A-2*f.s[k]*f.s[k])+2*offset[j]*f.s[k]-offset2[j]);
            }
        }
    }
    expect(outputs_ok,"stochastic caller-owned velocity and second residual follow documented equations and remain finite");
    expect(result.initialization.gamma.view().extent(0)==f.g,"result retains deterministic initialization diagnostics");

    velocity::StochasticRunOptions wide; wide.block_genes=64;
    velocity::StochasticWorkspace workspace_wide(f.prepared,wide);
    auto out_wide=velocity::reference::stochastic(f.prepared,workspace_wide,options());
    bool same=true; auto gwide=dense_values(out_wide.gamma);
    for(std::size_t j=0;j<f.g;++j) same &= close(gamma[j],gwide[j]);
    expect(same,"stochastic results are independent of gene block width");
}
void ratio_bounds_recompute_intercept_and_zero_scale_falls_back() {
    Fixture f;
    velocity::StochasticWorkspace workspace(f.prepared);
    auto bound=options(); bound.initialization.ratio_max=1.;
    auto result=velocity::reference::stochastic(f.prepared,workspace,bound);
    auto gamma=dense_values(result.gamma), offset=dense_values(result.offset);
    auto status=result.status.view().dense<std::uint8_t>();
    bool ok=true;
    for(std::size_t j=0;j<f.g;++j) {
        auto st=static_cast<velocity::StochasticStatus>(status.data()[j]);
        if(st==velocity::StochasticStatus::ok) {
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
    velocity::PreparedInput p({s.view(),u.view(),w.view()});
    velocity::StochasticRunOptions ro; ro.second_residual=true;
    velocity::StochasticWorkspace ws(p,ro);
    auto velocity_out=anndatapp::Array::zeros<double>({n,1});
    auto residual_out=anndatapp::Array::zeros<double>({n,1});
    auto exact=velocity::reference::stochastic_into(p,ws,velocity_out.edit(),options(),residual_out.edit());
    auto exact_status=exact.status.view().dense<std::uint8_t>();
    auto exact_gamma=dense_values(exact.gamma),exact_offset=dense_values(exact.offset);
    auto ev=dense_values(velocity_out),er=dense_values(residual_out);
    bool fallback=static_cast<velocity::StochasticStatus>(exact_status.data()[0])==velocity::StochasticStatus::zero_scale&&
        close(exact_gamma[0],2.)&&close(exact_offset[0],0.);
    for(double x:ev) fallback &= close(x,0.);
    for(double x:er) fallback &= close(x,0.);
    expect(fallback,"zero residual scale preserves deterministic parameters and zeroes second residual output");
}

template<class F> void expect_invalid(F&& f,const char* what) {
    try { f(); expect(false,what); }
    catch(const std::invalid_argument&) {}
    catch(...) { expect(false,what); }
}
void public_buffers_reject_aliases_and_invalid_strides() {
    Fixture f;
    double ms[8]{},mu[8]{},mss[8]{},mus[8]{},muu[8]{},ss[8]{},su[8]{};
    expect_invalid([&] { velocity::reference::second_moments(f.prepared,0,1,
        f.s.data(),mu,mss,mus,muu,ss,su); },
        "second-moment output may not overwrite an expression input");
    expect_invalid([&] { velocity::reference::second_moments(f.prepared,0,1,
        ms,mu,mss,mus,muu,ms,su); },
        "second-moment scratch may not overlap a requested output");

    auto old=f.s[0]; f.s[0]=std::numeric_limits<double>::quiet_NaN();
    expect_invalid([&] { velocity::reference::second_moments(f.prepared,0,1,
        ms,mu,mss,mus,muu,ss,su); },
        "full validation catches borrowed input mutation before second-moment evaluation");
    f.s[0]=old;

    double gamma=1,offset=0,offset2=0,scale=1,objective=0;
    std::uint8_t status=std::uint8_t(velocity::StochasticStatus::ok);
    velocity::StochasticFitBlock fit{0,1,&gamma,&offset,&offset2,&scale,&scale,&status,&objective};
    double xs[8]{1,2,3,4,5,6,7,8},ys[8]{2,4,6,8,10,12,14,16},q[8]{},c[8]{};
    velocity::StochasticMomentBlock m{0,1,8,1,xs,ys,q,c,nullptr};
    auto alias=anndatapp::Array::borrow_dense(xs,{8,1},{1,1});
    expect_invalid([&] { velocity::reference::stochastic_residuals(m,fit,alias.edit()); },
        "stochastic residual output may not overwrite borrowed moment storage");
    double shared[8]{};
    auto first=anndatapp::Array::borrow_dense(shared,{8,1},{1,1});
    auto second=anndatapp::Array::borrow_dense(shared,{8,1},{1,1});
    expect_invalid([&] { velocity::reference::stochastic_residuals(m,fit,first.edit(),second.edit()); },
        "stochastic velocity and second-residual outputs may not overlap");

    double scratch[3]{};
    auto init=velocity::GeneFit{1.,0.,.9,velocity::FitStatus::ok,true};
    auto opt=options();
    expect_invalid([&] { (void)velocity::reference::stochastic_fit_gene(nullptr,ys,q,c,3,1,init,{},opt,scratch); },
        "public per-gene stochastic primitive rejects null numerical inputs");
    auto too_wide=std::numeric_limits<std::ptrdiff_t>::max()/2+1;
    expect_invalid([&] { (void)velocity::reference::stochastic_fit_gene(xs,ys,q,c,3,too_wide,init,{},opt,scratch); },
        "public per-gene stochastic primitive rejects overflowing signed strides before reading");
}

void overflowed_primitive_stages_keep_finite_fallbacks() {
    const double ms[]{1e100,1e100},mu[]{1e100,1e100};
    const double mss[]{1e200,1e200},mus[]{2e200,2e200};
    double scratch[2]{};
    velocity::GeneFit initial{1.,0.,.9,velocity::FitStatus::ok,true};
    auto option=options();
    auto scales=velocity::reference::stochastic_scales(ms,mu,mss,mus,2,1,initial,option,scratch);
    bool ok=scales.status==velocity::StochasticStatus::nonfinite&&
        std::isfinite(scales.first)&&std::isfinite(scales.second);
    auto fit=velocity::reference::stochastic_fit_gene(ms,mu,mss,mus,2,1,initial,scales,option,scratch);
    ok &= std::isfinite(fit.gamma)&&std::isfinite(fit.offset)&&std::isfinite(fit.offset2)&&
          std::isfinite(fit.objective)&&close(fit.gamma,initial.gamma)&&close(fit.offset,initial.offset);
    velocity::StochasticScales bad{std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),velocity::StochasticStatus::nonfinite};
    auto direct=velocity::reference::stochastic_fit_gene(ms,mu,mss,mus,2,1,initial,bad,option,scratch);
    ok &= std::isfinite(direct.gamma)&&std::isfinite(direct.offset)&&std::isfinite(direct.offset2)&&
          std::isfinite(direct.objective)&&close(direct.gamma,initial.gamma);
    expect(ok,"overflowed public scale and fit primitives preserve finite deterministic fallback diagnostics");
}

void scratch_does_not_scale_with_total_gene_count() {
    Fixture f;
    velocity::Input small_in{f.sa.view(),f.ua.view(),f.wa.view()};
    velocity::PreparedInput small(std::move(small_in));
    velocity::Input large_in{f.sa.view(),f.ua.view(),f.wa.view()};
    large_in.genes=anndatapp::Axis::gather({0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1});
    velocity::PreparedInput large(std::move(large_in));
    velocity::StochasticRunOptions run; run.block_genes=2;
    velocity::StochasticWorkspace a(small,run), b(large,run);
    expect(a.block_genes()==2&&b.block_genes()==2&&a.scratch_bytes()==b.scratch_bytes(),
           "stochastic numerical scratch depends on rows and gene block, not full gene count");
}

void masks_and_fallbacks_are_explicit() {
    Fixture f;
    std::array<bool,8> rows{true,true,true,true,true,true,true,false};
    auto mask=anndatapp::Array::borrow_dense(rows.data(),{f.n},{1});
    velocity::Input in{f.sa.view(),f.ua.view(),f.wa.view()}; in.fit_mask=mask.view();
    velocity::PreparedInput p(std::move(in));
    velocity::StochasticWorkspace w(p);
    auto result=velocity::reference::stochastic(p,w,options());
    auto status=result.status.view().dense<std::uint8_t>();
    auto scales1=dense_values(result.first_scale), scales2=dense_values(result.second_scale);
    auto initial_gamma=dense_values(result.initialization.gamma);
    auto initial_offset=dense_values(result.initialization.offset);
    bool ok=true;
    for(std::size_t j=0;j<f.g;++j) {
        auto st=static_cast<velocity::StochasticStatus>(status.data()[j]);
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
        if(st==velocity::StochasticStatus::ok)
            ok &= close(scales1[j],std::sqrt(var1/7))&&close(scales2[j],std::sqrt(var2/7));
    }
    expect(ok,"fit-row mask controls both population residual scales (ddof zero)");

    std::array<bool,2> genes{false,false};
    auto gene_mask=anndatapp::Array::borrow_dense(genes.data(),{f.g},{1});
    velocity::Input none{f.sa.view(),f.ua.view(),f.wa.view()}; none.gene_mask=gene_mask.view();
    velocity::PreparedInput pn(std::move(none)); velocity::StochasticWorkspace wn(pn);
    auto retained=velocity::reference::stochastic(pn,wn,options());
    auto st=retained.status.view().dense<std::uint8_t>();
    auto sg=retained.gamma.view().dense<double>();
    auto retained_gamma=dense_values(retained.initialization.gamma);
    bool none_ok=!retained.refit_all;
    for(std::size_t j=0;j<f.g;++j) none_ok &= static_cast<velocity::StochasticStatus>(st.data()[j])==velocity::StochasticStatus::not_selected&&close(sg.data()[j],retained_gamma[j]);
    expect(none_ok,"no selected genes retain deterministic diagnostics by default");
    auto refit=options(); refit.refit_all_if_none=true;
    auto all=velocity::reference::stochastic(pn,wn,refit);
    expect(all.refit_all,"explicit option reports scVelo refit-all fallback when no genes pass selection");

    auto threshold=options(); threshold.initialization.min_r2=2.;
    velocity::StochasticWorkspace wt(f.prepared);
    auto kept_all=velocity::reference::stochastic(f.prepared,wt,threshold);
    auto attempted=threshold; attempted.refit_all_if_none=true;
    auto refined=velocity::reference::stochastic(f.prepared,wt,attempted);
    auto refined_status=refined.status.view().dense<std::uint8_t>();
    bool tried=refined.refit_all;
    for(std::size_t j=0;j<f.g;++j)
        tried &= static_cast<velocity::StochasticStatus>(refined_status.data()[j])!=velocity::StochasticStatus::not_selected;
    expect(tried,"refit-all option attempts valid genes after deterministic selection thresholds reject every gene");
    expect(!kept_all.refit_all,"default no-selection path does not silently refit all genes");
}
}
int main() {
    sparse_moments_coalesce_expression_duplicates();
    dense_csr_csc_and_signed_stride_match();
    stochastic_path_and_outputs();
    ratio_bounds_recompute_intercept_and_zero_scale_falls_back();
    public_buffers_reject_aliases_and_invalid_strides();
    overflowed_primitive_stages_keep_finite_fallbacks();
    scratch_does_not_scale_with_total_gene_count();
    masks_and_fallbacks_are_explicit();
    if(!failures) std::cout<<"stochastic tests passed\n";
    return failures?1:0;
}
