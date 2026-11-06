#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

import velocitypp.precision;

static size_t allocations = 0;
void* operator new(size_t n) {
    ++allocations;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }

namespace vp = velocitypp;
namespace pr = velocitypp::precision;
struct Source {
    double s[3][3]{{1,2,3},{4,5,6},{7,8,9}};
    float u[3][3]{{2,4,6},{8,10,12},{14,16,18}};
    unsigned calls[3]{};
    bool available[3]{false,true,true};
    bool repeat = false, reverse = false, overflow = false;
};
struct Output { double first[3]{}, second[3]{}; unsigned calls=0; };

bool acquire(void* opaque, size_t gene, pr::OriginalColumn& c) {
    auto& s=*static_cast<Source*>(opaque);
    ++s.calls[gene];
    if (!s.available[gene]) return false;
    size_t row=s.reverse ? 2 : 0;
    ptrdiff_t ss=s.repeat ? 0 : s.reverse ? -ptrdiff_t(sizeof(s.s[0])) : ptrdiff_t(sizeof(s.s[0]));
    ptrdiff_t us=s.repeat ? 0 : s.reverse ? -ptrdiff_t(sizeof(s.u[0])) : ptrdiff_t(sizeof(s.u[0]));
    c={&s.s[row][gene], &s.u[row][gene], s.overflow ? PTRDIFF_MAX : ss, us,
       pr::OriginalType::float64, pr::OriginalType::float32, 3};
    return true;
}
bool throwing_source(void*, size_t, pr::OriginalColumn&) { throw 7; }
pr::PrecisionCode replace(void* opaque, size_t position, size_t, const pr::OriginalColumn& c, bool second) {
    auto& out=*static_cast<Output*>(opaque);
    auto x=*static_cast<const double*>(c.spliced);
    auto y=*static_cast<const float*>(c.unspliced);
    (second ? out.second : out.first)[position]=x+double(y);
    ++out.calls;
    return pr::PrecisionCode::ok;
}

int main() {
    constexpr size_t count=3;
    size_t genes[count]{2,0,1};
    uint8_t mask[count]{1,0,1}, states[count]{};
    Source source;
    Output output;
    pr::Promotion p{genes,count,count,3,mask,count,states,count,0,
                    {&source,acquire},{&output,replace}};
    size_t before=allocations;
    if (pr::initialize_promotion(p).code != pr::PrecisionCode::ok) return 1;
    if (pr::promote_initial(p,0,2).code != pr::PrecisionCode::ok) return 2;
    if (pr::promote_initial(p,2,1).code != pr::PrecisionCode::ok) return 3;
    if (p.promoted != 1 || states[0] != uint8_t(pr::PromotionStatus::promoted) ||
        states[1] != uint8_t(pr::PromotionStatus::unavailable) ||
        states[2] != uint8_t(pr::PromotionStatus::not_requested)) return 4;
    if (output.first[0] != 9 || output.calls != 1 || source.calls[2] != 1 || source.calls[0] != 1) return 5;
    if (pr::promote_second(p,0,count).code != pr::PrecisionCode::ok ||
        output.second[0] != 9 || source.calls[2] != 2) return 6;
    source.available[2]=false;
    auto unavailable=pr::promote_second(p,0,count);
    if (unavailable.code != pr::PrecisionCode::unavailable || unavailable.location != 0) return 7;
    p.status_capacity=2;
    auto capacity=pr::validate_promotion(p);
    if (capacity.code != pr::PrecisionCode::insufficient_capacity || capacity.required != count) return 8;
    if (allocations != before) return 9;

    vp::FitOptions options;
    double x[]{1,2,3,4,5}, y[]{2,4,6,8,10}, scratch[5]{};
    vp::GeneFit baseline{2,0,1,vp::FitStatus::ok,true};
    auto check=pr::validate_sensitivity(5,1,x,y,scratch,4,options);
    if (check.code != pr::PrecisionCode::insufficient_capacity || check.required != 5) return 10;
    check=pr::validate_sensitivity(5,1,x,y,scratch,5,options);
    if (check.code != pr::PrecisionCode::ok) return 11;
    auto result=pr::sensitivity_gene(x,y,5,1,baseline,options,scratch);
    if (result.state == pr::SensitivityState::unscoreable || allocations != before) return 12;

    // Acquiring AnnData descriptors is caller work; all reference operations
    // below use them directly and must allocate nothing.
    float identity[]{1,0,0, 0,1,0, 0,0,1};
    float base_s[]{1,2,3, 4,5,6, 7,8,9};
    float base_u[]{2,4,6, 8,10,12, 14,16,18};
    auto sw=anndatapp::Array::borrow_dense(identity,{3,3},{3,1});
    auto ss=anndatapp::Array::borrow_dense(base_s,{3,3},{3,1});
    auto su=anndatapp::Array::borrow_dense(base_u,{3,3},{3,1});
    auto wd=sw.view().bytes(), sd=ss.view().bytes(), ud=su.view().bytes();
    vp::Input input{{&sd},{&ud},{&wd}};
    vp::PreparationSize size; vp::PreparedInput prepared;
    size_t indices[128]{}; vp::Entry entries[128]{};
    vp::PreparationBuffers buffers{indices,128,entries,128};
    before=allocations;
    if (!vp::preparation_size(input,size) || size.indices > 128 || size.entries > 128) return 13;
    if (!vp::prepare(input,buffers,prepared)) return 14;
    float ms[9]{},mu[9]{},mss[9]{},mus[9]{},muu[9]{};
    vp::BasicMomentBlock<float> initial{0,3,3,3,ms,mu};
    vp::BasicStochasticMomentBlock<float> second{0,3,3,3,ms,mu,mss,mus,muu};
    pr::ReferenceMoments<float> context{&prepared,&initial,&second};
    source.available[2]=true; p.status_capacity=count;
    p.replacement={&context,pr::replace_reference32};
    if (pr::initialize_promotion(p).code != pr::PrecisionCode::ok ||
        pr::promote_initial(p,0,3).code != pr::PrecisionCode::ok) return 15;
    for (size_t i=0;i<3;++i)
        if (ms[3*i] != source.s[i][2] || mu[3*i] != source.u[i][2]) return 16;
    if (pr::promote_second(p,0,3).code != pr::PrecisionCode::ok) return 17;
    for (size_t i=0;i<3;++i) {
        float sx=float(source.s[i][2]), uy=source.u[i][2];
        if (mss[3*i] != sx*sx || mus[3*i] != sx*uy || muu[3*i] != uy*uy) return 18;
    }
    source.repeat=true;
    if (pr::initialize_promotion(p).code != pr::PrecisionCode::ok ||
        pr::promote_initial(p,0,3).code != pr::PrecisionCode::ok) return 19;
    for (size_t i=0;i<3;++i) if (ms[3*i] != 3 || mu[3*i] != 6) return 20;
    source.repeat=false; source.reverse=true;
    if (pr::initialize_promotion(p).code != pr::PrecisionCode::ok ||
        pr::promote_initial(p,0,3).code != pr::PrecisionCode::ok) return 21;
    for (size_t i=0;i<3;++i)
        if (ms[3*i] != source.s[2-i][2] || mu[3*i] != source.u[2-i][2]) return 22;
    source.reverse=false; source.overflow=true;
    if (pr::initialize_promotion(p).code != pr::PrecisionCode::ok) return 23;
    ms[0]=123; mu[0]=456;
    auto invalid=pr::promote_initial(p,0,3);
    if (invalid.code != pr::PrecisionCode::invalid_argument || invalid.location != 0 ||
        ms[0]!=123 || mu[0]!=456 || p.promoted!=0 || states[0]!=uint8_t(pr::PromotionStatus::not_requested)) return 24;
    pr::OriginalColumn bad; if (!acquire(&source,2,bad)) return 25;
    if (pr::replace_reference32(&context,0,0,bad,false) != pr::PrecisionCode::invalid_argument ||
        ms[0]!=123 || mu[0]!=456) return 26;
    if (allocations != before) return 27;
    p.source={nullptr,throwing_source};
    bool propagated=false;
    try { (void)pr::promote_initial(p,0,1); } catch (int value) { propagated=value==7; }
    if (!propagated) return 28;
    return 0;
}
