module;
#include <stddef.h>
#include <stdint.h>

module velocitypp;
import anndatapp.core;

namespace velocitypp {
namespace {
using anndatapp::DType;
using anndatapp::Layout;

// The descriptor references existing caller-acquired AnnData++ storage. This
// local traversal state owns no storage and does not extend descriptor lifetime.
struct Source {
    Layout layout;
    DType dtype, index_type, offset_type;
    const unsigned char *values, *indices, *offsets;
    size_t rows, columns, nnz, value_size, index_size, offset_size;
    ptrdiff_t row_stride, column_stride;
};
Source source(const AcquiredMatrix& x) {
    Source a{};
    a.layout=x.layout; a.dtype=x.dtype; a.index_type=x.index_type; a.offset_type=x.offset_type;
    if (a.layout==Layout::dense) {
        const auto& d=*x.dense; a.values=reinterpret_cast<const unsigned char*>(d.data());
        a.rows=d.extent(0);a.columns=d.extent(1);a.value_size=d.element_size();
        a.row_stride=d.stride_bytes(0);a.column_stride=d.stride_bytes(1);
    } else {
        const auto& s=*x.sparse;a.values=reinterpret_cast<const unsigned char*>(s.values());
        a.indices=reinterpret_cast<const unsigned char*>(s.indices());
        a.offsets=reinterpret_cast<const unsigned char*>(s.offsets());
        a.rows=s.rows();a.columns=s.columns();a.value_size=s.value_size();
        a.index_size=s.index_size();a.offset_size=s.offset_size();a.nnz=s.nnz();
    }
    return a;
}
template<class T> T load(const unsigned char* p) { T x; __builtin_memcpy(&x,p,sizeof x); return x; }
template<class T> bool signed_integer(const unsigned char* p,size_t& out) {
    T x=load<T>(p); if(x<0)return false; out=size_t(x); return uint64_t(x)==uint64_t(out);
}
template<class T> bool unsigned_integer(const unsigned char* p,size_t& out) {
    T x=load<T>(p); out=size_t(x); return uint64_t(x)==uint64_t(out);
}
bool integer(const unsigned char* p,DType type,size_t& out) {
    switch(type) {
    case DType::int8:return signed_integer<int8_t>(p,out);
    case DType::uint8:return unsigned_integer<uint8_t>(p,out);
    case DType::int16:return signed_integer<int16_t>(p,out);
    case DType::uint16:return unsigned_integer<uint16_t>(p,out);
    case DType::int32:return signed_integer<int32_t>(p,out);
    case DType::uint32:return unsigned_integer<uint32_t>(p,out);
    case DType::int64:return signed_integer<int64_t>(p,out);
    case DType::uint64:return unsigned_integer<uint64_t>(p,out);
    default:return false;
    }
}
size_t index_at(const Source& a,size_t p) {
    size_t x=0; (void)integer(a.indices+p*a.index_size,a.index_type,x); return x;
}
size_t offset_at(const Source& a,size_t p) {
    size_t x=0; (void)integer(a.offsets+p*a.offset_size,a.offset_type,x); return x;
}
bool floating(DType t) { return t==DType::float32 || t==DType::float64; }
bool sparse_integer(DType t) {
    return t==DType::int8 || t==DType::uint8 || t==DType::int16 || t==DType::uint16 ||
           t==DType::int32 || t==DType::uint32 || t==DType::int64 || t==DType::uint64;
}
size_t type_width(DType t) {
    switch(t) {
    case DType::int8:case DType::uint8:return 1;
    case DType::int16:case DType::uint16:return 2;
    case DType::int32:case DType::uint32:case DType::float32:return 4;
    case DType::int64:case DType::uint64:case DType::float64:return 8;
    default:return 0;
    }
}
Status descriptor(const AcquiredMatrix& x,size_t& rows,size_t& columns) {
    if(!floating(x.dtype))return {Code::dtype};
    if(x.layout==Layout::dense) {
        if(!x.dense || x.sparse)return {Code::argument};
        if(x.dense->rank()!=2)return {Code::shape};
        if(x.dense->element_size()!=type_width(x.dtype))return {Code::dtype};
        rows=x.dense->extent(0);columns=x.dense->extent(1);
    } else if(x.layout==Layout::csr || x.layout==Layout::csc) {
        if(!x.sparse || x.dense)return {Code::argument};
        if(x.sparse->value_size()!=type_width(x.dtype) || !sparse_integer(x.index_type) ||
           !sparse_integer(x.offset_type) || x.sparse->index_size()!=type_width(x.index_type) ||
           x.sparse->offset_size()!=type_width(x.offset_type))return {Code::dtype};
        rows=x.sparse->rows();columns=x.sparse->columns();
        size_t nnz=x.sparse->nnz(),major=x.layout==Layout::csr?rows:columns;
        if(major==SIZE_MAX || major+1>size_t(PTRDIFF_MAX)/x.sparse->offset_size() ||
           nnz>size_t(PTRDIFF_MAX)/x.sparse->value_size() ||
           nnz>size_t(PTRDIFF_MAX)/x.sparse->index_size())return {Code::overflow};
    } else return {Code::layout};
    if(rows==SIZE_MAX || columns==SIZE_MAX || (rows && columns>SIZE_MAX/rows))return {Code::overflow};
    return {};
}
Status shape(const Input& in,size_t& n,size_t& g) {
    size_t ur,uc,wr,wc;
    Status s=descriptor(in.spliced,n,g);if(!s)return s;
    s=descriptor(in.unspliced,ur,uc);if(!s)return s;
    s=descriptor(in.weights,wr,wc);if(!s)return s;
    if(ur!=n || uc!=g || wr!=n || wc!=n)return {Code::shape};
    return {};
}
Status valid_source(const Source& a) {
    if(a.layout!=Layout::dense && a.layout!=Layout::csr && a.layout!=Layout::csc)return {Code::layout};
    if(a.layout==Layout::dense) {
        if(a.rows && a.columns && !a.values)return {Code::argument};
        if(a.rows && a.columns) {
            if(a.rows-1>size_t(PTRDIFF_MAX) || a.columns-1>size_t(PTRDIFF_MAX))return {Code::overflow};
            ptrdiff_t r,c,low,high,last;
            if(__builtin_mul_overflow(ptrdiff_t(a.rows-1),a.row_stride,&r) ||
               __builtin_mul_overflow(ptrdiff_t(a.columns-1),a.column_stride,&c) ||
               __builtin_add_overflow(r<0?r:0,c<0?c:0,&low) ||
               __builtin_add_overflow(r>0?r:0,c>0?c:0,&high) ||
               __builtin_add_overflow(high,ptrdiff_t(a.value_size-1),&last))return {Code::overflow};
        }
        for(size_t i=0;i<a.rows;++i)for(size_t j=0;j<a.columns;++j) {
            auto p=a.values+ptrdiff_t(i)*a.row_stride+ptrdiff_t(j)*a.column_stride;
            bool finite=a.dtype==DType::float32?__builtin_isfinite(load<float>(p)):__builtin_isfinite(load<double>(p));
            if(!finite)return {Code::nonfinite,i*a.columns+j};
        }
        return {};
    }
    if(!sparse_integer(a.index_type) || !sparse_integer(a.offset_type))return {Code::dtype};
    if(!a.offsets || (a.nnz && (!a.values || !a.indices)))return {Code::argument};
    size_t major=a.layout==Layout::csr?a.rows:a.columns, minor=a.layout==Layout::csr?a.columns:a.rows;
    size_t prior=0,off=0;
    if(!integer(a.offsets,a.offset_type,off) || off!=0)return {Code::bounds,0};
    for(size_t i=1;i<=major;++i) {
        if(!integer(a.offsets+i*a.offset_size,a.offset_type,off) || off<prior || off>a.nnz)return {Code::bounds,i};
        prior=off;
    }
    if(prior!=a.nnz)return {Code::bounds,major};
    for(size_t p=0;p<a.nnz;++p) {
        size_t k=0;
        if(!integer(a.indices+p*a.index_size,a.index_type,k) || k>=minor)return {Code::bounds,p};
        auto v=a.values+p*a.value_size;
        bool finite=a.dtype==DType::float32?__builtin_isfinite(load<float>(v)):__builtin_isfinite(load<double>(v));
        if(!finite)return {Code::nonfinite,p};
    }
    return {};
}
bool add(size_t& total,size_t x) {
    if(x>SIZE_MAX-total)return false;total+=x;return true;
}
bool size_for(const Source& s,const Source& u,const Source& w,PreparationSize& z) {
    z={};
    bool ss=s.layout!=Layout::dense,us=u.layout!=Layout::dense,ws=w.layout!=Layout::dense;
    if(ss && (!add(z.indices,s.columns+1)||!add(z.entries,s.nnz)))return false;
    if(us && (!add(z.indices,u.columns+1)||!add(z.entries,u.nnz)))return false;
    if(ws && (!add(z.indices,w.rows+1)||!add(z.indices,w.rows+1)||!add(z.entries,w.nnz)||!add(z.entries,w.nnz)))return false;
    // One cursor is reused for all column/row fills.
    size_t cursor=(ss||us)?s.columns:0;
    if(ws && w.rows>cursor)cursor=w.rows;
    if((ss||us||ws) && !add(z.indices,cursor+1))return false;
    return z.indices<=size_t(PTRDIFF_MAX)/sizeof(size_t) && z.entries<=size_t(PTRDIFF_MAX)/sizeof(Entry);
}
void prefix(size_t* p,size_t n) { for(size_t i=1;i<=n;++i)p[i]+=p[i-1]; }
bool less(Entry a,Entry b) { return a.index<b.index || (a.index==b.index && a.position<b.position); }
void sift(Entry* a,size_t root,size_t n) {
    while(root<n/2) {
        size_t c=2*root+1;if(c+1<n && less(a[c],a[c+1]))++c;
        if(!less(a[root],a[c]))return;
        Entry t=a[root];a[root]=a[c];a[c]=t;root=c;
    }
}
void sort(Entry* a,size_t n) {
    for(size_t i=n/2;i>0;--i)sift(a,i-1,n);
    for(size_t i=n;i>1;--i){Entry t=a[0];a[0]=a[i-1];a[i-1]=t;sift(a,0,i-1);}
}
void columns(const Source& a,ColumnIndex out,size_t* cursor) {
    size_t n=a.rows,g=a.columns;
    for(size_t j=0;j<=g;++j)out.offsets[j]=0;
    size_t major=a.layout==Layout::csr?n:g;
    for(size_t i=0;i<major;++i)for(size_t p=offset_at(a,i);p<offset_at(a,i+1);++p) {
        size_t j=a.layout==Layout::csr?index_at(a,p):i;
        ++out.offsets[j+1];
    }
    prefix(out.offsets,g);
    for(size_t j=0;j<g;++j)cursor[j]=out.offsets[j];
    for(size_t i=0;i<major;++i)for(size_t p=offset_at(a,i);p<offset_at(a,i+1);++p) {
        size_t k=index_at(a,p),j=a.layout==Layout::csr?k:i,row=a.layout==Layout::csr?i:k;
        out.entries[cursor[j]++]={row,p};
    }
    // CSC row indices may be unsorted; ordering preserves duplicate value order.
    if(a.layout==Layout::csc)for(size_t j=0;j<g;++j) {
        size_t first=out.offsets[j],count=out.offsets[j+1]-first;
        if(count>1)sort(out.entries+first,count);
    }
}
void graph(const Source& w,GraphIndex out,size_t* cursor) {
    size_t n=w.rows;
    for(size_t i=0;i<=n;++i){out.rows[i]=0;out.columns[i]=0;}
    for(size_t major=0;major<n;++major)for(size_t p=offset_at(w,major);p<offset_at(w,major+1);++p) {
        size_t minor=index_at(w,p),row=w.layout==Layout::csr?major:minor,col=w.layout==Layout::csr?minor:major;
        ++out.rows[row+1];++out.columns[col+1];
    }
    prefix(out.rows,n);prefix(out.columns,n);
    for(size_t i=0;i<n;++i)cursor[i]=out.rows[i];
    for(size_t major=0;major<n;++major)for(size_t p=offset_at(w,major);p<offset_at(w,major+1);++p) {
        size_t minor=index_at(w,p),row=w.layout==Layout::csr?major:minor,col=w.layout==Layout::csr?minor:major;
        out.row_edges[cursor[row]++]={col,p};
    }
    for(size_t i=0;i<n;++i)cursor[i]=out.columns[i];
    for(size_t major=0;major<n;++major)for(size_t p=offset_at(w,major);p<offset_at(w,major+1);++p) {
        size_t minor=index_at(w,p),row=w.layout==Layout::csr?major:minor,col=w.layout==Layout::csr?minor:major;
        out.column_edges[cursor[col]++]={row,p};
    }
    for(size_t i=0;i<n;++i) {
        size_t first=out.rows[i],count=out.rows[i+1]-first;
        if(count>1)sort(out.row_edges+first,count);
    }
}
Status bind(const Input& in,const PreparationBuffers& b,PreparedInput& result) {
    size_t n,g;Status s=shape(in,n,g);if(!s)return s;
    Source a=source(in.spliced),u=source(in.unspliced),w=source(in.weights);
    PreparationSize needed;if(!size_for(a,u,w,needed))return {Code::overflow};
    if(needed.indices>b.index_capacity || (needed.indices && !b.indices))return {Code::capacity,0,needed.indices};
    if(needed.entries>b.entry_capacity || (needed.entries && !b.entries))return {Code::capacity,1,needed.entries};
    PreparedInput p{};p.input=in;p.rows=n;p.columns=g;p.genes=in.genes?in.gene_count:g;
    size_t ni=0,ne=0;
    if(a.layout!=Layout::dense) {p.spliced.offsets=b.indices+ni;ni+=g+1;p.spliced.entries=b.entries?b.entries+ne:nullptr;ne+=a.nnz;}
    if(u.layout!=Layout::dense) {p.unspliced.offsets=b.indices+ni;ni+=g+1;p.unspliced.entries=b.entries?b.entries+ne:nullptr;ne+=u.nnz;}
    if(w.layout!=Layout::dense) {
        p.graph.rows=b.indices+ni;ni+=n+1;p.graph.columns=b.indices+ni;ni+=n+1;
        p.graph.row_edges=b.entries?b.entries+ne:nullptr;ne+=w.nnz;p.graph.column_edges=b.entries?b.entries+ne:nullptr;ne+=w.nnz;p.graph.edges=w.nnz;
    }
    size_t* cursor=needed.indices?b.indices+ni:nullptr;
    if(p.spliced.offsets)columns(a,p.spliced,cursor);
    if(p.unspliced.offsets)columns(u,p.unspliced,cursor);
    if(p.graph.rows)graph(w,p.graph,cursor);
    result=p;return {};
}
}

Status validate(const Input& in) {
    size_t n,g;Status s=shape(in,n,g);if(!s)return s;
    auto mask_range=[](size_t count,ptrdiff_t stride) {
        if(!count)return true;
        if(count-1>size_t(PTRDIFF_MAX))return false;
        ptrdiff_t last;
        return !__builtin_mul_overflow(ptrdiff_t(count-1),stride,&last);
    };
    if(in.fit_mask && !mask_range(n,in.fit_mask_stride))return {Code::overflow,0};
    if(in.gene_mask && !mask_range(g,in.gene_mask_stride))return {Code::overflow,1};
    if(in.genes)for(size_t j=0;j<in.gene_count;++j)if(in.genes[j]>=g)return {Code::bounds,j};
    s=valid_source(source(in.spliced));if(!s)return s;
    s=valid_source(source(in.unspliced));if(!s)return s;
    return valid_source(source(in.weights));
}
Status preparation_size(const Input& in,PreparationSize& size) {
    size_t n,g;Status s=shape(in,n,g);if(!s)return s;
    Source a=source(in.spliced),u=source(in.unspliced),w=source(in.weights);
    return size_for(a,u,w,size)?Status{}:Status{Code::overflow};
}
Status prepare(const Input& in,const PreparationBuffers& b,PreparedInput& result) {
    Status s=validate(in);return s?bind(in,b,result):s;
}
Status prepare_unchecked(const Input& in,const PreparationBuffers& b,PreparedInput& result) { return bind(in,b,result); }

namespace {
// The value loads preserve unaligned storage and source arithmetic precision.
template<class C,class T> C number(const unsigned char* p) { return C(load<T>(p)); }
template<class C,class W> void add_column(const Source& w,const GraphIndex& graph,size_t k,C value,C* out,size_t stride) {
    if(w.layout==Layout::dense) {
        const auto* col=w.values+ptrdiff_t(k)*w.column_stride;
        for(size_t i=0;i<w.rows;++i)out[i*stride]+=number<C,W>(col+ptrdiff_t(i)*w.row_stride)*value;
    } else for(size_t p=graph.columns[k];p<graph.columns[k+1];++p) {
        Entry e=graph.column_edges[p];out[e.index*stride]+=number<C,W>(w.values+e.position*sizeof(W))*value;
    }
}
template<class C,class W,class T>
void apply_gene(const Source& w,const GraphIndex& graph,const Source& x,const ColumnIndex& index,size_t gene,C* out,size_t stride) {
    if(x.layout==Layout::dense) {
        const auto* col=x.values+ptrdiff_t(gene)*x.column_stride;
        for(size_t k=0;k<x.rows;++k)add_column<C,W>(w,graph,k,number<C,T>(col+ptrdiff_t(k)*x.row_stride),out,stride);
    } else {
        size_t p=index.offsets[gene],end=index.offsets[gene+1];
        while(p<end) {
            size_t k=index.entries[p].index;C value=0;
            do {value+=number<C,T>(x.values+index.entries[p].position*sizeof(T));++p;}while(p<end && index.entries[p].index==k);
            add_column<C,W>(w,graph,k,value,out,stride);
        }
    }
}
template<class C,class W> void apply_gene(const Source& w,const GraphIndex& graph,const Source& x,const ColumnIndex& index,size_t gene,C* out,size_t stride) {
    if(x.dtype==DType::float32)apply_gene<C,W,float>(w,graph,x,index,gene,out,stride);
    else apply_gene<C,W,double>(w,graph,x,index,gene,out,stride);
}
template<class C,class T> void source_column(const Source& x,const ColumnIndex& index,size_t gene,C* out) {
    if(x.layout==Layout::dense) {
        const auto* col=x.values+ptrdiff_t(gene)*x.column_stride;
        for(size_t i=0;i<x.rows;++i)out[i]=number<C,T>(col+ptrdiff_t(i)*x.row_stride);
    } else {
        for(size_t i=0;i<x.rows;++i)out[i]=0;
        for(size_t p=index.offsets[gene];p<index.offsets[gene+1];++p) {
            Entry e=index.entries[p];out[e.index]+=number<C,T>(x.values+e.position*sizeof(T));
        }
    }
}
template<class C> void source_column(const Source& x,const ColumnIndex& index,size_t gene,C* out) {
    if(x.dtype==DType::float32)source_column<C,float>(x,index,gene,out);
    else source_column<C,double>(x,index,gene,out);
}
template<class C,class W>
void second_column(const Source& w,const GraphIndex& graph,const C* s,const C* u,size_t n,
    C* ms,C* mu,C* mss,C* mus,C* muu,size_t stride) {
    for(size_t k=0;k<n;++k) {
        C x=s?s[k]:0,y=u?u[k]:0;
        if(x==0 && y==0)continue;
        C xx=mss?x*x:0,xy=mus?x*y:0,yy=muu?y*y:0;
        if(w.layout==Layout::dense) {
            const auto* col=w.values+ptrdiff_t(k)*w.column_stride;
            for(size_t i=0;i<n;++i) {
                C weight=number<C,W>(col+ptrdiff_t(i)*w.row_stride);size_t q=i*stride;
                if(ms)ms[q]+=weight*x;if(mu)mu[q]+=weight*y;
                if(mss)mss[q]+=weight*xx;if(mus)mus[q]+=weight*xy;if(muu)muu[q]+=weight*yy;
            }
        } else for(size_t p=graph.columns[k];p<graph.columns[k+1];++p) {
            Entry e=graph.column_edges[p];C weight=number<C,W>(w.values+e.position*sizeof(W));size_t q=e.index*stride;
            if(ms)ms[q]+=weight*x;if(mu)mu[q]+=weight*y;
            if(mss)mss[q]+=weight*xx;if(mus)mus[q]+=weight*xy;if(muu)muu[q]+=weight*yy;
        }
    }
}
template<class C,class W,class S,class U>
void original_columns(const Source& w,const GraphIndex& graph,const void* sx,ptrdiff_t ss,
    const void* ux,ptrdiff_t us,C* ms,C* mu,C* mss,C* mus,C* muu,size_t stride,bool second) {
    for(size_t i=0;i<w.rows;++i) {
        size_t q=i*stride;
        if(ms)ms[q]=0;if(mu)mu[q]=0;if(mss)mss[q]=0;if(mus)mus[q]=0;if(muu)muu[q]=0;
    }
    auto s=reinterpret_cast<const unsigned char*>(sx),u=reinterpret_cast<const unsigned char*>(ux);
    for(size_t k=0;k<w.rows;++k) {
        C x=s?number<C,S>(s+ptrdiff_t(k)*ss):0,y=u?number<C,U>(u+ptrdiff_t(k)*us):0;
        if(second && x==0 && y==0)continue;
        if(ms)add_column<C,W>(w,graph,k,x,ms,stride);
        if(mu)add_column<C,W>(w,graph,k,y,mu,stride);
        if(mss)add_column<C,W>(w,graph,k,x*x,mss,stride);
        if(mus)add_column<C,W>(w,graph,k,x*y,mus,stride);
        if(muu)add_column<C,W>(w,graph,k,y*y,muu,stride);
    }
}
template<class C,class W,class S>
void original_columns_u(const Source& w,const GraphIndex& graph,const void* sx,ptrdiff_t ss,
    const void* ux,ptrdiff_t us,DType ut,C* ms,C* mu,C* mss,C* mus,C* muu,size_t stride,bool second) {
    if(ut==DType::float32)original_columns<C,W,S,float>(w,graph,sx,ss,ux,us,ms,mu,mss,mus,muu,stride,second);
    else original_columns<C,W,S,double>(w,graph,sx,ss,ux,us,ms,mu,mss,mus,muu,stride,second);
}
template<class C,class W>
void original_columns_s(const Source& w,const GraphIndex& graph,const void* sx,ptrdiff_t ss,DType st,
    const void* ux,ptrdiff_t us,DType ut,C* ms,C* mu,C* mss,C* mus,C* muu,size_t stride,bool second) {
    if(st==DType::float32)original_columns_u<C,W,float>(w,graph,sx,ss,ux,us,ut,ms,mu,mss,mus,muu,stride,second);
    else original_columns_u<C,W,double>(w,graph,sx,ss,ux,us,ut,ms,mu,mss,mus,muu,stride,second);
}
template<class C>
void original_columns(const PreparedInput& p,const void* sx,ptrdiff_t ss,DType st,const void* ux,ptrdiff_t us,DType ut,
    C* ms,C* mu,C* mss,C* mus,C* muu,size_t stride,bool second) {
    if(!p.rows)return;
    Source w=source(p.input.weights);
    if(w.dtype==DType::float32)original_columns_s<C,float>(w,p.graph,sx,ss,st,ux,us,ut,ms,mu,mss,mus,muu,stride,second);
    else original_columns_s<C,double>(w,p.graph,sx,ss,st,ux,us,ut,ms,mu,mss,mus,muu,stride,second);
}
template<class C,class W,class X>
void dense_many(const Source& w,const GraphIndex& graph,const Source& x,const Input& input,
    size_t first,size_t count,C* out) {
    for(size_t i=0;i<w.rows;++i) {
        size_t begin=w.layout==Layout::dense?0:graph.rows[i];
        size_t end=w.layout==Layout::dense?w.rows:graph.rows[i+1];
        for(size_t p=begin;p<end;++p) {
            size_t k=w.layout==Layout::dense?p:graph.row_edges[p].index;
            C weight=w.layout==Layout::dense?number<C,W>(w.values+ptrdiff_t(i)*w.row_stride+ptrdiff_t(k)*w.column_stride)
                :number<C,W>(w.values+graph.row_edges[p].position*sizeof(W));
            const auto* row=x.values+ptrdiff_t(k)*x.row_stride;
            for(size_t j=0;j<count;++j) {
                size_t gene=input.genes?input.genes[first+j]:first+j;
                out[i*count+j]+=weight*number<C,X>(row+ptrdiff_t(gene)*x.column_stride);
            }
        }
    }
}
template<class C,class W>
void apply_many(const Source& w,const GraphIndex& graph,const Source& x,const ColumnIndex& index,
    const Input& input,size_t first,size_t count,C* out) {
    if(x.layout==Layout::dense) {
        if(x.dtype==DType::float32)dense_many<C,W,float>(w,graph,x,input,first,count,out);
        else dense_many<C,W,double>(w,graph,x,input,first,count,out);
    } else for(size_t j=0;j<count;++j) {
        size_t gene=input.genes?input.genes[first+j]:first+j;
        apply_gene<C,W>(w,graph,x,index,gene,out+j,count);
    }
}
template<class C> void calculate(const PreparedInput& weights,const PreparedInput& input,size_t first,size_t count,C* ms,C* mu) {
    Source w=source(weights.input.weights),s=source(input.input.spliced),u=source(input.input.unspliced);
    for(size_t i=0;i<input.rows*count;++i){ms[i]=0;mu[i]=0;}
    if(w.dtype==DType::float32) {
        apply_many<C,float>(w,weights.graph,s,input.spliced,input.input,first,count,ms);
        apply_many<C,float>(w,weights.graph,u,input.unspliced,input.input,first,count,mu);
    } else {
        apply_many<C,double>(w,weights.graph,s,input.spliced,input.input,first,count,ms);
        apply_many<C,double>(w,weights.graph,u,input.unspliced,input.input,first,count,mu);
    }
}
}

namespace reference {
template<class C> void moments(const PreparedInput& p,size_t first,size_t count,C* ms,C* mu) {
    if(!count || !p.rows)return;calculate(p,p,first,count,ms,mu);
}
template<class C> void moments_from(const PreparedInput& graph,const PreparedInput& input,size_t first,size_t count,C* ms,C* mu) {
    if(!count || !graph.rows)return;calculate(graph,input,first,count,ms,mu);
}
template<class C> void moments_from(const PreparedInput& p,const void* sx,ptrdiff_t ss,DType st,
    const void* ux,ptrdiff_t us,DType ut,C* ms,C* mu,size_t stride) {
    original_columns(p,sx,ss,st,ux,us,ut,ms,mu,(C*)nullptr,(C*)nullptr,(C*)nullptr,stride,false);
}
template<class C> void second_moments_from(const PreparedInput& p,const void* sx,ptrdiff_t ss,DType st,
    const void* ux,ptrdiff_t us,DType ut,C* ms,C* mu,C* mss,C* mus,C* muu,size_t stride) {
    original_columns(p,sx,ss,st,ux,us,ut,ms,mu,mss,mus,muu,stride,true);
}
template<class C> void second_moments(const PreparedInput& p,size_t first,size_t count,
    C* ms,C* mu,C* mss,C* mus,C* muu,C* source_s,C* source_u) {
    if(!count || !p.rows)return;
    Source w=source(p.input.weights),s=source(p.input.spliced),u=source(p.input.unspliced);
    for(size_t i=0;i<p.rows*count;++i) {
        if(ms)ms[i]=0;if(mu)mu[i]=0;if(mss)mss[i]=0;if(mus)mus[i]=0;if(muu)muu[i]=0;
    }
    bool need_s=ms||mss||mus,need_u=mu||mus||muu;
    if(!need_s && !need_u)return;
    for(size_t j=0;j<count;++j) {
        size_t gene=p.input.genes?p.input.genes[first+j]:first+j;
        if(need_s)source_column(s,p.spliced,gene,source_s);
        if(need_u)source_column(u,p.unspliced,gene,source_u);
        if(w.dtype==DType::float32)second_column<C,float>(w,p.graph,need_s?source_s:nullptr,need_u?source_u:nullptr,p.rows,
            ms?ms+j:nullptr,mu?mu+j:nullptr,mss?mss+j:nullptr,mus?mus+j:nullptr,muu?muu+j:nullptr,count);
        else second_column<C,double>(w,p.graph,need_s?source_s:nullptr,need_u?source_u:nullptr,p.rows,
            ms?ms+j:nullptr,mu?mu+j:nullptr,mss?mss+j:nullptr,mus?mus+j:nullptr,muu?muu+j:nullptr,count);
    }
}
#define VELOCITY_PREPARED(T) \
template void moments<T>(const PreparedInput&,size_t,size_t,T*,T*); \
template void moments_from<T>(const PreparedInput&,const PreparedInput&,size_t,size_t,T*,T*); \
template void moments_from<T>(const PreparedInput&,const void*,ptrdiff_t,DType,const void*,ptrdiff_t,DType,T*,T*,size_t); \
template void second_moments_from<T>(const PreparedInput&,const void*,ptrdiff_t,DType,const void*,ptrdiff_t,DType,T*,T*,T*,T*,T*,size_t); \
template void second_moments<T>(const PreparedInput&,size_t,size_t,T*,T*,T*,T*,T*,T*,T*);
VELOCITY_PREPARED(float)
VELOCITY_PREPARED(double)
#undef VELOCITY_PREPARED
}
}
