module;
#include <Cellerator/ops/numeric/binary16.hh>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <type_traits>
#include "cpu_kernels.hh"

module velocity;
import anndatapp;
#include "prepared.hh"

namespace velocity::detail {
namespace {

template<class T> T load(const std::byte* p) {
    T x;
    std::memcpy(&x, p, sizeof x);
    return x;
}

template<class C, class Storage> C numeric(const std::byte* p) {
    if constexpr (std::is_same_v<Storage, std::uint16_t>) {
        return C(cellerator::numeric::load_f16(load<Storage>(p)));
    } else return C(load<Storage>(p));
}

std::size_t integer(const std::byte* p, std::size_t width, std::size_t i) {
    p += i * width;
    switch (width) {
    case 1: return load<std::uint8_t>(p);
    case 2: return load<std::uint16_t>(p);
    case 4: return load<std::uint32_t>(p);
    case 8: return static_cast<std::size_t>(load<std::uint64_t>(p));
    default: throw std::invalid_argument("unsupported sparse integer width");
    }
}

void prefix(std::vector<std::size_t>& offsets) {
    for (std::size_t i = 1; i < offsets.size(); ++i) offsets[i] += offsets[i - 1];
}

Matrix acquire(NumericView view) {
    view.validate();
    if (view.rank() != 2) throw std::invalid_argument("matrix must have rank two");
    if (view.dtype() != StorageType::float16 && view.dtype() != StorageType::float32 && view.dtype() != StorageType::float64)
        throw std::invalid_argument("matrix values must be float16, float32 or float64");
    Matrix m;
    m.view = std::move(view); m.layout = m.view.layout(); m.dtype = m.view.dtype();
    if (m.layout == anndatapp::Layout::dense) {
        auto a = m.view.bytes();
        m.values = a.data(); m.row_stride = a.stride_bytes(0); m.column_stride = a.stride_bytes(1);
    } else {
        auto a = m.view.sparse_bytes();
        m.values = a.values(); m.indices = a.indices(); m.index_width = a.index_size(); m.nnz = a.nnz();
        auto major = m.layout == anndatapp::Layout::csr ? a.rows() : a.columns();
        m.offsets.resize(major + 1);
        for (std::size_t i = 0; i <= major; ++i) m.offsets[i] = integer(a.offsets(), a.offset_size(), i);
    }
    return m;
}

void index_columns(Matrix& m, std::size_t rows, std::size_t cols) {
    if (m.layout == anndatapp::Layout::dense) return;
    if (m.layout == anndatapp::Layout::csc) {
        // Sorted CSC already groups each logical coordinate. Unsorted CSC
        // needs only a structural row/position index, never a value shadow.
        bool sorted = true;
        for (std::size_t j = 0; j < cols && sorted; ++j) {
            auto first=m.offsets[j],last=m.offsets[j+1];
            if(first==last)continue;
            for(std::size_t p=first+1;p<last;++p)
                if(integer(m.indices,m.index_width,p)<integer(m.indices,m.index_width,p-1)) {sorted=false;break;}
        }
        if (sorted) return;
        m.entries.resize(m.nnz);
        for (std::size_t p = 0; p < m.nnz; ++p) m.entries[p] = {integer(m.indices,m.index_width,p),p};
        for (std::size_t j = 0; j < cols; ++j)
            std::sort(m.entries.begin()+m.offsets[j],m.entries.begin()+m.offsets[j+1],[](Entry a,Entry b) {
                return a.index < b.index || (a.index == b.index && a.position < b.position);
            });
        return;
    }
    std::vector<std::size_t> offsets(cols + 1);
    for (std::size_t p = 0; p < m.nnz; ++p) ++offsets[integer(m.indices, m.index_width, p) + 1];
    prefix(offsets);
    auto cursor = offsets;
    std::vector<Entry> entries(m.nnz);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t p = m.offsets[i]; p < m.offsets[i + 1]; ++p)
            entries[cursor[integer(m.indices, m.index_width, p)]++] = {i, p};
    m.offsets = std::move(offsets); m.entries = std::move(entries);
}

template<class T> void check_matrix(const Matrix& m, std::size_t rows, std::size_t cols) {
    if (m.layout == anndatapp::Layout::dense) {
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < cols; ++j)
                if (!std::isfinite(numeric<double, T>(m.values + std::ptrdiff_t(i)*m.row_stride + std::ptrdiff_t(j)*m.column_stride)))
                    throw std::invalid_argument("matrix contains a nonfinite value");
    } else {
        for (std::size_t p = 0; p < m.nnz; ++p)
            if (!std::isfinite(numeric<double, T>(m.values + p*sizeof(T))))
                throw std::invalid_argument("matrix contains a nonfinite value");
    }
}

void check_matrix(const Matrix& m, std::size_t rows, std::size_t cols) {
    if (m.dtype == StorageType::float16) check_matrix<std::uint16_t>(m, rows, cols);
    else if (m.dtype == StorageType::float32) check_matrix<float>(m, rows, cols);
    else check_matrix<double>(m, rows, cols);
}

template<class C, class W, class T, bool Selected> void dense_moments(const Matrix& graph, const Matrix& x,
    const std::vector<std::size_t>& rows, const std::vector<Entry>& edges,
    std::size_t n, std::size_t first, const std::size_t* genes, std::size_t count, std::size_t stride, C* out, bool compiled) {
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t p = rows[i]; p < rows[i + 1]; ++p) {
            auto edge = edges[p];
            C w = numeric<C, W>(graph.values + edge.position*sizeof(W));
            const auto* v = x.values + std::ptrdiff_t(edge.index)*x.row_stride;
            if constexpr(!Selected && std::is_same_v<C,T>) {
                const auto* source=v+std::ptrdiff_t(first)*x.column_stride;
                if(compiled && x.column_stride==sizeof(C) && reinterpret_cast<std::uintptr_t>(source)%alignof(C)==0) {
                    cpu::weighted_add(reinterpret_cast<const C*>(source),w,out+i*stride,count);continue;
                }
            }
            if constexpr(!Selected && std::is_same_v<C,float> && std::is_same_v<T,std::uint16_t>) {
                if(compiled && x.column_stride==sizeof(std::uint16_t)) {
                    float values[64];
                    for(std::size_t begin=0;begin<count;begin+=64) {
                        auto length=std::min<std::size_t>(64,count-begin);
                        cpu::half_to_float(reinterpret_cast<const std::uint16_t*>(v+std::ptrdiff_t(first+begin)*x.column_stride),sizeof(std::uint16_t),values,length);
                        cpu::weighted_add(values,w,out+i*stride+begin,length);
                    }
                    continue;
                }
            }
            for (std::size_t j = 0; j < count; ++j) {
                auto gene = Selected ? genes[j] : first + j;
                out[i*stride + j] += w * numeric<C, T>(v + std::ptrdiff_t(gene)*x.column_stride);
            }
        }
}

template<class I> std::size_t csc_row(const Matrix& x, std::size_t p) {
    return static_cast<std::size_t>(load<I>(x.indices + p*sizeof(I)));
}

// Coordinate duplicates are summed in the selected arithmetic before any
// weighting or nonlinear transform. Prepared order preserves original value
// order within a coordinate and needs no numerical scratch or value copy.
template<class C,class T,class I,bool Indexed,class Apply>
void logical_column(const Matrix& x,std::size_t gene,Apply&& apply) {
    std::size_t p=x.offsets[gene],end=x.offsets[gene+1];
    while(p<end) {
        auto row=[&](std::size_t q) {if constexpr(Indexed)return x.entries[q].index;else return csc_row<I>(x,q);};
        auto source=row(p);
        C value=0;
        do {
            auto position=Indexed ? x.entries[p].position : p;
            value+=numeric<C,T>(x.values+position*sizeof(T));
            ++p;
        } while(p<end && row(p)==source);
        apply(source,value);
    }
}

template<class C, class W, class T, bool Selected> void dense_graph_dense(const Matrix& graph, const Matrix& x,
    std::size_t n, std::size_t first, const std::size_t* genes, std::size_t count, std::size_t stride, C* out, bool compiled) {
    for (std::size_t i = 0; i < n; ++i) for (std::size_t k = 0; k < n; ++k) {
        auto weight = numeric<C, W>(graph.values + std::ptrdiff_t(i)*graph.row_stride + std::ptrdiff_t(k)*graph.column_stride);
        const auto* row = x.values + std::ptrdiff_t(k)*x.row_stride;
        if constexpr(!Selected && std::is_same_v<C,T>) {
            const auto* source=row+std::ptrdiff_t(first)*x.column_stride;
            if(compiled && x.column_stride==sizeof(C) && reinterpret_cast<std::uintptr_t>(source)%alignof(C)==0) {
                cpu::weighted_add(reinterpret_cast<const C*>(source),weight,out+i*stride,count);continue;
            }
        }
        if constexpr(!Selected && std::is_same_v<C,float> && std::is_same_v<T,std::uint16_t>) {
            if(compiled && x.column_stride==sizeof(std::uint16_t)) {
                float values[64];
                for(std::size_t begin=0;begin<count;begin+=64) {
                    auto length=std::min<std::size_t>(64,count-begin);
                    cpu::half_to_float(reinterpret_cast<const std::uint16_t*>(row+std::ptrdiff_t(first+begin)*x.column_stride),sizeof(std::uint16_t),values,length);
                    cpu::weighted_add(values,weight,out+i*stride+begin,length);
                }
                continue;
            }
        }
        for (std::size_t j = 0; j < count; ++j) {
            auto gene = Selected ? genes[j] : first + j;
            out[i*stride + j] += weight * numeric<C, T>(row + std::ptrdiff_t(gene)*x.column_stride);
        }
    }
}

template<class C, class W, class T, class I, bool Indexed, bool Selected>
void dense_graph_sparse(const Matrix& graph, const Matrix& x, std::size_t n,
    std::size_t first, const std::size_t* genes, std::size_t count, std::size_t stride, C* out, bool /*compiled*/) {
    for (std::size_t j = 0; j < count; ++j) {
        auto gene = Selected ? genes[j] : first + j;
        logical_column<C,T,I,Indexed>(x,gene,[&](std::size_t row,C value) {
            const auto* column = graph.values + std::ptrdiff_t(row)*graph.column_stride;
            for (std::size_t i = 0; i < n; ++i)
                out[i*stride + j] += numeric<C, W>(column + std::ptrdiff_t(i)*graph.row_stride)*value;
        });
    }
}

template<class C, class W, class T, class I, bool Indexed, bool Selected> void sparse_moments(const Matrix& graph, const Matrix& x,
    const std::vector<std::size_t>& columns, const std::vector<Entry>& edges,
    std::size_t first, const std::size_t* genes, std::size_t count, std::size_t stride, C* out, bool /*compiled*/) {
    for (std::size_t j = 0; j < count; ++j) {
        auto gene = Selected ? genes[j] : first + j;
        logical_column<C,T,I,Indexed>(x,gene,[&](std::size_t row,C value) {
            for (std::size_t q = columns[row]; q < columns[row + 1]; ++q) {
                auto edge = edges[q];
                out[edge.index*stride + j] += numeric<C, W>(graph.values + edge.position*sizeof(W))*value;
            }
        });
    }
}

template<class C, class W, class T, bool Selected> void apply_matrix(const Matrix& graph, const Matrix& x,
    const std::vector<std::size_t>& rows, const std::vector<Entry>& row_edges,
    const std::vector<std::size_t>& columns, const std::vector<Entry>& column_edges,
    std::size_t n, std::size_t first, const std::size_t* genes, std::size_t count, std::size_t stride, C* out, bool compiled) {
    if (graph.layout == anndatapp::Layout::dense) {
        if (x.layout == anndatapp::Layout::dense)
            return dense_graph_dense<C, W, T, Selected>(graph, x, n, first, genes, count, stride, out, compiled);
        if (x.layout == anndatapp::Layout::csr || !x.entries.empty())
            return dense_graph_sparse<C, W, T, std::uint8_t, true, Selected>(graph, x, n, first, genes, count, stride, out, compiled);
        switch (x.index_width) {
        case 1: return dense_graph_sparse<C, W, T, std::uint8_t, false, Selected>(graph, x, n, first, genes, count, stride, out, compiled);
        case 2: return dense_graph_sparse<C, W, T, std::uint16_t, false, Selected>(graph, x, n, first, genes, count, stride, out, compiled);
        case 4: return dense_graph_sparse<C, W, T, std::uint32_t, false, Selected>(graph, x, n, first, genes, count, stride, out, compiled);
        case 8: return dense_graph_sparse<C, W, T, std::uint64_t, false, Selected>(graph, x, n, first, genes, count, stride, out, compiled);
        default: throw std::invalid_argument("unsupported sparse integer width");
        }
    }
    if (x.layout == anndatapp::Layout::dense)
        return dense_moments<C, W, T, Selected>(graph, x, rows, row_edges, n, first, genes, count, stride, out, compiled);
    if (x.layout == anndatapp::Layout::csr || !x.entries.empty())
        return sparse_moments<C, W, T, std::uint8_t, true, Selected>(graph, x, columns, column_edges, first, genes, count, stride, out, compiled);
    switch (x.index_width) {
    case 1: return sparse_moments<C, W, T, std::uint8_t, false, Selected>(graph, x, columns, column_edges, first, genes, count, stride, out, compiled);
    case 2: return sparse_moments<C, W, T, std::uint16_t, false, Selected>(graph, x, columns, column_edges, first, genes, count, stride, out, compiled);
    case 4: return sparse_moments<C, W, T, std::uint32_t, false, Selected>(graph, x, columns, column_edges, first, genes, count, stride, out, compiled);
    case 8: return sparse_moments<C, W, T, std::uint64_t, false, Selected>(graph, x, columns, column_edges, first, genes, count, stride, out, compiled);
    default: throw std::invalid_argument("unsupported sparse integer width");
    }
}

template<class C, class W, bool Selected> void apply_matrix(const Matrix& graph, const Matrix& x,
    const std::vector<std::size_t>& rows, const std::vector<Entry>& row_edges,
    const std::vector<std::size_t>& columns, const std::vector<Entry>& column_edges,
    std::size_t n, std::size_t first, const std::size_t* genes, std::size_t count, std::size_t stride, C* out, bool compiled) {
    if (x.dtype == StorageType::float16)
        apply_matrix<C, W, std::uint16_t, Selected>(graph, x, rows, row_edges, columns, column_edges, n, first, genes, count, stride, out, compiled);
    else if (x.dtype == StorageType::float32)
        apply_matrix<C, W, float, Selected>(graph, x, rows, row_edges, columns, column_edges, n, first, genes, count, stride, out, compiled);
    else apply_matrix<C, W, double, Selected>(graph, x, rows, row_edges, columns, column_edges, n, first, genes, count, stride, out, compiled);
}

// Gather one logical source column. Accumulating coordinates first is required
// before weighting and nonlinear products when compressed input has duplicates.
template<class C, class T> void source_column(const Matrix& x, std::size_t n, std::size_t gene, C* out, bool compiled) {
    if (x.layout == anndatapp::Layout::dense) {
        const auto* p = x.values + std::ptrdiff_t(gene)*x.column_stride;
        if constexpr(std::is_same_v<C,float> && std::is_same_v<T,std::uint16_t>) {
            if(compiled) {cpu::half_to_float(reinterpret_cast<const std::uint16_t*>(p),x.row_stride,out,n);return;}
        }
        for (std::size_t i = 0; i < n; ++i)
            out[i] = numeric<C, T>(p + std::ptrdiff_t(i)*x.row_stride);
        return;
    }
    std::fill_n(out, n, C(0));
    if (x.layout == anndatapp::Layout::csr || !x.entries.empty()) {
        for (std::size_t p = x.offsets[gene]; p < x.offsets[gene + 1]; ++p) {
            auto e = x.entries[p];
            out[e.index] += numeric<C, T>(x.values + e.position*sizeof(T));
        }
    } else {
        for (std::size_t p = x.offsets[gene]; p < x.offsets[gene + 1]; ++p)
            out[integer(x.indices, x.index_width, p)] += numeric<C, T>(x.values + p*sizeof(T));
    }
}

template<class C> void source_column(const Matrix& x, std::size_t n, std::size_t gene, C* out, bool compiled) {
    if (x.dtype == StorageType::float16) source_column<C, std::uint16_t>(x, n, gene, out, compiled);
    else if (x.dtype == StorageType::float32) source_column<C, float>(x, n, gene, out, compiled);
    else source_column<C, double>(x, n, gene, out, compiled);
}

template<class C, class W> void second_products(const Matrix& graph, const std::vector<std::size_t>& columns,
    const std::vector<Entry>& edges, std::size_t n, std::size_t stride, std::size_t j,
    const C* s, const C* u, C* ms, C* mu, C* mss, C* mus, C* muu, bool compiled) {
    auto add=[&](std::size_t k,C x,C y,C xx,C xy,C yy) {
        if(x==0 && y==0)return;
        if (graph.layout == anndatapp::Layout::dense) {
            const auto* column = graph.values + std::ptrdiff_t(k)*graph.column_stride;
            for (std::size_t i = 0; i < n; ++i) {
                C w = numeric<C, W>(column + std::ptrdiff_t(i)*graph.row_stride);
                auto q = i*stride + j;
                if (ms) ms[q] += w*x; if (mu) mu[q] += w*y;
                if (mss) mss[q] += w*xx; if (mus) mus[q] += w*xy; if (muu) muu[q] += w*yy;
            }
        } else {
            for (std::size_t p = columns[k]; p < columns[k + 1]; ++p) {
                auto e = edges[p];
                C w = numeric<C, W>(graph.values + e.position*sizeof(W));
                auto q = e.index*stride + j;
                if (ms) ms[q] += w*x; if (mu) mu[q] += w*y;
                if (mss) mss[q] += w*xx; if (mus) mus[q] += w*xy; if (muu) muu[q] += w*yy;
            }
        }
    };
    if(compiled && (mss||mus||muu)) {
        C ss[8],su[8],uu[8];
        for(std::size_t base=0;base<n;base+=8) {
            auto length=std::min<std::size_t>(8,n-base);
            cpu::products(s?s+base:nullptr,u?u+base:nullptr,mss?ss:nullptr,mus?su:nullptr,muu?uu:nullptr,length);
            for(std::size_t k=0;k<length;++k)add(base+k,s?s[base+k]:0,u?u[base+k]:0,mss?ss[k]:0,mus?su[k]:0,muu?uu[k]:0);
        }
    } else for(std::size_t k=0;k<n;++k) {
        C x=s?s[k]:0,y=u?u[k]:0;
        add(k,x,y,mss?x*x:0,mus?x*y:0,muu?y*y:0);
    }
}

}

PreparedInputs::PreparedInputs(NumericView w, NumericView s, NumericView u)
    : graph_(acquire(std::move(w))), spliced_(acquire(std::move(s))), unspliced_(acquire(std::move(u))) {
    n_ = spliced_.view.extent(0); g_ = spliced_.view.extent(1);
    if (graph_.view.extent(0) != n_ || graph_.view.extent(1) != n_)
        throw std::invalid_argument("graph must be an n_obs by n_obs matrix");
    if (unspliced_.view.extent(0) != n_ || unspliced_.view.extent(1) != g_)
        throw std::invalid_argument("spliced and unspliced shapes differ");
    if (n_ == std::numeric_limits<std::size_t>::max() || g_ == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("matrix dimensions overflow offsets");
    if (graph_.layout != anndatapp::Layout::dense) {
    graph_rows_.resize(n_ + 1); graph_columns_.resize(n_ + 1);
    auto csr = graph_.layout == anndatapp::Layout::csr;
    for (std::size_t major = 0; major < n_; ++major)
        for (std::size_t p = graph_.offsets[major]; p < graph_.offsets[major + 1]; ++p) {
            auto minor = integer(graph_.indices, graph_.index_width, p);
            auto row = csr ? major : minor, col = csr ? minor : major;
            ++graph_rows_[row + 1]; ++graph_columns_[col + 1];
        }
    prefix(graph_rows_); prefix(graph_columns_);
    auto row_cursor = graph_rows_, col_cursor = graph_columns_;
    row_edges_.resize(graph_.nnz); column_edges_.resize(graph_.nnz);
    for (std::size_t major = 0; major < n_; ++major)
        for (std::size_t p = graph_.offsets[major]; p < graph_.offsets[major + 1]; ++p) {
            auto minor = integer(graph_.indices, graph_.index_width, p);
            auto row = csr ? major : minor, col = csr ? minor : major;
            row_edges_[row_cursor[row]++] = {col, p}; column_edges_[col_cursor[col]++] = {row, p};
        }
    // Dense-expression gathers must visit source cells in the same order as
    // sparse expressions and second moments. Keep duplicate edges separate and
    // preserve their original value positions; sort only shuffled row indexes.
    auto ordered=[](Entry a,Entry b) {
        return a.index<b.index || (a.index==b.index && a.position<b.position);
    };
    for(std::size_t row=0;row<n_;++row) {
        auto first=graph_rows_[row],last=graph_rows_[row+1];
        if(last-first<2)continue;
        auto begin=row_edges_.begin()+first,end=row_edges_.begin()+last;
        if(!std::is_sorted(begin,end,ordered))std::sort(begin,end,ordered);
    }
    }
    index_columns(spliced_, n_, g_); index_columns(unspliced_, n_, g_);
}

std::size_t PreparedInputs::prepared_bytes() const noexcept {
    auto matrix_bytes = [](const Matrix& m) {
        return m.offsets.capacity()*sizeof(std::size_t) + m.entries.capacity()*sizeof(Entry);
    };
    return matrix_bytes(graph_) + matrix_bytes(spliced_) + matrix_bytes(unspliced_)
        + (graph_rows_.capacity() + graph_columns_.capacity())*sizeof(std::size_t)
        + (row_edges_.capacity() + column_edges_.capacity())*sizeof(Entry);
}

void PreparedInputs::check_values() const {
    check_matrix(graph_, n_, n_); check_matrix(spliced_, n_, g_); check_matrix(unspliced_, n_, g_);
}

template<class C> void PreparedInputs::moments(std::size_t first, std::size_t count, C* ms, C* mu, bool compiled) const {
    calculate(first, nullptr, count, ms, mu, count, spliced_, unspliced_, compiled);
}
template<class C> void PreparedInputs::moments_selected(const std::size_t* genes, std::size_t count, C* ms, C* mu, bool compiled) const {
    calculate(0, genes, count, ms, mu, count, spliced_, unspliced_, compiled);
}
template<class C> void PreparedInputs::second_moments_selected(const std::size_t* genes, std::size_t count,
    C* ms, C* mu, C* mss, C* mus, C* muu, C* source_s, C* source_u, bool compiled) const {
    second_calculate(genes,count,ms,mu,mss,mus,muu,source_s,source_u,count,spliced_,unspliced_,compiled);
}
template<class C> void PreparedInputs::second_calculate(const std::size_t* genes, std::size_t count,
    C* ms, C* mu, C* mss, C* mus, C* muu, C* source_s, C* source_u, std::size_t stride,
    const Matrix& spliced, const Matrix& unspliced, bool compiled) const {
    if (!count || !n_) return;
    for (auto* output : {ms,mu,mss,mus,muu}) if (output)
        for (std::size_t i=0;i<n_;++i) std::fill_n(output+i*stride,count,C(0));
    bool need_s=ms||mss||mus, need_u=mu||mus||muu;
    if (!need_s && !need_u) return;
    for (std::size_t j=0;j<count;++j) {
        auto gene=genes ? genes[j] : j;
        if (need_s) source_column(spliced,n_,gene,source_s,compiled);
        if (need_u) source_column(unspliced,n_,gene,source_u,compiled);
        auto* s=need_s?source_s:nullptr;auto* u=need_u?source_u:nullptr;
        if (graph_.dtype==StorageType::float16)
            second_products<C,std::uint16_t>(graph_,graph_columns_,column_edges_,n_,stride,j,s,u,ms,mu,mss,mus,muu,compiled);
        else if (graph_.dtype==StorageType::float32)
            second_products<C,float>(graph_,graph_columns_,column_edges_,n_,stride,j,s,u,ms,mu,mss,mus,muu,compiled);
        else second_products<C,double>(graph_,graph_columns_,column_edges_,n_,stride,j,s,u,ms,mu,mss,mus,muu,compiled);
    }
}
template<class C> void PreparedInputs::calculate(std::size_t first, const std::size_t* genes, std::size_t count,
    C* ms, C* mu, std::size_t stride, const Matrix& spliced, const Matrix& unspliced, bool compiled) const {
    if (!count || !n_) return;
    if(genes) {
        bool consecutive=true;
        for(std::size_t j=1;j<count;++j)if(genes[j]!=genes[0]+j){consecutive=false;break;}
        if(consecutive){first=genes[0];genes=nullptr;}
    }
    for(std::size_t i=0;i<n_;++i) {std::fill_n(ms+i*stride,count,C(0));std::fill_n(mu+i*stride,count,C(0));}
    auto apply=[&]<class W,bool Selected>() {
        apply_matrix<C,W,Selected>(graph_,spliced,graph_rows_,row_edges_,graph_columns_,column_edges_,n_,first,genes,count,stride,ms,compiled);
        apply_matrix<C,W,Selected>(graph_,unspliced,graph_rows_,row_edges_,graph_columns_,column_edges_,n_,first,genes,count,stride,mu,compiled);
    };
    auto dispatch=[&]<class W>() {if(genes) apply.template operator()<W,true>();else apply.template operator()<W,false>();};
    if(graph_.dtype==StorageType::float16) dispatch.template operator()<std::uint16_t>();
    else if(graph_.dtype==StorageType::float32) dispatch.template operator()<float>();
    else dispatch.template operator()<double>();
}
template<class C> void PreparedInputs::moments_from(NumericView s,NumericView u,
    C* ms,C* mu,std::size_t stride,bool checked) const {
    auto a=acquire(s),b=acquire(u);auto count=s.extent(1);
    if(s.extent(0)!=n_ || u.extent(0)!=n_ || u.extent(1)!=count || stride<count)
        throw std::invalid_argument("alternate sources must have matching N by K shapes and destination stride >= K");
    index_columns(a,n_,count);index_columns(b,n_,count);
    if(checked) {check_matrix(a,n_,count);check_matrix(b,n_,count);}
    calculate(0,nullptr,count,ms,mu,stride,a,b,false);
}
template<class C> void PreparedInputs::second_moments_from(NumericView s,NumericView u,
    C* ms,C* mu,C* mss,C* mus,C* muu,C* source_s,C* source_u,std::size_t stride,bool checked) const {
    auto a=acquire(s),b=acquire(u);auto count=s.extent(1);
    if(s.extent(0)!=n_ || u.extent(0)!=n_ || u.extent(1)!=count || stride<count)
        throw std::invalid_argument("alternate sources must have matching N by K shapes and destination stride >= K");
    index_columns(a,n_,count);index_columns(b,n_,count);
    if(checked) {check_matrix(a,n_,count);check_matrix(b,n_,count);}
    second_calculate(nullptr,count,ms,mu,mss,mus,muu,source_s,source_u,stride,a,b,false);
}
#define VELOCITY_PREPARED_INSTANTIATE(C) \
template void PreparedInputs::moments<C>(std::size_t,std::size_t,C*,C*,bool) const; \
template void PreparedInputs::moments_selected<C>(const std::size_t*,std::size_t,C*,C*,bool) const; \
template void PreparedInputs::second_moments_selected<C>(const std::size_t*,std::size_t,C*,C*,C*,C*,C*,C*,C*,bool) const; \
template void PreparedInputs::moments_from<C>(NumericView,NumericView,C*,C*,std::size_t,bool) const; \
template void PreparedInputs::second_moments_from<C>(NumericView,NumericView,C*,C*,C*,C*,C*,C*,C*,std::size_t,bool) const;
VELOCITY_PREPARED_INSTANTIATE(float)
VELOCITY_PREPARED_INSTANTIATE(double)
#undef VELOCITY_PREPARED_INSTANTIATE
}
