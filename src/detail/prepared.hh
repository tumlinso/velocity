#pragma once

// Include <cstddef>, <vector>, and <memory> before this header in the global
// module fragment; import anndata before including it in a module unit.
namespace velocity::detail {

struct Entry {
    std::size_t index, position;
};

struct Matrix {
    NumericView view;
    anndatapp::Layout layout = anndatapp::Layout::dense;
    StorageType dtype = StorageType::float32;
    const std::byte* values = nullptr;
    const std::byte* indices = nullptr;
    std::size_t index_width = 0, nnz = 0;
    std::ptrdiff_t row_stride = 0, column_stride = 0;
    // Sorted CSC uses its source indices directly. CSR and unsorted CSC
    // have a column index of row/original-position pairs, without value copies.
    std::vector<std::size_t> offsets;
    std::vector<Entry> entries;
};

struct PreparedInputs {
    PreparedInputs(NumericView w, NumericView s, NumericView u);

    [[nodiscard]] inline std::size_t n_obs() const noexcept { return n_; }
    [[nodiscard]] inline std::size_t n_vars() const noexcept { return g_; }
    [[nodiscard]] std::size_t prepared_bytes() const noexcept;
    void check_values() const;
    // Caller validates range, output capacities and non-aliasing; outputs are
    // row-major N x count. This trusted call performs no allocation.
    template<class C> void moments(std::size_t first, std::size_t count, C* ms, C* mu, bool compiled = false) const;
    template<class C> void moments_selected(const std::size_t* genes, std::size_t count, C* ms, C* mu, bool compiled = false) const;
    // Each requested output is row-major N x count. Unrequested fields are
    // null; source_s/source_u are disjoint caller-owned scratch of N doubles.
    // Sparse duplicate coordinates are summed before weighting and products.
    template<class C> void second_moments_selected(const std::size_t* genes, std::size_t count,
        C* ms, C* mu, C* mss, C* mus, C* muu,
        C* source_s, C* source_u, bool compiled = false) const;

    template<class C> void moments_from(NumericView s,NumericView u,C* ms,C* mu,std::size_t stride,bool checked) const;
    template<class C> void second_moments_from(NumericView s,NumericView u,C* ms,C* mu,C* mss,C* mus,C* muu,C* source_s,C* source_u,std::size_t stride,bool checked) const;

private:
    std::size_t n_ = 0, g_ = 0;
    Matrix graph_, spliced_, unspliced_;
    // Graph entries map a row to source cells and a source cell to rows.
    // position indexes the unchanged graph values in their original storage.
    std::vector<std::size_t> graph_rows_, graph_columns_;
    std::vector<Entry> row_edges_, column_edges_;
    template<class C> void calculate(std::size_t first,const std::size_t* genes,std::size_t count,C* ms,C* mu,
        std::size_t stride,const Matrix& s,const Matrix& u,bool compiled) const;
    template<class C> void second_calculate(const std::size_t* genes,std::size_t count,C* ms,C* mu,C* mss,C* mus,C* muu,
        C* source_s,C* source_u,std::size_t stride,const Matrix& s,const Matrix& u,bool compiled) const;
};

}
