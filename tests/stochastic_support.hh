#pragma once
#include "support.hh"
#include <memory>
#include <stdexcept>
#include <vector>

inline void require_status(velocitypp::Status s) {
    if (!s) throw std::runtime_error("velocitypp status " + std::to_string(int(s.code)));
}
struct PreparedStorage {
    Acquired spliced, unspliced, weights;
    std::vector<size_t> indices;
    std::vector<velocitypp::Entry> entries;
    velocitypp::PreparedInput value;
    PreparedStorage(anndatapp::ArrayView s, anndatapp::ArrayView u, anndatapp::ArrayView w,
                    const size_t* genes=nullptr, size_t gene_count=0,
                    const uint8_t* fit_mask=nullptr, const uint8_t* gene_mask=nullptr,
                    ptrdiff_t fit_stride=1, ptrdiff_t gene_stride=1)
        : spliced(s), unspliced(u), weights(w) {
        velocitypp::Input input{spliced.matrix,unspliced.matrix,weights.matrix};
        input.genes=genes; input.gene_count=gene_count; input.fit_mask=fit_mask; input.gene_mask=gene_mask;
        input.fit_mask_stride=fit_stride; input.gene_mask_stride=gene_stride;
        velocitypp::PreparationSize size;
        require_status(velocitypp::preparation_size(input,size));
        indices.resize(size.indices); entries.resize(size.entries);
        require_status(velocitypp::prepare(input,{indices.data(),indices.size(),entries.data(),entries.size()},value));
    }
};
template<class T> struct StochasticStorage {
    std::vector<T> scalars, ig, io, ir, gamma, offset, offset2, first, second, objective;
    std::vector<uint8_t> masks, initial_status, status;
    std::unique_ptr<bool[]> selected;
    velocitypp::Workspace<T> workspace;
    velocitypp::BasicStochasticFitResult<T> result;
    StochasticStorage(const velocitypp::PreparedInput& p, velocitypp::RunOptions options = {}) {
        velocitypp::WorkspaceSize size;
        require_status(velocitypp::workspace_size<T>(p,options,true,size));
        scalars.resize(size.scalars); masks.resize(size.masks);
        require_status(velocitypp::bind_workspace(p,options,true,scalars.data(),scalars.size(),masks.data(),masks.size(),workspace));
        auto n=p.genes;
        ig.resize(n);io.resize(n);ir.resize(n);gamma.resize(n);offset.resize(n);offset2.resize(n);
        first.resize(n);second.resize(n);objective.resize(n);initial_status.resize(n);status.resize(n);
        selected=std::make_unique<bool[]>(n);
        result.initialization={ig.data(),io.data(),ir.data(),selected.get(),initial_status.data(),n};
        result.gamma=gamma.data();result.offset=offset.data();result.offset2=offset2.data();
        result.first_scale=first.data();result.second_scale=second.data();result.objective=objective.data();
        result.status=status.data();result.capacity=n;
    }
    auto& run(const velocitypp::PreparedInput& p, velocitypp::StochasticOptions options = {}) {
        require_status(velocitypp::reference::stochastic(p,workspace,result,options)); return result;
    }
};
