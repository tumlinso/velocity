#pragma once
// Development-only: retains already acquired AnnData++ byte descriptors.
// Construct before measuring/reference execution; do not move after binding.
#include <optional>

struct Acquired {
    anndatapp::ArrayView view;
    std::optional<velocitypp::DenseInput> dense;
    std::optional<velocitypp::SparseInput> sparse;
    velocitypp::AcquiredMatrix matrix;
    explicit Acquired(anndatapp::ArrayView input) : view(input) {
        matrix.dtype=view.dtype(); matrix.layout=view.layout();
        if(matrix.layout==anndatapp::Layout::dense) {
            dense.emplace(view.bytes()); matrix.dense=&*dense;
        } else {
            sparse.emplace(view.sparse_bytes()); matrix.sparse=&*sparse;
            matrix.index_type=view.index_dtype(); matrix.offset_type=view.offset_dtype();
        }
    }
    Acquired(const Acquired&)=delete;
    Acquired& operator=(const Acquired&)=delete;
    Acquired(Acquired&&)=delete;
    Acquired& operator=(Acquired&&)=delete;
};
