#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace velocity_benchmark {

inline std::size_t checked_bytes(std::size_t count, std::size_t width) {
    if (count && width > std::numeric_limits<std::size_t>::max() / count)
        throw std::runtime_error("benchmark buffer byte count overflows size_t");
    return count * width;
}

inline std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::runtime_error("benchmark buffer byte count overflows size_t");
    return a + b;
}

inline void require(velocitypp::Status status, const char* operation) {
    if (status) return;
    throw std::runtime_error(std::string(operation) + " failed (code="
        + std::to_string(static_cast<unsigned>(status.code)) + ", location="
        + std::to_string(status.location) + ", required="
        + std::to_string(status.required) + ")");
}

// Retain the byte descriptors whose addresses AcquiredMatrix borrows.
struct Acquired {
    anndatapp::ArrayView view;
    std::optional<velocitypp::DenseInput> dense;
    std::optional<velocitypp::SparseInput> sparse;
    velocitypp::AcquiredMatrix matrix;

    explicit Acquired(anndatapp::ArrayView source) : view(source) {
        matrix.dtype = view.dtype();
        matrix.layout = view.layout();
        if (matrix.layout == anndatapp::Layout::dense) {
            dense.emplace(view.bytes());
            matrix.dense = &*dense;
        } else {
            matrix.index_type = view.index_dtype();
            matrix.offset_type = view.offset_dtype();
            sparse.emplace(view.sparse_bytes());
            matrix.sparse = &*sparse;
        }
    }

    Acquired(const Acquired&) = delete;
    Acquired& operator=(const Acquired&) = delete;
    Acquired(Acquired&&) = delete;
    Acquired& operator=(Acquired&&) = delete;
};

struct PreparedStorage {
    std::vector<std::size_t> indices;
    std::vector<velocitypp::Entry> entries;
    velocitypp::PreparedInput input;

    explicit PreparedStorage(const velocitypp::Input& source) {
        velocitypp::PreparationSize size{};
        require(velocitypp::preparation_size(source, size), "velocitypp preparation sizing");
        indices.resize(size.indices);
        entries.resize(size.entries);
        velocitypp::PreparationBuffers buffers{
            indices.data(), indices.size(), entries.data(), entries.size()};
        require(velocitypp::prepare(source, buffers, input), "velocitypp preparation");
    }

    std::size_t bytes() const {
        return checked_add(checked_bytes(indices.capacity(), sizeof(std::size_t)),
                           checked_bytes(entries.capacity(), sizeof(velocitypp::Entry)));
    }
};

template<class T> struct WorkspaceStorage {
    velocitypp::WorkspaceSize size{};
    std::vector<T> scalars;
    std::vector<std::uint8_t> masks;
    velocitypp::Workspace<T> workspace{};

    WorkspaceStorage(const velocitypp::PreparedInput& input,
                     velocitypp::RunOptions options, bool stochastic) {
        require(velocitypp::workspace_size<T>(input, options, stochastic, size),
                "velocitypp workspace sizing");
        scalars.resize(size.scalars);
        masks.resize(size.masks);
        require(velocitypp::bind_workspace<T>(input, options, stochastic,
                    scalars.data(), scalars.size(), masks.data(), masks.size(), workspace),
                "velocitypp workspace binding");
    }

    std::size_t bytes() const {
        return checked_add(checked_bytes(scalars.capacity(), sizeof(T)), masks.capacity());
    }
};

template<class T, class O>
void fit_workspace_budget(const velocitypp::PreparedInput& input,
                          velocitypp::RunOptions& options, bool stochastic,
                          std::size_t output_slots) {
    for (;;) {
        velocitypp::WorkspaceSize size{};
        require(velocitypp::workspace_size<T>(input, options, stochastic, size),
                "velocitypp workspace sizing");
        const auto output_bytes = checked_bytes(
            checked_bytes(input.rows, size.block), checked_bytes(output_slots, sizeof(O)));
        if (size.bytes <= options.workspace_bytes
            && output_bytes <= options.workspace_bytes - size.bytes) {
            options.block_genes = size.block;
            return;
        }
        if (size.block <= 1)
            throw std::runtime_error("combined workspace and output block exceed benchmark workspace budget");
        options.block_genes = size.block - 1;
    }
}

template<class T> struct FitStorage {
    std::vector<T> gamma, offset, r2;
    std::unique_ptr<bool[]> genes;
    std::vector<std::uint8_t> status;

    explicit FitStorage(std::size_t count)
        : gamma(count), offset(count), r2(count), genes(new bool[count]{}), status(count) {}

    velocitypp::BasicFitResult<T> result() {
        return {gamma.data(), offset.data(), r2.data(), genes.get(), status.data(), gamma.size(), false};
    }

    std::size_t bytes() const {
        const auto scalar_capacity = checked_add(checked_add(gamma.capacity(), offset.capacity()), r2.capacity());
        auto n = checked_add(checked_bytes(scalar_capacity, sizeof(T)),
                             checked_bytes(status.capacity(), sizeof(std::uint8_t)));
        return checked_add(n, checked_bytes(gamma.size(), sizeof(bool)));
    }
};

template<class T> struct StochasticFitStorage {
    FitStorage<T> initialization;
    std::vector<T> gamma, offset, offset2, first_scale, second_scale, objective;
    std::vector<std::uint8_t> status;

    explicit StochasticFitStorage(std::size_t count)
        : initialization(count), gamma(count), offset(count), offset2(count), first_scale(count),
          second_scale(count), objective(count), status(count) {}

    velocitypp::BasicStochasticFitResult<T> result() {
        return {initialization.result(), gamma.data(), offset.data(), offset2.data(),
                first_scale.data(), second_scale.data(), objective.data(), status.data(),
                gamma.size(), false};
    }

    std::size_t bytes() const {
        auto scalar_count = checked_add(gamma.capacity(), offset.capacity());
        scalar_count = checked_add(scalar_count, offset2.capacity());
        scalar_count = checked_add(scalar_count, first_scale.capacity());
        scalar_count = checked_add(scalar_count, second_scale.capacity());
        scalar_count = checked_add(scalar_count, objective.capacity());
        auto n = checked_add(checked_bytes(scalar_count, sizeof(T)),
                             checked_bytes(status.capacity(), sizeof(std::uint8_t)));
        return checked_add(n, initialization.bytes());
    }
};

} // namespace velocity_benchmark
