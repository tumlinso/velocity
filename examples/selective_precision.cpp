#include <stddef.h>
#include <stdint.h>

import velocitypp.precision;

namespace {

struct OriginalColumns {
    const double* spliced = nullptr;
    const double* unspliced = nullptr;
    size_t rows = 0, genes = 0;
};

bool acquire_original(void* context, size_t gene, velocitypp::precision::OriginalColumn& column) {
    auto& source = *static_cast<OriginalColumns*>(context);
    if (gene >= source.genes) return false;
    ptrdiff_t stride = ptrdiff_t(source.genes * sizeof(double));
    column = {source.spliced + gene, source.unspliced + gene, stride, stride,
        velocitypp::precision::OriginalType::float64,
        velocitypp::precision::OriginalType::float64, source.rows};
    return true;
}

velocitypp::Status promotion_status(velocitypp::precision::PrecisionCheck check) {
    if (check.code == velocitypp::precision::PrecisionCode::ok) return {};
    if (check.code == velocitypp::precision::PrecisionCode::insufficient_capacity)
        return {velocitypp::Code::capacity, check.location, check.required};
    return {velocitypp::Code::argument, check.location, check.required};
}

struct StochasticOperations {
    const velocitypp::PreparedInput* input = nullptr;
    velocitypp::Workspace32* workspace = nullptr;
    velocitypp::StochasticOptions options;
    velocitypp::StochasticFitResult32 result;
    velocitypp::precision::Promotion* promotion = nullptr;
    velocitypp::precision::ReferenceMoments<float>* replacement = nullptr;
    float* velocity = nullptr;
    float* second_velocity = nullptr;
    velocitypp::StochasticReferenceStages32 stages;
    velocitypp::StochasticFitResult32 completed;

    velocitypp::Status run() {
        auto status = velocitypp::bind_stages(*input, *workspace, options, result, stages);
        if (!status) return status;
        auto check = velocitypp::precision::initialize_promotion(*promotion);
        status = promotion_status(check);
        if (!status) return status;

        for (size_t first = 0; first < input->genes;) {
            size_t count = input->genes - first < workspace->block ? input->genes - first : workspace->block;
            auto moments = stages.initial_moments(first, count);
            replacement->initial = &moments;
            replacement->second = nullptr;
            check = velocitypp::precision::promote_initial(*promotion, first, count);
            status = promotion_status(check);
            if (!status) return status;
            auto fit = stages.initial_fit(moments);
            stages.accept_initialization(moments, fit);
            first += count;
        }
        stages.resolve_selection();

        for (size_t first = 0; first < input->genes;) {
            size_t count = input->genes - first < workspace->block ? input->genes - first : workspace->block;
            auto moments = stages.moments(first, count);
            replacement->initial = nullptr;
            replacement->second = &moments;
            check = velocitypp::precision::promote_second(*promotion, first, count);
            status = promotion_status(check);
            if (!status) return status;
            auto scales = stages.scales(moments);
            auto fit = stages.fit(moments, scales);
            velocitypp::reference::stochastic_residuals(moments, fit,
                velocity ? velocity + first : nullptr,
                second_velocity ? second_velocity + first : nullptr,
                ptrdiff_t(input->genes));
            stages.emit(moments, fit);
            first += count;
        }
        completed = stages.finish();
        return {};
    }
};

}

int main() {
    constexpr size_t cells = 8, genes = 2;
    constexpr size_t preparation_capacity = 128;
    constexpr size_t scalar_capacity = 256, mask_capacity = 64;

    auto spliced = anndatapp::Array::empty<float>({cells, genes});
    auto unspliced = anndatapp::Array::empty<float>({cells, genes});
    auto original_spliced = anndatapp::Array::empty<double>({cells, genes});
    auto original_unspliced = anndatapp::Array::empty<double>({cells, genes});
    auto weights = anndatapp::Array::zeros<float>({cells, cells});
    auto* s = spliced.edit().dense<float>().data();
    auto* u = unspliced.edit().dense<float>().data();
    auto* source_s = original_spliced.edit().dense<double>().data();
    auto* source_u = original_unspliced.edit().dense<double>().data();
    auto* w = weights.edit().dense<float>().data();
    for (size_t i = 0; i < cells; ++i) {
        w[i * cells + i] = 1;
        for (size_t j = 0; j < genes; ++j) {
            double x = 1.001 + .4 * double(i) + .2 * double(j);
            double y = 1.37 * x + .03 * double(i % 3);
            size_t k = i * genes + j;
            source_s[k] = x;
            source_u[k] = y;
            s[k] = float(x);
            u[k] = float(y);
        }
    }

    auto s_view = spliced.view();
    auto u_view = unspliced.view();
    auto w_view = weights.view();
    auto s_bytes = s_view.bytes();
    auto u_bytes = u_view.bytes();
    auto w_bytes = w_view.bytes();
    velocitypp::Input input{
        {&s_bytes, nullptr, s_view.dtype()},
        {&u_bytes, nullptr, u_view.dtype()},
        {&w_bytes, nullptr, w_view.dtype()}};

    velocitypp::PreparationSize preparation_needed;
    auto status = velocitypp::preparation_size(input, preparation_needed);
    if (!status) return int(status.code);
    size_t indices[preparation_capacity] = {};
    velocitypp::Entry entries[preparation_capacity] = {};
    if (preparation_needed.indices > preparation_capacity ||
        preparation_needed.entries > preparation_capacity)
        return int(velocitypp::Code::capacity);
    velocitypp::PreparedInput prepared;
    status = velocitypp::prepare(input,
        {indices, preparation_capacity, entries, preparation_capacity}, prepared);
    if (!status) return int(status.code);

    velocitypp::RunOptions run_options;
    run_options.block_genes = genes;
    run_options.unspliced_second = true;
    velocitypp::WorkspaceSize workspace_needed;
    status = velocitypp::workspace_size<float>(prepared, run_options, true, workspace_needed);
    if (!status) return int(status.code);
    float workspace_storage[scalar_capacity] = {};
    uint8_t workspace_mask[mask_capacity] = {};
    if (workspace_needed.scalars > scalar_capacity || workspace_needed.masks > mask_capacity)
        return int(velocitypp::Code::capacity);
    velocitypp::Workspace32 workspace;
    status = velocitypp::bind_workspace<float>(prepared, run_options, true,
        workspace_storage, scalar_capacity, workspace_mask, mask_capacity, workspace);
    if (!status) return int(status.code);

    float initial_gamma[genes] = {}, initial_offset[genes] = {}, initial_r2[genes] = {};
    bool selected[genes] = {};
    uint8_t initial_fit_status[genes] = {};
    velocitypp::FitResult32 initialization{
        initial_gamma, initial_offset, initial_r2, selected, initial_fit_status, genes, false};
    float gamma[genes] = {}, offset[genes] = {}, offset2[genes] = {};
    float first_scale[genes] = {}, second_scale[genes] = {}, objective[genes] = {};
    uint8_t fit_status[genes] = {};
    velocitypp::StochasticFitResult32 result{
        initialization, gamma, offset, offset2, first_scale, second_scale,
        objective, fit_status, genes, false};
    float velocity[cells * genes] = {}, second_velocity[cells * genes] = {};
    velocitypp::StochasticOptions options;
    options.initialization.trimming = velocitypp::Trimming::none;
    options.initialization.min_r2 = -1;
    options.initialization.min_ratio = 0;

    OriginalColumns original_columns{source_s, source_u, cells, genes};
    const uint8_t promote[genes] = {1, 0};
    uint8_t promotion_state[genes] = {};
    velocitypp::precision::ReferenceMoments<float> replacement{&prepared, nullptr, nullptr};
    velocitypp::precision::Promotion promotion{
        nullptr, genes, genes, cells, promote, genes, promotion_state, genes, 0,
        {&original_columns, acquire_original},
        {&replacement, velocitypp::precision::replace_reference32}};
    StochasticOperations operations{
        &prepared, &workspace, options, result, &promotion, &replacement,
        velocity, second_velocity};

    status = velocitypp::stochastic_with(operations, prepared.genes, workspace.block);
    if (!status) return int(status.code);
    if (promotion.promoted != 1 ||
        promotion_state[0] != uint8_t(velocitypp::precision::PromotionStatus::promoted) ||
        promotion_state[1] != uint8_t(velocitypp::precision::PromotionStatus::not_requested) ||
        operations.completed.status[0] != uint8_t(velocitypp::StochasticStatus::ok))
        return 1;
    for (size_t i = 0; i < genes; ++i)
        if (!__builtin_isfinite(gamma[i])) return 1;
    for (size_t i = 0; i < cells * genes; ++i)
        if (!__builtin_isfinite(velocity[i]) || !__builtin_isfinite(second_velocity[i])) return 1;
    return 0;
}
