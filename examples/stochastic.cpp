#include <stddef.h>
#include <stdint.h>

import velocitypp;

int main() {
    constexpr size_t cells = 4, genes = 1;
    constexpr size_t preparation_capacity = 64;
    constexpr size_t scalar_capacity = 256, mask_capacity = 64;

    auto spliced = anndatapp::Array::empty<double>({cells, genes});
    auto unspliced = anndatapp::Array::empty<double>({cells, genes});
    auto weights = anndatapp::Array::zeros<double>({cells, cells});
    auto* s = spliced.edit().dense<double>().data();
    auto* u = unspliced.edit().dense<double>().data();
    auto* w = weights.edit().dense<double>().data();
    for (size_t i = 0; i < cells; ++i) {
        s[i] = double(i + 1);
        u[i] = 2 * s[i] + (i % 2 ? .3 : -.2);
        w[i * cells + i] = 1;
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

    velocitypp::PreparationSize needed;
    auto status = velocitypp::preparation_size(input, needed);
    if (!status) return int(status.code);
    size_t indices[preparation_capacity] = {};
    velocitypp::Entry entries[preparation_capacity] = {};
    if (needed.indices > preparation_capacity || needed.entries > preparation_capacity)
        return int(velocitypp::Code::capacity);
    velocitypp::PreparedInput prepared;
    status = velocitypp::prepare(input,
        {indices, preparation_capacity, entries, preparation_capacity}, prepared);
    if (!status) return int(status.code);

    velocitypp::RunOptions run_options;
    run_options.block_genes = genes;
    velocitypp::WorkspaceSize workspace_needed;
    status = velocitypp::workspace_size<double>(prepared, run_options, true, workspace_needed);
    if (!status) return int(status.code);
    double workspace_storage[scalar_capacity] = {};
    uint8_t workspace_mask[mask_capacity] = {};
    if (workspace_needed.scalars > scalar_capacity || workspace_needed.masks > mask_capacity)
        return int(velocitypp::Code::capacity);
    velocitypp::Workspace<double> workspace;
    status = velocitypp::bind_workspace<double>(prepared, run_options, true,
        workspace_storage, scalar_capacity, workspace_mask, mask_capacity, workspace);
    if (!status) return int(status.code);

    double initial_gamma[genes] = {}, initial_offset[genes] = {}, initial_r2[genes] = {};
    bool selected[genes] = {};
    uint8_t initial_status[genes] = {};
    velocitypp::FitResult initialization{
        initial_gamma, initial_offset, initial_r2, selected, initial_status, genes, false};
    double gamma[genes] = {}, offset[genes] = {}, offset2[genes] = {};
    double first_scale[genes] = {}, second_scale[genes] = {}, objective[genes] = {};
    uint8_t fit_status[genes] = {};
    velocitypp::StochasticFitResult result{
        initialization, gamma, offset, offset2, first_scale, second_scale,
        objective, fit_status, genes, false};
    double velocity[cells * genes] = {}, second_velocity[cells * genes] = {};
    velocitypp::StochasticOptions options;
    options.initialization.trimming = velocitypp::Trimming::none;
    options.initialization.min_r2 = -1;
    options.initialization.min_ratio = 0;

    status = velocitypp::reference::stochastic_into(prepared, workspace, result,
        velocity, cells * genes, second_velocity, cells * genes, genes, options);
    if (!status) return int(status.code);
    if (initial_status[0] != uint8_t(velocitypp::FitStatus::ok) ||
        fit_status[0] != uint8_t(velocitypp::StochasticStatus::ok) ||
        !__builtin_isfinite(gamma[0]))
        return 1;
    for (size_t i = 0; i < cells * genes; ++i)
        if (!__builtin_isfinite(velocity[i]) || !__builtin_isfinite(second_velocity[i]))
            return 1;
    return 0;
}
