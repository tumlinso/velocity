#include <stddef.h>
#include <stdint.h>

import velocitypp;

int main() {
    constexpr size_t cells = 3, genes = 1;
    constexpr size_t preparation_capacity = 64;
    constexpr size_t scalar_capacity = 128, mask_capacity = 64;

    auto spliced = anndatapp::Array::empty<double>({cells, genes});
    auto unspliced = anndatapp::Array::empty<double>({cells, genes});
    auto weights = anndatapp::Array::zeros<double>({cells, cells});
    auto* s = spliced.edit().dense<double>().data();
    auto* u = unspliced.edit().dense<double>().data();
    auto* w = weights.edit().dense<double>().data();
    for (size_t i = 0; i < cells; ++i) {
        s[i] = double(i + 1);
        u[i] = 2 * s[i];
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
    status = velocitypp::workspace_size<double>(prepared, run_options, false, workspace_needed);
    if (!status) return int(status.code);
    double workspace_storage[scalar_capacity] = {};
    uint8_t workspace_mask[mask_capacity] = {};
    if (workspace_needed.scalars > scalar_capacity || workspace_needed.masks > mask_capacity)
        return int(velocitypp::Code::capacity);
    velocitypp::Workspace<double> workspace;
    status = velocitypp::bind_workspace<double>(prepared, run_options, false,
        workspace_storage, scalar_capacity, workspace_mask, mask_capacity, workspace);
    if (!status) return int(status.code);

    double gamma[genes] = {}, offset[genes] = {}, r2[genes] = {};
    bool selected[genes] = {};
    uint8_t fit_status[genes] = {};
    velocitypp::FitResult result{gamma, offset, r2, selected, fit_status, genes, false};
    double velocity[cells * genes] = {};
    velocitypp::FitOptions fit_options;
    fit_options.trimming = velocitypp::Trimming::none;
    fit_options.min_r2 = -1;
    fit_options.min_ratio = 0;

    status = velocitypp::reference::steady_state_into(prepared, workspace, result,
        velocity, cells * genes, genes, fit_options);
    if (!status) return int(status.code);
    return fit_status[0] == uint8_t(velocitypp::FitStatus::ok) &&
        __builtin_fabs(gamma[0] - 2) < 1e-12 ? 0 : 1;
}
