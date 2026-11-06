#include <cmath>
#include <cstddef>
#include <cstdint>

import velocitypp;
import velocitypp.precision;

int main() {
    constexpr std::size_t cells = 4, genes = 1;
    auto spliced = anndatapp::Array::empty<float>({cells, genes});
    auto unspliced = anndatapp::Array::empty<double>({cells, genes});
    auto weights = anndatapp::Array::zeros<float>({cells, cells});
    auto* s = spliced.edit().dense<float>().data();
    auto* u = unspliced.edit().dense<double>().data();
    auto* w = weights.edit().dense<float>().data();
    const float s_values[cells] = {1, 2, 3, 4};
    const double u_values[cells] = {2, 2, 5, 7};
    for (std::size_t i = 0; i < cells; ++i) {
        s[i] = s_values[i];
        u[i] = u_values[i];
        w[i*cells + i] = 1;
    }

    velocitypp::PreparedInput input({spliced.view(), unspliced.view(), weights.view()});
    if (input.output_dtype() != anndatapp::DType::float64) return 1;

    // Compile and exercise the installed precision module against mixed native
    // FP32/FP64 AnnData views.
    velocitypp::precision::OriginalArraySource originals(spliced.view(), unspliced.view());
    auto source = originals.source();
    velocitypp::precision::OriginalColumn column;
    if (!source.acquire(source.context, 0, column) ||
        column.spliced.dtype() != anndatapp::DType::float32 ||
        column.unspliced.dtype() != anndatapp::DType::float64 ||
        column.spliced.extent(0) != cells || column.unspliced.extent(0) != cells)
        return 1;

    velocitypp::FitOptions fit_options;
    fit_options.trimming = velocitypp::Trimming::none;
    fit_options.min_r2 = -10;
    fit_options.min_ratio = 0;

#if defined(VELOCITYPP_CONSUMER_STOCHASTIC)
    velocitypp::StochasticRunOptions run_options;
    run_options.compute_precision = velocitypp::ComputePrecision::float32;
    run_options.output_dtype = anndatapp::DType::float32;
    velocitypp::StochasticWorkspace workspace(input, run_options);
    velocitypp::StochasticOptions options;
    options.initialization = fit_options;
    auto output = anndatapp::Array::zeros<float>({cells, genes});
    auto fit = velocitypp::reference::stochastic_into(input, workspace, output.edit(), options);
    auto status = fit.status.view().dense<std::uint8_t>();
    auto velocity = output.view().dense<float>();
    if (workspace.compute_precision() != velocitypp::ComputePrecision::float32 ||
        status.data()[0] != std::uint8_t(velocitypp::StochasticStatus::ok) ||
        !std::isfinite(fit.gamma.view().dense<float>().data()[0]))
        return 1;
    for (std::size_t i = 0; i < cells; ++i)
        if (!std::isfinite(velocity.data()[i])) return 1;
#else
    velocitypp::RunOptions run_options;
    run_options.compute_precision = velocitypp::ComputePrecision::float32;
    run_options.output_dtype = anndatapp::DType::float32;
    velocitypp::Workspace workspace(input, run_options);
    auto output = anndatapp::Array::zeros<float>({cells, genes});
    auto fit = velocitypp::reference::steady_state_into(input, workspace, output.edit(), fit_options);
    auto status = fit.status.view().dense<std::uint8_t>();
    auto velocity = output.view().dense<float>();
    if (workspace.compute_precision() != velocitypp::ComputePrecision::float32 ||
        status.data()[0] != std::uint8_t(velocitypp::FitStatus::ok) ||
        !std::isfinite(fit.gamma.view().dense<float>().data()[0]))
        return 1;
    for (std::size_t i = 0; i < cells; ++i)
        if (!std::isfinite(velocity.data()[i])) return 1;
#endif
    return 0;
}
