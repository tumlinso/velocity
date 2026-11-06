#include <cstddef>
#include <cstdint>
#include <iostream>

import velocitypp.precision;

int main() {
    constexpr std::size_t cells = 8, genes = 2;
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

    for (std::size_t i = 0; i < cells; ++i) {
        w[i*cells + i] = 1;
        for (std::size_t j = 0; j < genes; ++j) {
            const double x = 1.001 + .4*double(i) + .2*double(j);
            const double y = 1.37*x + .03*double(i % 3);
            const auto k = i*genes + j;
            source_s[k] = x;
            source_u[k] = y;
            s[k] = static_cast<float>(x);
            u[k] = static_cast<float>(y);
        }
    }

    velocitypp::PreparedInput input({spliced.view(), unspliced.view(), weights.view()});
    velocitypp::StochasticRunOptions run;
    run.compute_precision = velocitypp::ComputePrecision::float64;
    velocitypp::StochasticWorkspace workspace(input, run);
    velocitypp::StochasticReferenceStages reference(input, workspace);
    velocitypp::precision::OriginalArraySource originals(original_spliced.view(), original_unspliced.view());
    const std::uint8_t promotion_mask[genes]{1, 0};
    velocitypp::precision::PromotionOptions promotion{promotion_mask, genes};
    velocitypp::precision::SelectiveStochasticStages stages(
        input, workspace, reference, originals.source(), promotion);
    auto result = velocitypp::stochastic_with(stages, input.n_genes(), workspace.block_genes());
    auto gamma = result.gamma.view().dense<double>();
    const auto* status = stages.promotion_status();
    for (std::size_t j = 0; j < genes; ++j)
        std::cout << "gene " << j << ": gamma=" << gamma.data()[j] << ", source="
                  << (status[j] == std::uint8_t(velocitypp::precision::PromotionStatus::promoted)
                      ? "original FP64" : "baseline FP32") << '\n';
}
