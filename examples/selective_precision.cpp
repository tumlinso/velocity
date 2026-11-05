#include <cstddef>
#include <Cellerator/ops/numeric/binary16.hh>
#include <cstdint>
#include <vector>
#include <iostream>

import velocity.precision;

int main() {
    constexpr std::size_t cells = 8, genes = 2;
    auto s = anndatapp::Array::empty<float>({cells, genes});
    auto u = anndatapp::Array::empty<float>({cells, genes});
    std::vector<std::uint16_t> hs(cells * genes), hu(cells * genes);
    auto w = anndatapp::Array::zeros<float>({cells, cells});
    for (std::size_t i = 0; i < cells; ++i) {
        w.edit().dense<float>().data()[i*cells+i] = 1;
        for (std::size_t j = 0; j < genes; ++j) {
            float x = 1.001f + .4f*float(i) + .2f*float(j);
            float y = 1.37f*x + .03f*float(i%3);
            s.edit().dense<float>().data()[i*genes+j] = x;
            u.edit().dense<float>().data()[i*genes+j] = y;
            hs[i*genes+j] = cellerator::numeric::round_f16(x);
            hu[i*genes+j] = cellerator::numeric::round_f16(y);
        }
    }

    auto hs_view = velocity::NumericView::borrow_dense_f16(hs.data(), {cells, genes}, {std::ptrdiff_t(genes), 1});
    auto hu_view = velocity::NumericView::borrow_dense_f16(hu.data(), {cells, genes}, {std::ptrdiff_t(genes), 1});
    velocity::PreparedInput input({hs_view, hu_view, w.view()});
    velocity::StochasticRunOptions run;
    run.compute_precision = velocity::ComputePrecision::float32;
    velocity::StochasticWorkspace workspace(input, run);
    velocity::StochasticReferenceStages32 reference(input, workspace);
    velocity::precision::OriginalArraySource originals(s.view(), u.view());
    const std::uint8_t promotion_mask[genes]{1, 0};
    velocity::precision::PromotionOptions promotion{promotion_mask, genes};
    velocity::precision::SelectiveStochasticStages32 stages(
        input, workspace, reference, originals.source(), promotion);
    auto result = velocity::stochastic_with(stages, input.n_genes(), workspace.block_genes());
    auto gamma = result.gamma.view().dense<float>();
    auto status = stages.promotion_status();
    for (std::size_t j = 0; j < genes; ++j)
        std::cout << "gene " << j << ": gamma=" << gamma.data()[j] << ", source="
                  << (status[j] == std::uint8_t(velocity::precision::PromotionStatus::promoted)
                      ? "original" : "FP16") << '\n';
}
