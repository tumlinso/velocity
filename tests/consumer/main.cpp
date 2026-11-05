#include <Cellerator/ops/numeric/binary16.hh>
#include <cstdint>
#include <vector>

import velocity;

int main() {
    auto a = anndatapp::Array::zeros<double>({2, 1});
    const double x[] = {1, 2}, y[] = {2, 4};
    double scratch[2];
    auto f = velocity::reference::fit_gene(x, y, 2, 1, {}, scratch);
    auto s = anndatapp::Array::empty<double>({4, 1});
    auto u = anndatapp::Array::empty<double>({4, 1});
    auto w = anndatapp::Array::zeros<double>({4, 4});
    auto* sp = s.edit().dense<double>().data();
    auto* up = u.edit().dense<double>().data();
    auto* wp = w.edit().dense<double>().data();
    const double observations[] = {2, 2, 5, 7};
    for (int i = 0; i < 4; ++i) { sp[i] = i + 1; up[i] = observations[i]; wp[5*i] = 1; }
    velocity::PreparedInput input({s.view(), u.view(), w.view()});
    velocity::StochasticWorkspace workspace(input);
    auto output = anndatapp::Array::empty<double>({4, 1});
    velocity::StochasticOptions options;
    options.initialization.trimming = velocity::Trimming::none;
    options.initialization.min_r2 = -10;
    auto fit = velocity::reference::stochastic_into(input, workspace, output.edit(), options);

    std::vector<std::uint16_t> half_s(4), half_u(4);
    for (std::size_t i = 0; i < half_s.size(); ++i) {
        half_s[i] = cellerator::numeric::round_f16(float(i + 1));
        half_u[i] = cellerator::numeric::round_f16(float(i + 2) * 2);
    }
    auto hs = velocity::NumericView::borrow_dense_f16(half_s.data(), {4, 1}, {1, 1});
    auto hu = velocity::NumericView::borrow_dense_f16(half_u.data(), {4, 1}, {1, 1});
    velocity::PreparedInput mixed_input({hs, hu, w.view()});
    velocity::Workspace automatic_mixed(mixed_input);
    velocity::RunOptions half_run;
    half_run.compute_precision = velocity::ComputePrecision::float32;
    half_run.output_dtype = velocity::StorageType::float16;
    velocity::Workspace half_workspace(mixed_input, half_run);
    std::vector<std::uint16_t> half_output(4);
    auto half_edit = velocity::NumericEdit::borrow_dense_f16(half_output.data(), {4, 1}, {1, 1});
    auto half_fit = velocity::reference::steady_state_into(
        mixed_input, half_workspace, half_edit, options.initialization);
    auto half_gamma = half_fit.gamma.view().dense<float>();
    auto half_offset = half_fit.offset.view().dense<float>();
    bool half_ok = automatic_mixed.compute_precision() == velocity::ComputePrecision::float64 &&
        half_fit.gamma.view().dtype() == anndatapp::DType::float32 &&
        half_edit.view().dtype() == velocity::StorageType::float16;
    for (std::size_t i = 0; i < half_output.size(); ++i) {
        const float expected = cellerator::numeric::load_f16(half_u[i]) -
            half_gamma.data()[0] * cellerator::numeric::load_f16(half_s[i]) -
            half_offset.data()[0];
        half_ok = half_ok && half_output[i] == cellerator::numeric::round_f16(expected);
    }

    return a.extent(0) == 2 && f.gamma == 2 && fit.gamma.extent(0) == 1 &&
        fit.status.view().dense<unsigned char>().data()[0] ==
            static_cast<unsigned char>(velocity::StochasticStatus::ok) && half_ok ? 0 : 1;
}
