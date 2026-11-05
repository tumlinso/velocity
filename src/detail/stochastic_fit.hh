#pragma once

// Include <cstddef> and <cstdint> and import velocity before this header.
namespace velocity::detail {

// Each pointer addresses one gene; successive rows are stride doubles apart.
// The optional mask has one byte per row. scratch holds at least rows doubles.
StochasticScales stochastic_scales(const double* ms, const double* mu,
    const double* mss, const double* mus, std::size_t rows, std::ptrdiff_t stride,
    const GeneFit&, const StochasticOptions&, const std::uint8_t* mask);

StochasticGeneFit stochastic_fit_gene(const double* ms, const double* mu,
    const double* mss, const double* mus, std::size_t rows, std::ptrdiff_t stride,
    const GeneFit&, StochasticScales, const StochasticOptions&, double* scratch,
    const std::uint8_t* mask);

StochasticScales32 stochastic_scales(const float* ms, const float* mu,
    const float* mss, const float* mus, std::size_t rows, std::ptrdiff_t stride,
    const GeneFit32&, const StochasticOptions&, const std::uint8_t* mask);

StochasticGeneFit32 stochastic_fit_gene(const float* ms, const float* mu,
    const float* mss, const float* mus, std::size_t rows, std::ptrdiff_t stride,
    const GeneFit32&, StochasticScales32, const StochasticOptions&, float* scratch,
    const std::uint8_t* mask);

}
