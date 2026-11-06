#pragma once

// Include <stddef.h> and <stdint.h> and import velocitypp before this header.
namespace velocitypp::detail {

// Each pointer addresses one gene; successive rows are stride elements apart.
// The optional mask has one byte per row. The caller validates all buffers,
// stride range, options, and nonaliasing. Fit scratch holds rows elements when
// trimming is enabled.
StochasticScales stochastic_scales(const double* ms, const double* mu,
    const double* mss, const double* mus, size_t rows, ptrdiff_t stride,
    const GeneFit&, const StochasticOptions&, const uint8_t* mask);

StochasticGeneFit stochastic_fit_gene(const double* ms, const double* mu,
    const double* mss, const double* mus, size_t rows, ptrdiff_t stride,
    const GeneFit&, StochasticScales, const StochasticOptions&, double* scratch,
    const uint8_t* mask);

StochasticScales32 stochastic_scales(const float* ms, const float* mu,
    const float* mss, const float* mus, size_t rows, ptrdiff_t stride,
    const GeneFit32&, const StochasticOptions&, const uint8_t* mask);

StochasticGeneFit32 stochastic_fit_gene(const float* ms, const float* mu,
    const float* mss, const float* mus, size_t rows, ptrdiff_t stride,
    const GeneFit32&, StochasticScales32, const StochasticOptions&, float* scratch,
    const uint8_t* mask);

}
