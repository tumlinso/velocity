#pragma once

// Include after importing velocitypp; this is an internal module-unit header.
namespace velocitypp::detail {

// Caller validates options, pointers, stride range, scratch capacity, and
// nonaliasing before entering these raw numerical routines.
GeneFit fit_gene(const double* spliced, const double* unspliced,
                 size_t rows, ptrdiff_t stride,
                 const FitOptions&, double* score_scratch,
                 const uint8_t* fit_mask, bool gene_enabled);

GeneFit32 fit_gene(const float* spliced, const float* unspliced,
                 size_t rows, ptrdiff_t stride,
                 const FitOptions&, float* score_scratch,
                 const uint8_t* fit_mask, bool gene_enabled);

}
