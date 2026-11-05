#pragma once

// Include after importing velocity; this is an internal module-unit header.
namespace velocity::detail {

GeneFit fit_gene(const double* spliced, const double* unspliced,
                 std::size_t rows, std::ptrdiff_t stride,
                 const FitOptions&, double* score_scratch,
                 const std::uint8_t* fit_mask, bool gene_enabled);

GeneFit32 fit_gene(const float* spliced, const float* unspliced,
                 std::size_t rows, std::ptrdiff_t stride,
                 const FitOptions&, float* score_scratch,
                 const std::uint8_t* fit_mask, bool gene_enabled);

}
