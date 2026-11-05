#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

import velocity.precision;

// Bounded independent fixture driver: four FP32 moment columns arrive on
// stdin, and the actual compiled reference fit and advisory report are used.
int main() {
    std::size_t n = 0;
    if (!(std::cin >> n) || n > 1000000) return 2;
    std::vector<float> original_s(n), original_u(n), half_s(n), half_u(n), scratch(n);
    for (std::size_t i = 0; i < n; ++i)
        if (!(std::cin >> original_s[i] >> original_u[i] >> half_s[i] >> half_u[i])) return 2;
    velocity::FitOptions options;
    options.fit_intercept = true;
    auto original = velocity::reference::fit_gene(original_s.data(), original_u.data(), n, 1, options, scratch.data());
    auto half = velocity::reference::fit_gene(half_s.data(), half_u.data(), n, 1, options, scratch.data());
    auto report = velocity::precision::sensitivity_gene(half_s.data(), half_u.data(), n, 1,
        half, options, scratch.data());
    std::cout << std::setprecision(9)
        << "{\"original_r2\":" << original.r2 << ",\"half_r2\":" << half.r2
        << ",\"original_selected\":" << int(original.selected) << ",\"half_selected\":" << int(half.selected)
        << ",\"original_status\":" << int(original.status) << ",\"half_status\":" << int(half.status)
        << ",\"report_state\":" << int(report.state) << ",\"prediction_change\":" << report.prediction_change
        << ",\"r2_change\":" << report.r2_change << ",\"delta\":" << report.delta << "}\n";
}
