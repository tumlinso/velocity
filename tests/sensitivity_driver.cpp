#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

import velocitypp.precision;

// Compare FP64 moments against the same input represented in FP32.  The
// advisory report describes the FP32 fit; it does not promise to detect all
// differences caused by narrowing the input.
int main() {
    std::size_t n = 0;
    if (!(std::cin >> n) || n > 1000000) return 2;
    std::vector<double> original_s(n), original_u(n), original_scratch(n);
    std::vector<float> narrow_s(n), narrow_u(n), scratch(n);
    for (std::size_t i = 0; i < n; ++i)
        if (!(std::cin >> original_s[i] >> original_u[i] >> narrow_s[i] >> narrow_u[i])) return 2;
    velocitypp::FitOptions options;
    options.fit_intercept = true;
    auto original = velocitypp::reference::fit_gene(original_s.data(), original_u.data(), n, 1, options, original_scratch.data());
    auto narrow = velocitypp::reference::fit_gene(narrow_s.data(), narrow_u.data(), n, 1, options, scratch.data());
    auto report = velocitypp::precision::sensitivity_gene(narrow_s.data(), narrow_u.data(), n, 1,
        narrow, options, scratch.data());
    std::cout << std::setprecision(17)
        << "{\"original_r2\":" << original.r2 << ",\"narrow_r2\":" << narrow.r2
        << ",\"original_selected\":" << int(original.selected) << ",\"narrow_selected\":" << int(narrow.selected)
        << ",\"original_status\":" << int(original.status) << ",\"narrow_status\":" << int(narrow.status)
        << ",\"report_state\":" << int(report.state) << ",\"prediction_change\":" << report.prediction_change
        << ",\"r2_change\":" << report.r2_change << ",\"delta\":" << report.delta << "}\n";
}
