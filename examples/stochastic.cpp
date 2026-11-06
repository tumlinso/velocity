#include <cstdio>
import velocitypp;
int main() {
    auto s = anndatapp::Array::empty<double>({4, 1});
    auto u = anndatapp::Array::empty<double>({4, 1});
    auto w = anndatapp::Array::zeros<double>({4, 4});
    for (int i = 0; i < 4; ++i) {
        s.edit().dense<double>().data()[i] = i + 1;
        u.edit().dense<double>().data()[i] = 2*(i + 1) + (i % 2 ? .3 : -.2);
        w.edit().dense<double>().data()[5*i] = 1;
    }
    velocitypp::PreparedInput input({s.view(), u.view(), w.view()});
    velocitypp::StochasticWorkspace workspace(input);
    auto out = anndatapp::Array::empty<double>({4, 1});
    auto result = velocitypp::reference::stochastic_into(input, workspace, out.edit());
    std::printf("stochastic gamma %.6g, scratch %zu bytes\n", result.gamma.view().dense<double>().data()[0], workspace.scratch_bytes());
}
