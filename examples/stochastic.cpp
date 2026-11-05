#include <cstdio>
import velocity;
int main() {
    auto s = anndatapp::Array::empty<double>({4, 1});
    auto u = anndatapp::Array::empty<double>({4, 1});
    auto w = anndatapp::Array::zeros<double>({4, 4});
    for (int i = 0; i < 4; ++i) {
        s.edit().dense<double>().data()[i] = i + 1;
        u.edit().dense<double>().data()[i] = 2*(i + 1) + (i % 2 ? .3 : -.2);
        w.edit().dense<double>().data()[5*i] = 1;
    }
    velocity::PreparedInput input({s.view(), u.view(), w.view()});
    velocity::StochasticWorkspace workspace(input);
    auto out = anndatapp::Array::empty<double>({4, 1});
    auto result = velocity::reference::stochastic_into(input, workspace, out.edit());
    std::printf("stochastic gamma %.6g, scratch %zu bytes\n", result.gamma.view().dense<double>().data()[0], workspace.scratch_bytes());
}
