#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>
import velocitypp;

void check(bool condition) { if (!condition) throw std::runtime_error("customization check failed"); }

struct Opaque { std::size_t first, count; };
struct OpaqueStages {
    int blocks = 0, values = 0;
    Opaque moments(std::size_t first, std::size_t count) { return {first, count}; }
    Opaque fit(Opaque token) { ++blocks; return token; }
    void emit(Opaque token, Opaque fit) { check(token.first == fit.first); values += int(token.count); }
    int finish() { return values; }
};
struct Fused {
    int blocks = 0, values = 0;
    void process_block(std::size_t, std::size_t count) { ++blocks; values += int(count); }
    int finish() { return values; }
};
struct CustomFit {
    velocitypp::ReferenceStages reference;
    double gamma[1]{3}, offset[1]{0}, r2[1]{.9};
    bool selected[1]{true};
    std::uint8_t status[1]{std::uint8_t(velocitypp::FitStatus::ok)};
    velocitypp::MomentBlock moments(std::size_t first, std::size_t count) { return reference.moments(first, count); }
    velocitypp::FitBlock fit(velocitypp::MomentBlock m) {
        return {m.first, m.count, gamma, offset, r2, selected, status};
    }
    void emit(velocitypp::MomentBlock m, velocitypp::FitBlock f) { reference.emit(m, f); }
    velocitypp::FitResult finish() { return reference.finish(); }
};
struct CustomMoments {
    velocitypp::ReferenceStages reference;
    double spliced[3]{1, 2, 3}, unspliced[3]{6, 12, 18};
    velocitypp::MomentBlock moments(std::size_t first, std::size_t count) {
        return {first, count, 3, 1, spliced, unspliced};
    }
    velocitypp::FitBlock fit(velocitypp::MomentBlock m) { return reference.fit(m); }
    void emit(velocitypp::MomentBlock m, velocitypp::FitBlock f) { reference.emit(m, f); }
    velocitypp::FitResult finish() { return reference.finish(); }
};

void collect(void* context, const velocitypp::OutputBlock& block) {
    auto* values = static_cast<double*>(context);
    auto view = block.velocity.dense<double>();
    for (std::size_t i = 0; i < view.extent(0); ++i)
        values[i] = view.data()[std::ptrdiff_t(i)*view.stride(0)];
}

int main() {
    try {
        OpaqueStages opaque;
        check(velocitypp::steady_state_with(opaque, 5, 2) == 5 && opaque.blocks == 3);
        Fused fused;
        check(velocitypp::steady_state_with(fused, 5, 2) == 5 && fused.blocks == 3);
        auto s = anndatapp::Array::empty<double>({3, 1});
        auto u = anndatapp::Array::empty<double>({3, 1});
        auto w = anndatapp::Array::zeros<double>({3, 3});
        auto* sp = s.edit().dense<double>().data();
        auto* up = u.edit().dense<double>().data();
        auto* wp = w.edit().dense<double>().data();
        for (int i = 0; i < 3; ++i) { sp[i] = i + 1; up[i] = 2*sp[i]; wp[4*i] = 1; }
        velocitypp::PreparedInput input({s.view(), u.view(), w.view()});
        velocitypp::Workspace workspace(input);
        double custom_velocity[3]{};
        CustomFit custom{velocitypp::ReferenceStages(input, workspace, {}, {custom_velocity, collect})};
        auto replaced = velocitypp::steady_state_with(custom, input.n_genes(), workspace.block_genes());
        check(std::abs(replaced.gamma.view().dense<double>().data()[0] - 3) < 1e-12);
        for (int i = 0; i < 3; ++i) check(std::abs(custom_velocity[i] + sp[i]) < 1e-12);
        CustomMoments custom_moments{velocitypp::ReferenceStages(input, workspace, {}, {custom_velocity, collect})};
        auto other = velocitypp::steady_state_with(custom_moments, input.n_genes(), workspace.block_genes());
        check(std::abs(other.gamma.view().dense<double>().data()[0] - 6) < 1e-12);
        for (int i = 0; i < 3; ++i) check(std::abs(custom_velocity[i]) < 1e-12);
        auto v = anndatapp::Array::empty<double>({3, 1});
        auto baseline = velocitypp::reference::steady_state_into(input, workspace, v.edit());
        check(std::abs(baseline.gamma.view().dense<double>().data()[0] - 2) < 1e-12);
        auto* vp = v.view().dense<double>().data();
        for (int i = 0; i < 3; ++i) check(std::abs(vp[i]) < 1e-12);
        double x[4]{1e8, 1e8 + 1, 1e8 + 2, 1e8 + 3};
        double y[4]{2e8 + 1, 2e8 + 3, 2e8 + 5, 2e8 + 7}, scratch[4];
        velocitypp::FitOptions centered;
        centered.trimming = velocitypp::Trimming::none;
        centered.fit_intercept = true;
        auto stable = velocitypp::reference::fit_gene(x, y, 4, 1, centered, scratch);
        check(std::abs(stable.gamma - 2) < 1e-12 && std::abs(stable.offset - 1) < 1e-12);
        std::puts("custom stages, opaque handles, fusion, and explicit reference passed");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
