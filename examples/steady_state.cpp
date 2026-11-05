import velocity;

int main() {
    auto data = anndatapp::AnnData::empty(3, 1);
    auto s = anndatapp::Array::empty<float>({3, 1});
    auto u = anndatapp::Array::empty<float>({3, 1});
    auto w = anndatapp::Array::zeros<float>({3, 3});
    auto* sp = s.edit().dense<float>().data();
    auto* up = u.edit().dense<float>().data();
    auto* wp = w.edit().dense<float>().data();
    for (int i = 0; i < 3; ++i) { sp[i] = float(i + 1); up[i] = 2 * sp[i]; wp[4*i] = 1; }
    data.set_layer("spliced", static_cast<anndatapp::Array&&>(s));
    data.set_layer("unspliced", static_cast<anndatapp::Array&&>(u));
    data.set_obsp("connectivities", static_cast<anndatapp::Array&&>(w));
    velocity::PreparedInput input(velocity::from_anndata(data));
    velocity::Workspace workspace(input);
    auto v = anndatapp::Array::empty<float>({3, 1});
    auto fit = velocity::reference::steady_state_into(input, workspace, v.edit());
    return fit.gamma.view().dense<double>().data()[0] == 2 ? 0 : 1;
}
