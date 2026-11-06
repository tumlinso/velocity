module;
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include "detail/sensitivity.hh"

module velocitypp.precision;
import velocitypp;

namespace velocitypp::precision {
namespace {

struct Sort {
    template<class T> void operator()(T* p, std::size_t n) const { std::sort(p, p+n); }
};

template<class T> velocitypp::detail::sensitivity::Options<T> numeric(SensitivityOptions o) {
    return {T(o.band), T(o.prediction_limit), T(o.r2_limit)};
}

template<class T> velocitypp::detail::scientific::Options<T> numeric(const FitOptions& o) {
    using namespace velocitypp::detail;
    return {scientific::Trimming(o.trimming), T(o.percentile_low), T(o.percentile_high),
            o.fit_intercept, o.positive_intercept, o.r2_adjusted,
            T(o.ratio_min), T(o.ratio_max), T(o.min_r2), T(o.min_ratio)};
}

template<class T> BasicSensitivityReport<T> report_gene(const T* x, const T* y,
    std::size_t rows, std::ptrdiff_t stride, const BasicGeneFit<T>& base,
    const FitOptions& fit, T* scratch, const std::uint8_t* mask, bool enabled,
    SensitivityOptions o) {
    if (!std::isfinite(o.band) || !std::isfinite(o.prediction_limit) || !std::isfinite(o.r2_limit) ||
        o.band < 0 || o.prediction_limit < 0 || o.r2_limit < 0)
        throw std::invalid_argument("invalid sensitivity options");
    T dummy = 0;
    reference::fit_gene(static_cast<const T*>(nullptr), static_cast<const T*>(nullptr), 0, 1, fit, &dummy);
    if (rows) {
        if (!x || !y || !stride) throw std::invalid_argument("invalid sensitivity inputs or stride");
        if (fit.trimming != Trimming::none && !scratch) throw std::invalid_argument("sensitivity requires score scratch");
        auto step = stride > 0 ? std::uintmax_t(stride) : std::uintmax_t(-(stride + 1)) + 1;
        if (rows > 1 && std::uintmax_t(rows - 1) > std::uintmax_t(std::numeric_limits<std::ptrdiff_t>::max())/step)
            throw std::invalid_argument("sensitivity stride overflows row offsets");
    }
    using namespace velocitypp::detail;
    auto result = sensitivity::report(x, y, rows, stride,
        scientific::Fit<T>{base.gamma, base.offset, base.r2, scientific::FitStatus(base.status), base.selected},
        numeric<T>(fit), scratch, mask, enabled, numeric<T>(o), Sort{});
    return {SensitivityState(result.state), result.low, result.high, result.delta,
            result.prediction_change, result.r2_change, FitStatus(result.expanded_status),
            FitStatus(result.contracted_status), result.status_change, result.selection_change};
}

bool higher_precision(const OriginalColumn& c, std::size_t n) {
    auto usable = [n](const anndatapp::ArrayView& a) {
        if (a.rank() != 2 || a.extent(0) != n || a.extent(1) != 1)
            throw std::invalid_argument("an original source column must have shape cells by one");
        return a.dtype() == anndatapp::DType::float32 || a.dtype() == anndatapp::DType::float64;
    };
    return usable(c.spliced) && usable(c.unspliced);
}

template<class T> std::pair<std::unique_ptr<T[]>, std::size_t> allocate_score(
    std::size_t rows, std::size_t stage_bytes, std::size_t budget, std::size_t staging) {
    if (rows > std::numeric_limits<std::size_t>::max()/sizeof(T))
        throw std::invalid_argument("sensitivity scratch size overflows");
    auto bytes = rows*sizeof(T);
    if (stage_bytes > budget || staging > budget - stage_bytes || bytes > budget - stage_bytes - staging)
        throw std::invalid_argument("sensitivity scratch exceeds the numerical workspace budget");
    return {rows ? std::unique_ptr<T[]>(new T[rows]) : nullptr, bytes};
}

}

template<class T> BasicSensitivityWorkspace<T>::BasicSensitivityWorkspace(const PreparedInput& p,
    const Workspace& w, const BasicReferenceStages<T>& stages, std::size_t budget, std::size_t staging) {
    if (!stages.bound_to(p, w)) throw std::invalid_argument("sensitivity stage belongs to another input or workspace");
    auto allocation = allocate_score<T>(p.n_cells(), w.scratch_bytes(), budget, staging);
    data_ = std::move(allocation.first); bytes_ = allocation.second;
}
template<class T> BasicSensitivityWorkspace<T>::BasicSensitivityWorkspace(const PreparedInput& p,
    const StochasticWorkspace& w, const BasicStochasticReferenceStages<T>& stages,
    std::size_t budget, std::size_t staging) {
    if (!stages.bound_to(p, w)) throw std::invalid_argument("sensitivity stage belongs to another input or workspace");
    auto allocation = allocate_score<T>(p.n_cells(), w.scratch_bytes(), budget, staging);
    data_ = std::move(allocation.first); bytes_ = allocation.second;
}
template<class T> BasicSensitivityWorkspace<T>::~BasicSensitivityWorkspace() = default;
template<class T> BasicSensitivityWorkspace<T>::BasicSensitivityWorkspace(BasicSensitivityWorkspace&&) noexcept = default;
template<class T> BasicSensitivityWorkspace<T>& BasicSensitivityWorkspace<T>::operator=(BasicSensitivityWorkspace&&) noexcept = default;
template<class T> T* BasicSensitivityWorkspace<T>::data() noexcept { return data_.get(); }
template<class T> std::size_t BasicSensitivityWorkspace<T>::scratch_bytes() const noexcept { return bytes_; }

SensitivityReport sensitivity_gene(const double* x, const double* y, std::size_t rows,
    std::ptrdiff_t stride, const GeneFit& base, const FitOptions& fit, double* scratch,
    const std::uint8_t* mask, bool enabled, SensitivityOptions o) {
    return report_gene(x, y, rows, stride, base, fit, scratch, mask, enabled, o);
}
SensitivityReport32 sensitivity_gene(const float* x, const float* y, std::size_t rows,
    std::ptrdiff_t stride, const GeneFit32& base, const FitOptions& fit, float* scratch,
    const std::uint8_t* mask, bool enabled, SensitivityOptions o) {
    return report_gene(x, y, rows, stride, base, fit, scratch, mask, enabled, o);
}

OriginalArraySource::OriginalArraySource(anndatapp::ArrayView s, anndatapp::ArrayView u)
    : spliced_(std::move(s)), unspliced_(std::move(u)) {
    spliced_.validate(); unspliced_.validate();
    if (spliced_.rank() != 2 || unspliced_.rank() != 2 ||
        spliced_.layout() != anndatapp::Layout::dense || unspliced_.layout() != anndatapp::Layout::dense ||
        spliced_.extent(0) != unspliced_.extent(0) || spliced_.extent(1) != unspliced_.extent(1))
        throw std::invalid_argument("original source arrays must be matching dense matrices");
    auto floating = [](anndatapp::DType t) {
        return t == anndatapp::DType::float32 || t == anndatapp::DType::float64;
    };
    if (!floating(spliced_.dtype()) || !floating(unspliced_.dtype()))
        throw std::invalid_argument("original source arrays must use floating storage");
}
bool OriginalArraySource::acquire(void* context, std::size_t gene, OriginalColumn& c) {
    auto& source = *static_cast<OriginalArraySource*>(context);
    if (gene >= source.spliced_.extent(1)) return false;
    auto column = anndatapp::Axis::stride(gene, 1);
    c.spliced = source.spliced_.select({anndatapp::Axis::all(), column});
    c.unspliced = source.unspliced_.select({anndatapp::Axis::all(), column});
    return true;
}
OriginalSource OriginalArraySource::source() noexcept { return {this, &OriginalArraySource::acquire, 0}; }

namespace detail {
template<class T> struct SelectionState {
    const PreparedInput* input;
    BasicReferenceStages<T>* steady = nullptr;
    BasicStochasticReferenceStages<T>* stochastic = nullptr;
    OriginalSource source;
    PromotionOptions options;
    std::vector<std::uint8_t> status;
    std::size_t promoted = 0;

    SelectionState(const PreparedInput& p, std::size_t scratch, OriginalSource s, PromotionOptions o)
        : input(&p), source(s), options(o) {
        if (o.source_gene_mask && o.source_gene_count != p.source_genes())
            throw std::invalid_argument("promotion mask must cover the source gene axis");
        bool requested = false;
        if (o.source_gene_mask && s.acquire)
            for (std::size_t i = 0; i < p.n_genes() && !requested; ++i)
                requested = o.source_gene_mask[p.gene_index(i)] != 0;
        if (scratch > o.workspace_bytes || (requested && s.staging_bytes > o.workspace_bytes - scratch))
            throw std::invalid_argument("promotion staging exceeds the numerical workspace budget");
        if (o.source_gene_mask) status.resize(p.n_genes(), std::uint8_t(PromotionStatus::not_requested));
    }

    bool fetch(std::size_t position, OriginalColumn& column) const {
        auto gene = input->gene_index(position);
        return source.acquire && source.acquire(source.context, gene, column) && higher_precision(column, input->n_cells());
    }

    template<class Block, class Stages> void initialize(Block& b, Stages& stages) {
        if (!options.source_gene_mask) return;
        for (std::size_t j = 0; j < b.count; ++j) {
            auto position = b.first + j;
            if (!options.source_gene_mask[input->gene_index(position)]) continue;
            OriginalColumn c;
            if (!fetch(position, c)) { status[position] = std::uint8_t(PromotionStatus::unavailable); continue; }
            stages.replace_moments(b, j, 1, c.spliced, c.unspliced);
            status[position] = std::uint8_t(PromotionStatus::promoted); ++promoted;
        }
    }

    template<class Block> void reapply(Block& b, BasicStochasticReferenceStages<T>& stages) {
        if (status.empty()) return;
        for (std::size_t j = 0; j < b.count; ++j) {
            auto position = b.first + j;
            if (status[position] != std::uint8_t(PromotionStatus::promoted)) continue;
            OriginalColumn c;
            if (!fetch(position, c))
                throw std::runtime_error("promoted original source became unavailable between passes");
            stages.replace_moments(b, j, 1, c.spliced, c.unspliced);
        }
    }
};
}

template<class T> BasicSelectiveStages<T>::BasicSelectiveStages(const PreparedInput& p, Workspace& w,
    BasicReferenceStages<T>& stages, OriginalSource source, PromotionOptions options) {
    if (!stages.bound_to(p, w)) throw std::invalid_argument("promotion stage belongs to another input or workspace");
    state_ = std::make_unique<detail::SelectionState<T>>(p, w.scratch_bytes(), source, options);
    state_->steady = &stages;
}
template<class T> BasicSelectiveStages<T>::~BasicSelectiveStages() = default;
template<class T> BasicSelectiveStages<T>::BasicSelectiveStages(BasicSelectiveStages&&) noexcept = default;
template<class T> BasicSelectiveStages<T>& BasicSelectiveStages<T>::operator=(BasicSelectiveStages&&) noexcept = default;
template<class T> BasicMomentBlock<T> BasicSelectiveStages<T>::moments(std::size_t first, std::size_t count) {
    auto b = state_->steady->moments(first, count); state_->initialize(b, *state_->steady); return b;
}
template<class T> BasicFitBlock<T> BasicSelectiveStages<T>::fit(BasicMomentBlock<T> b) { return state_->steady->fit(b); }
template<class T> void BasicSelectiveStages<T>::emit(BasicMomentBlock<T> m, BasicFitBlock<T> f) { state_->steady->emit(m, f); }
template<class T> FitResult BasicSelectiveStages<T>::finish() { return state_->steady->finish(); }
template<class T> const std::uint8_t* BasicSelectiveStages<T>::promotion_status() const noexcept { return state_->status.empty() ? nullptr : state_->status.data(); }
template<class T> std::size_t BasicSelectiveStages<T>::promotion_count() const noexcept { return state_->promoted; }

template<class T> BasicSelectiveStochasticStages<T>::BasicSelectiveStochasticStages(const PreparedInput& p,
    StochasticWorkspace& w, BasicStochasticReferenceStages<T>& stages, OriginalSource source, PromotionOptions options) {
    if (!stages.bound_to(p, w)) throw std::invalid_argument("promotion stage belongs to another input or workspace");
    state_ = std::make_unique<detail::SelectionState<T>>(p, w.scratch_bytes(), source, options);
    state_->stochastic = &stages;
}
template<class T> BasicSelectiveStochasticStages<T>::~BasicSelectiveStochasticStages() = default;
template<class T> BasicSelectiveStochasticStages<T>::BasicSelectiveStochasticStages(BasicSelectiveStochasticStages&&) noexcept = default;
template<class T> BasicSelectiveStochasticStages<T>& BasicSelectiveStochasticStages<T>::operator=(BasicSelectiveStochasticStages&&) noexcept = default;
template<class T> BasicMomentBlock<T> BasicSelectiveStochasticStages<T>::initial_moments(std::size_t first, std::size_t count) {
    auto b = state_->stochastic->initial_moments(first, count); state_->initialize(b, *state_->stochastic); return b;
}
template<class T> BasicFitBlock<T> BasicSelectiveStochasticStages<T>::initial_fit(BasicMomentBlock<T> b) { return state_->stochastic->initial_fit(b); }
template<class T> void BasicSelectiveStochasticStages<T>::accept_initialization(BasicMomentBlock<T> m, BasicFitBlock<T> f) { state_->stochastic->accept_initialization(m, f); }
template<class T> void BasicSelectiveStochasticStages<T>::resolve_selection() { state_->stochastic->resolve_selection(); }
template<class T> BasicStochasticMomentBlock<T> BasicSelectiveStochasticStages<T>::moments(std::size_t first, std::size_t count) {
    auto b = state_->stochastic->moments(first, count); state_->reapply(b, *state_->stochastic); return b;
}
template<class T> BasicStochasticScaleBlock<T> BasicSelectiveStochasticStages<T>::scales(BasicStochasticMomentBlock<T> b) { return state_->stochastic->scales(b); }
template<class T> BasicStochasticFitBlock<T> BasicSelectiveStochasticStages<T>::fit(BasicStochasticMomentBlock<T> m, BasicStochasticScaleBlock<T> s) { return state_->stochastic->fit(m, s); }
template<class T> void BasicSelectiveStochasticStages<T>::emit(BasicStochasticMomentBlock<T> m, BasicStochasticFitBlock<T> f) { state_->stochastic->emit(m, f); }
template<class T> StochasticFitResult BasicSelectiveStochasticStages<T>::finish() { return state_->stochastic->finish(); }
template<class T> const std::uint8_t* BasicSelectiveStochasticStages<T>::promotion_status() const noexcept { return state_->status.empty() ? nullptr : state_->status.data(); }
template<class T> std::size_t BasicSelectiveStochasticStages<T>::promotion_count() const noexcept { return state_->promoted; }

template class BasicSensitivityWorkspace<float>;
template class BasicSensitivityWorkspace<double>;
template class BasicSelectiveStages<float>;
template class BasicSelectiveStages<double>;
template class BasicSelectiveStochasticStages<float>;
template class BasicSelectiveStochasticStages<double>;
}
