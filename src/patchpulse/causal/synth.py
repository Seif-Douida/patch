"""Synthetic control (design D6, spec §9.2 M2): a blend of comparison games as the patched
game's twin.

- **Weights:** non-negative, summing to 1, fitted on the 28 days before the patch, each series
  centred on its own pre-period mean (demeaned synthetic control, Ferman & Pinto 2021): positive
  shares differ a lot between games, and a plain blend can't reach a game rated above all of them.
- **Fit check:** the twin's pre-period error, corrected for the degrees of freedom the weights
  used (with many comparison games the fit also tracks noise, so the raw error understates the
  misfit), must be at most 1.5 times the error sampling alone would give (the treated game's and
  the weighted comparison games' binomial or Poisson noise).
- **Placebo-in-space:** each comparison game fitted as if it had been patched; the p-value ranks the
  treated game's post/pre error ratio among them (Abadie, Diamond & Hainmueller 2010).
- **90% interval:** conformal inference (Chernozhukov, Wuthrich & Zhu 2021): a hypothesised effect
  is kept unless the post days' mean residual stands out among every cyclic shift of the residuals.
  The statistic is the absolute mean residual, aimed at a level shift; the mean absolute residual
  gave intervals three times as wide on simulated panels (median 14.5 points against 5.0, both
  covering ~0.87), and ran off the grid on 1 panel in 10.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from patchpulse.causal.its import Kind, NoEstimate
from patchpulse.causal.series import DailySeries, FloatArray, too_thin

PRE_DAYS = 28
MIN_FIT_DAYS = 21
FIT_FACTOR = 1.5
LEVEL = 0.90
# Conformal grids: hypothesised effects in points (shares) or percent (volume).
SHARE_STEP, SHARE_SPAN = 0.25, 25.0
VOLUME_STEP, VOLUME_SPAN = 2.0, 60.0
WIDER = "interval wider than the grid"
EMPTY = "interval empty: no effect size passed the conformal test"


def fit_weights(treated_pre: FloatArray, donors_pre: FloatArray) -> FloatArray:
    """Non-negative weights summing to 1 that make the comparison games' demeaned blend track the
    treated game's demeaned series as closely as possible (least squares, SLSQP from equal
    weights: deterministic)."""
    target = treated_pre - treated_pre.mean()
    pool = donors_pre - donors_pre.mean(axis=0)
    count = pool.shape[1]
    if count == 1:
        return np.ones(1)
    scale = 1e4  # SLSQP's tolerance is absolute; daily shares give objectives near 1e-3

    def objective(w: FloatArray) -> float:
        gap = pool @ w - target
        return float(scale * (gap @ gap))

    def gradient(w: FloatArray) -> FloatArray:
        result: FloatArray = 2 * scale * pool.T @ (pool @ w - target)
        return result

    result = minimize(
        objective,
        np.full(count, 1.0 / count),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints=LinearConstraint(np.ones((1, count)), 1.0, 1.0),
        tol=1e-12,
        options={"maxiter": 500},
    )
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, None)
    normalised: FloatArray = weights / weights.sum()
    return normalised


@dataclass(frozen=True)
class SynthFit:
    effect: float  # points (share) or percent (volume)
    post_gaps: FloatArray  # in model units: share fractions or log volume
    pre_rmse: float
    post_rmse: float
    noise_se: float
    fit_ok: bool
    weights: dict[int, float]


@dataclass(frozen=True)
class _Panel:
    treated: FloatArray  # rows: the days with data in every series
    treated_var: FloatArray
    donors: FloatArray  # rows x games
    donors_var: FloatArray
    post: np.ndarray  # bool per row
    appids: list[int]


def _values(
    series: DailySeries, first: date, end: date, kind: Kind
) -> tuple[FloatArray, FloatArray]:
    part = series.require_span(first, end)
    totals = part.totals.astype(np.float64)
    if kind == "share":
        y = part.share()
        with np.errstate(divide="ignore", invalid="ignore"):
            var = y * (1 - y) / totals
    else:
        y = np.where(part.observed, np.log1p(totals), np.nan)
        var = 1.0 / (1.0 + totals)
    return y, np.where(np.isfinite(y), var, np.nan)


def _panel(
    treated: DailySeries,
    donors: Mapping[int, DailySeries],
    first: date,
    end: date,
    start: date,
    kind: Kind,
) -> _Panel:
    y, y_var = _values(treated, first, end, kind)
    appids = sorted(appid for appid, s in donors.items() if s.span(first, end) is not None)
    columns = [_values(donors[appid], first, end, kind) for appid in appids]
    pool = np.column_stack([c[0] for c in columns]) if columns else np.empty((len(y), 0))
    pool_var = np.column_stack([c[1] for c in columns]) if columns else np.empty((len(y), 0))
    rows = np.isfinite(y) & np.isfinite(pool).all(axis=1)
    days = np.array([(first + timedelta(days=i) - start).days for i in range(len(y))])
    return _Panel(y[rows], y_var[rows], pool[rows], pool_var[rows], days[rows] >= 0, appids)


def _gaps(
    panel: _Panel, weights: FloatArray, treated: FloatArray, fit_rows: np.ndarray
) -> FloatArray:
    centre_t = treated[fit_rows].mean()
    centre_d = panel.donors[fit_rows].mean(axis=0)
    gaps: FloatArray = (treated - centre_t) - (panel.donors - centre_d) @ weights
    return gaps


def _to_output(delta: float, kind: Kind) -> float:
    return 100 * delta if kind == "share" else 100 * float(np.expm1(delta))


def _to_model(effect: float, kind: Kind) -> float:
    return effect / 100 if kind == "share" else float(np.log1p(effect / 100))


def synth_effect(
    treated: DailySeries,
    donors: Mapping[int, DailySeries],
    start: date,
    horizon: int,
    kind: Kind,
    *,
    pre_days: int = PRE_DAYS,
    fit_factor: float = FIT_FACTOR,
) -> SynthFit | NoEstimate:
    """The patch's effect over `horizon` days from `start`: the mean gap between the game and its
    twin after the patch."""
    end = start + timedelta(days=horizon - 1)
    first = start - timedelta(days=pre_days)
    if end > treated.last_day:
        return NoEstimate("window not complete")
    if first < treated.first_day:
        return NoEstimate("pre-period under 28 days")
    if too_thin(treated, first, end):
        return NoEstimate("too few reviews (median under 10 a day)")
    panel = _panel(treated, donors, first, end, start, kind)
    if not panel.appids:
        return NoEstimate("no comparison games")
    pre = ~panel.post
    if pre.sum() < MIN_FIT_DAYS or not panel.post.any():
        return NoEstimate("too few days with data in the fit window")
    weights = fit_weights(panel.treated[pre], panel.donors[pre])
    gaps = _gaps(panel, weights, panel.treated, pre)
    pre_rmse = float(np.sqrt(np.mean(gaps[pre] ** 2)))
    post_rmse = float(np.sqrt(np.mean(gaps[panel.post] ** 2)))
    variance = panel.treated_var + panel.donors_var @ (weights**2)
    noise_se = float(np.sqrt(np.mean(variance[pre])))
    fit_days = int(pre.sum())
    used = int((weights > 1e-4).sum()) + 1  # the active weights and the centring
    fit_rmse = pre_rmse * float(np.sqrt(fit_days / max(1, fit_days - used)))
    return SynthFit(
        effect=_to_output(float(gaps[panel.post].mean()), kind),
        post_gaps=gaps[panel.post],
        pre_rmse=pre_rmse,
        post_rmse=post_rmse,
        noise_se=noise_se,
        fit_ok=fit_rmse <= fit_factor * noise_se,
        weights={appid: float(w) for appid, w in zip(panel.appids, weights, strict=True)},
    )


def _ratio(fit: SynthFit) -> float:
    return fit.post_rmse / fit.pre_rmse if fit.pre_rmse > 0 else float("inf")


def placebo_p_value(
    treated: DailySeries,
    donors: Mapping[int, DailySeries],
    start: date,
    horizon: int,
    kind: Kind,
    treated_fit: SynthFit,
) -> float:
    """Placebo-in-space: each comparison game fitted against the others as if it had been patched;
    the share of games (the treated one included) whose post/pre error ratio is at least the
    treated game's."""
    ratios = []
    for appid in sorted(donors):
        others = {k: v for k, v in donors.items() if k != appid}
        if not others:
            continue
        fit = synth_effect(donors[appid], others, start, horizon, kind)
        if isinstance(fit, SynthFit):
            ratios.append(_ratio(fit))
    observed = _ratio(treated_fit)
    return (1 + sum(r >= observed for r in ratios)) / (1 + len(ratios))


def conformal_interval(
    treated: DailySeries,
    donors: Mapping[int, DailySeries],
    start: date,
    horizon: int,
    kind: Kind,
    *,
    level: float = LEVEL,
    step: float | None = None,
    span: float | None = None,
) -> tuple[float, float] | NoEstimate:
    """The effects (points or percent) not rejected at 1 - `level`: from the point estimate, out
    along the grid while the cyclic-shift p-value stays above 1 - `level`; each end is the first
    rejected grid point, so the range holds the estimate and every kept effect. No range when the
    estimate itself is rejected, or the kept effects reach the grid's edge."""
    step = step if step is not None else (SHARE_STEP if kind == "share" else VOLUME_STEP)
    span = span if span is not None else (SHARE_SPAN if kind == "share" else VOLUME_SPAN)
    point = synth_effect(treated, donors, start, horizon, kind)
    if not isinstance(point, SynthFit):
        return point
    if abs(point.effect) > span:
        return NoEstimate(WIDER)
    panel = _panel(
        treated,
        donors,
        start - timedelta(days=PRE_DAYS),
        start + timedelta(days=horizon - 1),
        start,
        kind,
    )
    every = np.ones(len(panel.treated), dtype=bool)
    shifts = [np.roll(np.arange(len(panel.treated)), -k) for k in range(len(panel.treated))]

    def kept(effect: float) -> bool:
        target = panel.treated - _to_model(effect, kind) * panel.post
        residuals = _gaps(panel, fit_weights(target, panel.donors), target, every)
        observed = abs(residuals[panel.post].mean())
        p_value = np.mean(
            [abs(residuals[order][panel.post].mean()) >= observed for order in shifts]
        )
        return bool(p_value > 1 - level)

    if not kept(point.effect):
        return NoEstimate(EMPTY)
    lo = math.floor(point.effect / step) * step
    while kept(lo):
        lo -= step
        if lo < -span - 1e-9:
            return NoEstimate(WIDER)
    hi = math.ceil(point.effect / step) * step
    while kept(hi):
        hi += step
        if hi > span + 1e-9:
            return NoEstimate(WIDER)
    return lo, hi
