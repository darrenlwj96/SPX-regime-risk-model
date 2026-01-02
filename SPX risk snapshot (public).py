"""
SPX Daily Risk Snapshot Engine
=============================

What this script provides:
- A daily (after-close) risk snapshot derived from SPX end-of-day data.
- Outputs:
  (1) A simple volatility regime label (CALM / STRESSED) + an interpretability score (heuristic).
  (2) A 1-day-ahead conditional volatility forecast (sigma_next) using Student-t GARCH-family models.
  (3) A conditional return-quantile ladder ("VaR ladder" colloquially; technically return quantiles).

What this script intentionally does NOT do:
- No option chains, no strike selection, no execution logic, no P&L, no backtest reporting.
- This is a risk diagnostics tool only.

Notes on tuning knobs:
- Configuration values are defaults chosen for stability and interpretability.
- They are not presented as "optimal" or "strategy parameters".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# OPTIONAL DEPENDENCIES (graceful fallback where possible)
# ---------------------------------------------------------
# - yfinance: only required if you use download mode.
# - arch: required for EGARCH/GJR-GARCH volatility models.
# - scipy: required for EVT/GPD tail enhancement; otherwise falls back to empirical quantiles.
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

try:
    from arch import arch_model
except Exception:  # pragma: no cover
    arch_model = None

try:
    from scipy.stats import genpareto
except Exception:  # pragma: no cover
    genpareto = None


# =========================================================
# 0) CONFIG (all knobs in one place)
# =========================================================
@dataclass(frozen=True)
class RiskConfig:
    # -----------------------
    # Data source
    # -----------------------
    ticker: str = "^GSPC"
    start_date: str = "2000-01-01"

    # -----------------------
    # Calendar assumptions
    # -----------------------
    trading_days: int = 252  # used to annualize/de-annualize RV

    # -----------------------
    # Realized-volatility feature windows (annualized)
    # -----------------------
    rv_short: int = 20
    rv_long: int = 63

    # -----------------------
    # Regime detection (adaptive thresholds)
    # -----------------------
    slope_win: int = 5          # coarse slope of RV ratio over this window
    persist_k: int = 3          # require K consecutive days above/below threshold
    q_hi: float = 0.75          # expanding quantile threshold for "high vol"
    q_lo: float = 0.25          # expanding quantile threshold for "low vol"

    # This is *labeling-only* (transition flag), NOT a probability of correctness.
    transition_conf_threshold: float = 0.70

    # -----------------------
    # Shock flag (diagnostic)
    # -----------------------
    shock_k: float = 3.0        # shock day if |ret| > k * daily_sigma (daily_sigma from shifted RV)

    # -----------------------
    # Vol model selection heuristic
    # -----------------------
    z_cut: float = -3.0         # count extreme negative standardized residuals z < z_cut

    # -----------------------
    # EVT hybrid (optional)
    # -----------------------
    use_evt: bool = True
    evt_cut: float = 0.01       # EVT used only beyond this tail probability (left/right)
    evt_q: Tuple[float, float] = (0.12, 0.88)  # generic thresholds in z-space
    min_exceed: int = 150       # minimum exceedances per tail to attempt GPD fit

    # -----------------------
    # Output probability levels (generic)
    # -----------------------
    # These are conditional return quantiles: Q_ret(p) ≈ sigma_next * Q_z(p)
    levels: Tuple[float, ...] = (
        0.0025, 0.005, 0.01, 0.02, 0.03, 0.05,
        0.10, 0.15, 0.20, 0.25, 0.30
    )

    # -----------------------
    # Optional output CSV path for daily log rows
    # -----------------------
    out_csv: Optional[str] = None


# =========================================================
# 1) SMALL HELPERS
# =========================================================
def _normalize_date(s: pd.Series) -> pd.Series:
    """Coerce to timezone-naive normalized dates (YYYY-MM-DD 00:00)."""
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()

def _safe_float(x, default=np.nan) -> float:
    """Convert any input to a finite float if possible; else default."""
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)

def _require(dep, name: str):
    """Raise a clean error if an optional dependency is missing."""
    if dep is None:
        raise RuntimeError(f"Missing optional dependency '{name}'. Install it or disable that feature.")

def _clip01(x: pd.Series) -> pd.Series:
    """Clip numeric values into [0, 1]."""
    return x.clip(lower=0.0, upper=1.0)


# =========================================================
# 2) DATA LOADING (EOD)
# =========================================================
# This is designed as an "after-close snapshot" utility.
# For research/backtests, replace this with a fixed-range data loader to avoid fetch drift.
def load_spx(cfg: RiskConfig) -> pd.DataFrame:
    _require(yf, "yfinance")

    # End date as "tomorrow" to include today's bar if available.
    end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.download(
        cfg.ticker,
        start=cfg.start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    # Some yfinance outputs come with a MultiIndex columns structure.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Keep only what we need for risk diagnostics.
    df = (
        raw[["High", "Low", "Close"]]
        .rename(columns={"High": "SPX_HIGH", "Low": "SPX_LOW", "Close": "SPX_CLOSE"})
        .dropna()
        .reset_index()
    )
    df["Date"] = _normalize_date(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


# =========================================================
# 3) FEATURES + REGIME (adaptive, leakage-safe)
# =========================================================
# Key principle:
# - Any feature that would be used as "today's risk posture" is built from information up to t-1
#   by shifting rolling RV measures by 1 day.
def add_features_and_regime(df_in: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    df = df_in.copy().sort_values("Date").reset_index(drop=True)

    # -----------------------
    # 3.1 Returns (log)
    # -----------------------
    df["ret"] = np.log(df["SPX_CLOSE"]).diff()

    # -----------------------
    # 3.2 Realized volatility proxies (annualized), shifted by 1 day
    # -----------------------
    # On date t, RV_s and RV_l reflect returns up to t-1 (leakage-safe).
    df["RV_s"] = df["ret"].rolling(cfg.rv_short).std().shift(1) * np.sqrt(cfg.trading_days)
    df["RV_l"] = df["ret"].rolling(cfg.rv_long).std().shift(1) * np.sqrt(cfg.trading_days)

    # Stress proxy: short vol relative to longer vol
    df["ratio"] = df["RV_s"] / df["RV_l"]

    # -----------------------
    # 3.3 Shock day flag (diagnostic only)
    # -----------------------
    # daily_sigma from shifted RV_l (convert annualized -> daily)
    daily_sigma = df["RV_l"] / np.sqrt(cfg.trading_days)
    df["shock"] = (df["ret"].abs() > cfg.shock_k * daily_sigma).astype("int8")

    # -----------------------
    # 3.4 Ratio slope (coarse motion indicator)
    # -----------------------
    df["ratio_slope"] = df["ratio"].diff(cfg.slope_win) / max(cfg.slope_win, 1)

    # -----------------------
    # 3.5 Adaptive thresholds (expanding quantiles)
    # -----------------------
    # Expanding window = thresholds use only past data.
    # min_periods prevents meaningless early-sample thresholds.
    df["th_hi"] = df["ratio"].expanding(min_periods=200).quantile(cfg.q_hi)
    df["th_lo"] = df["ratio"].expanding(min_periods=200).quantile(cfg.q_lo)

    # -----------------------
    # 3.6 Persistence logic (reduce noise)
    # -----------------------
    above = (df["ratio"] > df["th_hi"]).rolling(cfg.persist_k).sum() == cfg.persist_k
    below = (df["ratio"] < df["th_lo"]).rolling(cfg.persist_k).sum() == cfg.persist_k

    # Regime coding: 1=STRESSED, 0=CALM
    df["regime"] = np.where(above, 1, np.where(below, 0, np.nan))

    # Conservative default:
    # - forward-fill once a regime appears
    # - if still NaN (very early), default to STRESSED
    df["regime"] = df["regime"].ffill().fillna(1).astype("int8")
    df["regime_label"] = np.where(df["regime"] == 0, "CALM", "STRESSED")

    # -----------------------
    # 3.7 Interpretability score ("confidence", heuristic)
    # -----------------------
    # This is NOT a calibrated probability. It answers:
    # "Does the current regime label look strongly supported by persistence + slope + distance?"
    persistence_signal = (above | below).astype("int8")

    slope_strength = _clip01(df["ratio_slope"].abs() / (df["ratio"].abs() + 1e-12))

    dist = np.where(df["regime"] == 1, (df["ratio"] - df["th_hi"]), (df["th_lo"] - df["ratio"]))
    dist_conf = _clip01(np.abs(dist) / (np.abs(df["ratio"]) + 1e-12))

    df["conf"] = _clip01(0.5 * persistence_signal + 0.25 * slope_strength + 0.25 * dist_conf)

    return df


# =========================================================
# 4) CONDITIONAL VOL MODELS (EGARCH vs GJR-GARCH, Student-t)
# =========================================================
def _fit_vol_model(r_pct: pd.Series, model_name: str):
    _require(arch_model, "arch")

    m = model_name.upper()
    if m == "EGARCH":
        am = arch_model(r_pct, mean="Zero", vol="EGARCH", p=1, o=0, q=1, dist="t")
    elif m in ("GJR", "GJR-GARCH", "GJR_GARCH"):
        am = arch_model(r_pct, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t")
    else:
        raise ValueError("model_name must be 'EGARCH' or 'GJR'")
    return am.fit(disp="off")

def _forecast_next_sigma(vol_res) -> float:
    """
    Return next-day sigma in decimal (not %).
    arch_model forecasts variance of % returns; convert back to decimal.
    """
    f = vol_res.forecast(horizon=1, reindex=False)
    var1_pct2 = float(f.variance.values[-1, 0])
    sigma_pct = float(np.sqrt(max(var1_pct2, 0.0)))
    return sigma_pct / 100.0

def _tail_miss_count(z: pd.Series, z_cut: float) -> int:
    z = pd.Series(z).dropna()
    return int((z < z_cut).sum())

def fit_sigma_and_choose_model(df_feat: pd.DataFrame, cfg: RiskConfig):
    df = df_feat.copy().sort_values("Date").reset_index(drop=True)

    # arch expects returns on a typical % scale
    r_pct = df["ret"] * 100.0
    mask = r_pct.notna()
    r_pct = r_pct.loc[mask]

    if len(r_pct) < 500:
        raise ValueError(f"Too few return points for GARCH fit: {len(r_pct)}")

    # Fit both candidates
    res_E = _fit_vol_model(r_pct, "EGARCH")
    res_G = _fit_vol_model(r_pct, "GJR")

    # Align conditional vols and standardized residuals back to the same dates
    tmpE = df.loc[mask].copy()
    tmpE["sigma_is"] = res_E.conditional_volatility / 100.0
    tmpE["z_is"] = tmpE["ret"] / tmpE["sigma_is"]

    tmpG = df.loc[mask].copy()
    tmpG["sigma_is"] = res_G.conditional_volatility / 100.0
    tmpG["z_is"] = tmpG["ret"] / tmpG["sigma_is"]

    # Conservative heuristic:
    # If sigma is too small on bad days, z will have more extreme negative events.
    miss_E = _tail_miss_count(tmpE["z_is"], cfg.z_cut)
    miss_G = _tail_miss_count(tmpG["z_is"], cfg.z_cut)

    if miss_E < miss_G:
        chosen = "EGARCH"
        sf = tmpE
        vol_res = res_E
    else:
        chosen = "GJR-GARCH"
        sf = tmpG
        vol_res = res_G

    # One-step-ahead forecast
    sf = sf.copy()
    sf["sigma_next"] = _forecast_next_sigma(vol_res)

    # Reviewer-friendly diagnostic
    print("=== Conditional Volatility Model Selection (SPX) ===")
    print(f"Chosen model: {chosen}")
    print(f"Sample size N: {len(sf)}")
    print(f"Downside tail misses (z < {cfg.z_cut}): EGARCH={miss_E} | GJR={miss_G}")
    print("Note: heuristic selection for downside underestimation control.")

    return sf.reset_index(drop=True), chosen


# =========================================================
# 5) OPTIONAL EVT HYBRID (empirical center + GPD tails)
# =========================================================
def fit_evt_hybrid(
    df_in: pd.DataFrame,
    cfg: RiskConfig,
    z_col: str = "z_is",
) -> Dict:
    """
    Returns a dict containing:
      - q_z(p): callable quantile function for standardized residuals z
      - evt_ok: bool
      - reason: string for auditability
      - tail diagnostics if EVT was attempted

    Design:
      - Center quantiles: empirical
      - Extreme tails: EVT/GPD when stable and supported by enough exceedances
      - Fallback: empirical quantiles if anything is missing/unstable
    """
    z = pd.Series(df_in.get(z_col, pd.Series(dtype=float))).dropna().to_numpy()
    z = z[np.isfinite(z)]
    z_sorted = np.sort(z)

    # If EVT is disabled, scipy missing, or too few points -> empirical quantiles only
    if (not cfg.use_evt) or (genpareto is None) or (z_sorted.size < 500):
        def q_z(p: float) -> float:
            return float(np.quantile(z_sorted, float(p))) if z_sorted.size else np.nan
        return {"evt_ok": False, "reason": "evt_off_or_unavailable", "q_z": q_z}

    qL, qR = cfg.evt_q
    uL = float(np.quantile(z_sorted, qL))
    uR = float(np.quantile(z_sorted, qR))

    # Exceedances are defined as positive values beyond thresholds
    yL = (uL - z_sorted[z_sorted < uL])
    yR = (z_sorted[z_sorted > uR] - uR)

    if (len(yL) < cfg.min_exceed) or (len(yR) < cfg.min_exceed):
        def q_z(p: float) -> float:
            return float(np.quantile(z_sorted, float(p)))
        return {
            "evt_ok": False,
            "reason": f"too_few_exceedances(L={len(yL)},R={len(yR)})",
            "q_z": q_z,
            "uL": uL, "uR": uR, "nL": int(len(yL)), "nR": int(len(yR)),
            "qL": qL, "qR": qR,
        }

    # Fit GPD with basic sanity guards
    try:
        cL, _, sL = genpareto.fit(yL, floc=0)
        cR, _, sR = genpareto.fit(yR, floc=0)

        if (not np.isfinite(cL)) or (not np.isfinite(sL)) or sL <= 0:
            raise ValueError("Invalid left-tail GPD parameters.")
        if (not np.isfinite(cR)) or (not np.isfinite(sR)) or sR <= 0:
            raise ValueError("Invalid right-tail GPD parameters.")
    except Exception as e:
        def q_z(p: float) -> float:
            return float(np.quantile(z_sorted, float(p)))
        return {
            "evt_ok": False,
            "reason": f"evt_fit_failed({type(e).__name__})",
            "q_z": q_z,
            "uL": uL, "uR": uR, "nL": int(len(yL)), "nR": int(len(yR)),
            "qL": qL, "qR": qR,
        }

    # Quantile function for z
    def q_z(p: float) -> float:
        p = float(p)
        if p <= 0.0 or p >= 1.0:
            return np.nan

        # Left tail
        if p < cfg.evt_cut:
            t = np.clip(p / qL, 1e-12, 1 - 1e-12)
            y = genpareto.ppf(1.0 - t, cL, loc=0, scale=sL)
            return float(uL - y)

        # Right tail
        if p > 1.0 - cfg.evt_cut:
            t = np.clip((p - qR) / (1.0 - qR), 1e-12, 1 - 1e-12)
            y = genpareto.ppf(t, cR, loc=0, scale=sR)
            return float(uR + y)

        # Center
        return float(np.quantile(z_sorted, p))

    return {
        "evt_ok": True,
        "reason": "ok",
        "q_z": q_z,
        "uL": uL, "uR": uR,
        "nL": int(len(yL)), "nR": int(len(yR)),
        "qL": qL, "qR": qR,
    }


# =========================================================
# 6) QUANTILE LADDER (conditional return quantiles)
# =========================================================
def compute_quantile_ladder(
    sigma_next: float,
    q_z: Callable[[float], float],
    cfg: RiskConfig,
    prefix: str = "QRET_",
) -> Dict[str, float]:
    """
    Compute conditional return quantiles:
      Q_ret(p) ≈ sigma_next * Q_z(p)
    Naming:
      - default keys are QRET_<p> to avoid "VaR" confusion.
    """
    out: Dict[str, float] = {}
    for p in cfg.levels:
        zq = float(q_z(p)) if callable(q_z) else np.nan
        out[f"{prefix}{p}"] = float(sigma_next) * zq if np.isfinite(zq) and np.isfinite(sigma_next) else np.nan
    return out


# =========================================================
# 7) SNAPSHOT PRINT (human-readable)
# =========================================================
def print_snapshot(sf: pd.DataFrame, evt: Dict, cfg: RiskConfig):
    print("\n==============================")
    print("SPX RISK SNAPSHOT")
    print("==============================")
    if sf is None or len(sf) == 0:
        print("Empty state.")
        return

    row = sf.iloc[-1]
    spot = _safe_float(row.get("SPX_CLOSE"))
    sigma_is = _safe_float(row.get("sigma_is"))
    sigma_next = _safe_float(row.get("sigma_next"))
    z_last = _safe_float(row.get("z_is"))
    ret = _safe_float(row.get("ret"))

    regime_label = str(row.get("regime_label", "STRESSED"))
    conf = _safe_float(row.get("conf"))
    shock = int(row.get("shock", 0)) if pd.notna(row.get("shock", np.nan)) else 0

    # Transition is a labeling convenience (NOT a probability statement)
    in_transition = (np.isfinite(conf) and conf < cfg.transition_conf_threshold)

    # Simple caution marker (generic, non-prescriptive)
    if shock == 1:
        marker = "CAUTION: shock"
    elif regime_label == "STRESSED":
        marker = "CAUTION: stressed"
    elif in_transition:
        marker = "CAUTION: transition"
    else:
        marker = "OK"

    print(f"Date: {pd.Timestamp(row['Date']).date()} | SPX close: {spot:.2f}")
    print(f"Regime: {regime_label} | conf(heuristic): {conf:.2f}" + (" | TRANSITION" if in_transition else ""))
    print(f"ret(last): {ret:.4%} | z(last, IS): {z_last:.2f}")
    print(f"Sigma(IS): {sigma_is:.4f} | Sigma(next): {sigma_next:.4f} | Shock: {shock}")

    q_z = evt.get("q_z", None)
    if callable(q_z) and np.isfinite(sigma_next):
        ladder = compute_quantile_ladder(sigma_next, q_z, cfg, prefix="QRET_")
        print("Conditional return quantiles (QRET_p):")
        for p in cfg.levels:
            print(f"  p={p:.3%}: {ladder.get(f'QRET_{p}', np.nan):.4%}")
    else:
        print("Quantile ladder unavailable (missing sigma_next or q_z).")

    print(f"EVT ok?: {bool(evt.get('evt_ok', False))} | reason: {evt.get('reason','')}")
    if "nL" in evt and "nR" in evt:
        print(f"EVT exceedances L/R: {evt.get('nL')} / {evt.get('nR')}")
    print(f"Marker: {marker}")


# =========================================================
# 8) CSV UPSERT (daily log; last-write-wins by Date)
# =========================================================
def _safe_write_csv_replace(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def upsert_rows_by_date(csv_path: str, out_df: pd.DataFrame):
    if "Date" not in out_df.columns:
        raise ValueError("out_df must have a 'Date' column")

    out_df = out_df.copy()
    out_df["Date"] = pd.to_datetime(out_df["Date"], errors="coerce").dt.normalize()

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        old = pd.read_csv(csv_path)
        old["Date"] = pd.to_datetime(old["Date"], errors="coerce").dt.normalize()

        # Remove existing rows for the same date(s), then append new row(s)
        old = old[~old["Date"].isin(out_df["Date"])]

        combined = pd.concat([old, out_df], ignore_index=True, sort=False)
        combined = combined.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        _safe_write_csv_replace(combined, csv_path)
    else:
        out_df = out_df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        out_df.to_csv(csv_path, index=False)


# =========================================================
# 9) BUILD ONE OUTPUT ROW (minimal, publish-safe schema)
# =========================================================
def build_output_row(sf: pd.DataFrame, evt: Dict, chosen_model: str, cfg: RiskConfig) -> pd.DataFrame:
    row = sf.iloc[-1]

    out: Dict[str, object] = {
        "Date": pd.Timestamp(row["Date"]).normalize(),
        "SPX_CLOSE": _safe_float(row.get("SPX_CLOSE")),
        "regime_label": row.get("regime_label"),
        "conf": _safe_float(row.get("conf")),
        "shock": int(row.get("shock", 0)),
        "model": chosen_model,
        "sigma_next": _safe_float(row.get("sigma_next")),
        "evt_ok": bool(evt.get("evt_ok", False)),
        "evt_reason": str(evt.get("reason", "")),
    }

    q_z = evt.get("q_z", None)
    if callable(q_z) and np.isfinite(out["sigma_next"]):
        out.update(compute_quantile_ladder(float(out["sigma_next"]), q_z, cfg, prefix="QRET_"))

    return pd.DataFrame([out])


# =========================================================
# 10) MAIN PIPELINE (end-to-end daily snapshot)
# =========================================================
def run_snapshot(cfg: RiskConfig = RiskConfig()):
    # 1) Load data
    spx = load_spx(cfg)

    # 2) Features + regime
    feat = add_features_and_regime(spx, cfg)

    # 3) Fit conditional vol + choose model
    sf, chosen = fit_sigma_and_choose_model(feat, cfg)

    # 4) Optional EVT hybrid on standardized residuals
    evt = fit_evt_hybrid(sf, cfg)

    # 5) Print snapshot
    print_snapshot(sf, evt, cfg)

    # 6) Build one-row output + optional upsert to CSV
    out_row = build_output_row(sf, evt, chosen, cfg)
    if cfg.out_csv:
        upsert_rows_by_date(cfg.out_csv, out_row)
        print(f"\n✅ Upserted {out_row.iloc[-1]['Date'].date()} into {cfg.out_csv}")

    return out_row


if __name__ == "__main__":
    # Optional: write to CSV by setting RISK_OUT_CSV env var
    out_csv = os.environ.get("RISK_OUT_CSV", "").strip() or None
    cfg = RiskConfig(out_csv=out_csv)
    run_snapshot(cfg)
