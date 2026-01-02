# SPX Conditional Risk Engine

## Overview

This repository documents a **research-grade SPX conditional risk engine** designed to estimate **volatility-driven risk states**, **conditional volatility**, and **downside tail behavior** using only information available at the time of evaluation.

The engine is **risk-first and execution-agnostic** by design.

It is intended to serve as an **upstream risk diagnostics layer** that can inform downstream decision frameworks (for example, whether market risk should be taken), while remaining strictly separated from trading logic, execution, and performance evaluation.

This project focuses on **risk estimation architecture**, not on producing trading signals or deployable strategies.

---

## Quickstart

To run the public after-close snapshot locally:

1. Install dependencies:
   
   `pip install -r requirements.txt`

2. Run the snapshot script:

   `python "SPX risk snapshot (public).py"`

The script produces a single **after-close risk snapshot** based on SPX end-of-day data.

---

## What This Project Is

This project is:

- A standalone **conditional risk diagnostics engine** for the S&P 500 index  
- Focused on **probability, distribution, and tail risk**, not price direction  
- Built with **leakage-aware, walk-forward discipline** as a core design principle  
- Designed to be **auditable, conservative, and failure-aware**  

The architecture reflects how institutional risk teams typically separate **risk estimation** from **decision-making and execution**.

---

## What This Project Is Not

This repository intentionally does **not** include:

- Trading rules, alpha signals, or execution logic  
- Option structures, strike selection, or premium evaluation  
- Capital allocation rules, policy thresholds, or cooldown logic  
- Backtests, P&L attribution, or performance claims  

Any downstream use of the outputs (if any) is considered **out of scope** for this repository.

---

## Methodology (High Level)

### Leakage-Safe Feature Construction

All realized-volatility features are constructed using **only historical data available prior to the evaluation date**. Rolling volatility measures are explicitly shifted to prevent look-ahead bias.

### Conditional Risk State Labeling

A two-state conditional risk label (**CALM / STRESSED**) is inferred from:

- Volatility level
- Volatility persistence
- Volatility slope behavior

This labeling is intended to describe **risk conditions**, not market direction.

### Conditional Volatility Modeling

Conditional volatility is estimated using **GARCH-family models with Student-t innovations**, specifically:

- EGARCH  
- GJR-GARCH  

Model selection is performed conservatively based on **downside tail underestimation diagnostics**, rather than in-sample likelihood or fit quality.

### Distributional Tail Modeling

Extreme tail behavior is characterized using a **hybrid empirical + EVT/GPD framework**:

- Empirical quantiles are used in the center of the distribution  
- EVT/GPD modeling is applied only when stability conditions are met  
- Safe empirical fallbacks are used when EVT assumptions fail  

This ensures robustness in stressed or data-limited regimes.

### Diagnostic Quantile Ladder

The engine produces a **probability-indexed conditional return quantile ladder** for next-day downside risk.  
This is often colloquially referred to as a “VaR ladder,” but is presented strictly as a **diagnostic distributional view**, not a risk limit or trading constraint.

---

## Outputs (Conceptual)

A typical daily snapshot includes:

- Conditional risk state label (CALM / STRESSED)  
- Heuristic interpretability / confidence score  
- One-day-ahead conditional volatility estimate  
- Diagnostic left-tail conditional return quantiles across multiple probabilities  
- Shock flag indicating abnormal return behavior  

All outputs are **diagnostic only** and do **not** constitute trading signals, forecasts, or recommendations.

---

## Walk-Forward Discipline

All features and estimates are computed using **only information available at the time of the snapshot**.  
No future data is used in risk labeling, volatility estimation, or tail modeling.

This repository focuses on the **single-snapshot risk pipeline**. When embedded in a broader research or production framework, the same pipeline is intended to be invoked in a **walk-forward manner**.

Operational policy governance (such as when or how to act on risk diagnostics) is intentionally excluded.

---

## Governance Note

This repository represents a **policy-sanitized research implementation**.

Downstream decision rules, capital constraints, execution mechanics, and live deployment parameters are deliberately omitted to preserve model clarity, prevent overfitting, and maintain a clean separation between **risk estimation** and **decision-making**.

---

## Intended Use

This project is suitable for:

- Risk research and education  
- Architecture discussion in interviews or design reviews  
- Demonstrating probabilistic and tail-risk-aware thinking  
- Illustrating institutional separation between risk diagnostics and trading decisions  

It is **not** intended to be used as a standalone trading system.

---

## License

MIT License.
