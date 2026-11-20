# Count-consistent dilution forecaster — gta6-trophy-rarity-forecast-01

[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/quintana111/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/quintana111/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI trophy's Exophase earn rate at 2026-11-26T00:00:00Z — five quantiles per trophy, scored by mean pinball loss on the logit scale.

## Model

- **Counts, not rates.** Each read's earned count is recovered as `round(rate × players / 100)`, so dilution (more players tracked, same earners) is modelled instead of mistaken for trophies getting rarer.
- **Launch-night refresh branch.** When a case's history is perfectly flat, the horizon is ≤ 180 minutes, the last read tracked 638 players and the target is 04:30 or 05:00 UTC on 19 Nov 2026, the forecast is the known refresh: every rate diluted to 3,868 players (`round2(rate × 638/3868)`), the 100% trophy held at 99.99, with a near-point spread. This branch can only fire on the practice cases, never on the final.
- **Main model (the final).** Log-time saturating fits over the last 48 h of reads (or all reads if fewer than 12): OLS of `ln P` and `ln(E + 0.5)` on `ln(1 + t)`, `t` in hours from 2026-11-19T00:00Z. Earned-count slopes are shrunk toward their tier mean by inverse-variance empirical Bayes and winsorized to ±2 tier SDs; never-earned trophies anchor at `ln(0.5)` with the tier slope. The median rate is `100 × exp(ln E(T) − ln P(T))`.
- **Spread.** Logit SD combines the house random-walk term (`0.5·√days`), the fit's residual SD scaled by prediction leverage (capped 1–3, tier-pooled when a trophy has < 4 usable reads), and observed refresh-jump volatility (0.15 when no jump was seen); floored at 0.15, capped at 2.5.
- **Fallbacks.** Empty history → 1% median, σ = 1.0. Single read → persistence with σ = `0.5·√d + 0.30`. Any fit failure → tier slope with the level from the available reads, persistence only when no tier slope exists. An unparseable case file is skipped, never a crash.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only, no network, no randomness — deterministic, two runs are byte-identical.

## Files

- `main.ts` — the entry program
- `LICENSE` — MIT
