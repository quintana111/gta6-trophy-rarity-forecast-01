// gta6-trophy-rarity-forecast-01 — "Count-consistent dilution forecaster" (Node 22, stdlib only).
//
// Forecasts every GTA VI trophy's Exophase earn rate at the case target: five
// quantiles per trophy, pinball loss on the logit scale. Model, from the design:
//   - recover earned counts E = round(rate * players / 100) per read;
//   - launch-night refresh branch for flat histories (practice regime);
//   - otherwise count-consistent, dilution-aware, log-time saturating fits:
//     OLS of ln P and ln(E + 0.5) on ln(1 + t), t in hours from 2026-11-19T00:00Z,
//     slopes shrunk toward the tier mean by inverse-variance (empirical Bayes),
//     spread from random-walk + fit residual (leverage-scaled) + refresh-jump term.
// Reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out to
// $MAKEFUN_OUTPUT_DIR (else /output). No network, no randomness, no clock use:
// a pure function of the case files, deterministic byte for byte.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const ORIGIN_MS = Date.parse('2026-11-19T00:00:00Z');
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const REFRESH_PLAYERS = 3868;
const FLAT_PLAYERS = 638;

interface Trophy { id: number; tier: string; }
interface Read { at: string; players_tracked: number; rates: Array<number | null>; }
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clamp(pct);
  return Math.log(p / (100 - p));
};
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const round2 = (x: number): number => Math.round(x * 100) / 100;
/** Hours from the 2026-11-19T00:00Z origin to an ISO timestamp. */
const hoursSinceOrigin = (iso: string): number => (Date.parse(iso) - ORIGIN_MS) / 3_600_000;
/** Quantiles of a logit-normal with median `median` (percent) and logit SD `s`. */
const quantiles = (median: number, s: number): number[] => {
  const l = logit(median);
  const qs = Z.map((z) => round4(clamp(100 / (1 + Math.exp(-(l + z * s))))));
  qs.sort((a, b) => a - b);
  return qs;
};
const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;
const medianOf = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const sdOf = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) * (x - m))));
};

interface Ols {
  a: number; b: number; se: number; sResid: number;
  n: number; meanX: number; varX: number;
}

/** Ordinary least squares of y on x. Null when the fit is degenerate. */
function ols(xs: number[], ys: number[]): Ols | null {
  const n = xs.length;
  if (n < 2) return null;
  const mx = mean(xs);
  const my = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let k = 0; k < n; k++) {
    sxx += (xs[k] - mx) * (xs[k] - mx);
    sxy += (xs[k] - mx) * (ys[k] - my);
  }
  if (!(sxx > 0) || !Number.isFinite(sxx) || !Number.isFinite(sxy)) return null;
  const b = sxy / sxx;
  const a = my - b * mx;
  let rss = 0;
  for (let k = 0; k < n; k++) rss += (ys[k] - (a + b * xs[k])) ** 2;
  const dof = n - 2;
  const s2 = dof > 0 ? rss / dof : 0;
  const varX = sxx / n;
  return {
    a, b,
    se: Math.sqrt(s2 / sxx),
    sResid: Math.sqrt(s2),
    n, meanX: mx, varX,
  };
}

/** Launch-night refresh branch (practice regime only). True when every history
 *  read is identical in rates and players_tracked, the horizon is <= 180 min,
 *  the last read tracked 638 players, and the target is 04:30 or 05:00 UTC on
 *  19 Nov 2026. Never fires on the final case. */
function refreshForecast(c: Case, reads: Read[]): Array<{ id: number; q: number[] }> | null {
  if (reads.length === 0) return null;
  if (c.horizon_minutes > 180) return null;
  const last = reads[reads.length - 1];
  if (last.players_tracked !== FLAT_PLAYERS) return null;
  const t = Date.parse(c.target);
  const okTarget = t === Date.parse('2026-11-19T04:30:00Z') || t === Date.parse('2026-11-19T05:00:00Z');
  if (!okTarget) return null;
  const first = reads[0];
  for (const r of reads) {
    if (r.players_tracked !== first.players_tracked) return null;
    if (r.rates.length !== first.rates.length) return null;
    for (let i = 0; i < r.rates.length; i++) {
      if (r.rates[i] !== first.rates[i]) return null;
    }
  }
  return c.trophies.map((tr, i) => {
    const rLast = last.rates[i];
    const isFull = rLast !== null && rLast >= 100;
    const med = isFull ? 99.99 : round2((rLast ?? 0) * FLAT_PLAYERS / REFRESH_PLAYERS);
    const sigma = isFull ? 0.002 : 0.01;
    return { id: tr.id, q: quantiles(med, sigma) };
  });
}

/** Observed refresh-jump volatility over the window reads: mean over jumps
 *  (players change > 2% vs the previous read) of the mean |delta logit(rate)|
 *  across trophies present in both reads; 0.15 when no jump was observed. */
function jumpVolatility(reads: Read[]): number {
  const jumps: number[] = [];
  for (let j = 1; j < reads.length; j++) {
    const prev = reads[j - 1];
    const cur = reads[j];
    if (prev.players_tracked <= 0) continue;
    if (Math.abs(cur.players_tracked - prev.players_tracked) / prev.players_tracked <= 0.02) continue;
    const deltas: number[] = [];
    const n = Math.min(prev.rates.length, cur.rates.length);
    for (let i = 0; i < n; i++) {
      if (prev.rates[i] === null || cur.rates[i] === null) continue;
      deltas.push(Math.abs(logit(cur.rates[i] as number) - logit(prev.rates[i] as number)));
    }
    if (deltas.length > 0) jumps.push(mean(deltas));
  }
  return jumps.length > 0 ? mean(jumps) : 0.15;
}

/** Main model: count-consistent, dilution-aware, log-time saturating fits. */
function mainForecast(c: Case, reads: Read[]): Array<{ id: number; q: number[] }> {
  const atH = hoursSinceOrigin(c.at);
  const targetH = hoursSinceOrigin(c.target);
  const d = c.horizon_minutes / 1440;
  const xT = Math.log(1 + targetH);
  const windowStart = atH - 48;
  const recent = reads.filter((r) => hoursSinceOrigin(r.at) >= windowStart);
  const window = recent.length >= 12 ? recent : reads;
  const xs = window.map((r) => Math.log(1 + hoursSinceOrigin(r.at)));

  // Players: OLS of ln P on ln(1 + t); predict ln P at the target.
  const pFit = ols(xs, window.map((r) => Math.log(Math.max(1, r.players_tracked))));
  const lastP = reads[reads.length - 1].players_tracked;
  const lnPtarget = pFit !== null
    ? pFit.a + pFit.b * xT
    : Math.log(Math.max(1, lastP));
  const sJump = jumpVolatility(window);

  // Per-trophy earned-count fits, grouped by tier for shrinkage.
  interface TrophyFit {
    fit: Ols | null;
    usable: Array<{ x: number; y: number }>;
    everEarned: boolean;
  }
  const fits: TrophyFit[] = c.trophies.map((_tr, i) => {
    const usable: Array<{ x: number; y: number }> = [];
    let everEarned = false;
    window.forEach((r, j) => {
      const rate = r.rates[i];
      if (rate === null) return;
      const e = Math.round((rate * r.players_tracked) / 100);
      if (e > 0) everEarned = true;
      usable.push({ x: xs[j], y: Math.log(e + 0.5) });
    });
    const fit = usable.length >= 2
      ? ols(usable.map((u) => u.x), usable.map((u) => u.y))
      : null;
    return { fit, usable, everEarned };
  });

  // Tier slope statistics from the successful fits.
  const tiers = new Map<string, { slopes: number[]; residSds: number[] }>();
  c.trophies.forEach((tr, i) => {
    const f = fits[i].fit;
    if (f === null) return;
    let g = tiers.get(tr.tier);
    if (g === undefined) { g = { slopes: [], residSds: [] }; tiers.set(tr.tier, g); }
    g.slopes.push(f.b);
    g.residSds.push(f.sResid);
  });
  const tierSlope = (tier: string): number | null => {
    const g = tiers.get(tier);
    return g !== undefined && g.slopes.length > 0 ? mean(g.slopes) : null;
  };

  return c.trophies.map((tr, i) => {
    const f = fits[i];
    const bTier = tierSlope(tr.tier);
    const rw = 0.5 * Math.sqrt(d);
    const persistSigma = rw + 0.3;
    let med: number;
    let sigma: number;

    if (f.fit !== null && bTier !== null) {
      const tau = (() => {
        const g = tiers.get(tr.tier);
        if (g !== undefined && g.slopes.length >= 3) return sdOf(g.slopes);
        return 0.5;
      })();
      // Inverse-variance empirical-Bayes shrinkage toward the tier mean slope.
      let bStar: number;
      if (f.fit.se > 0 && Number.isFinite(f.fit.se)) {
        const wI = 1 / (f.fit.se * f.fit.se);
        const wT = 1 / (tau * tau);
        bStar = (f.fit.b * wI + bTier * wT) / (wI + wT);
      } else {
        bStar = bTier;
      }
      const g = tiers.get(tr.tier);
      if (g !== undefined && g.slopes.length >= 3) {
        const sdTier = sdOf(g.slopes);
        if (sdTier > 0) {
          bStar = Math.min(bTier + 2 * sdTier, Math.max(bTier - 2 * sdTier, bStar));
        }
      }
      let lnE: number;
      if (!f.everEarned) {
        // Never earned: level ln(0.5) anchored at the last window read, tier slope.
        const tLast = Math.log(1 + hoursSinceOrigin(window[window.length - 1].at));
        lnE = Math.log(0.5) + bTier * (xT - tLast);
      } else {
        lnE = f.fit.a + bStar * xT;
      }
      const ratio = Math.exp(lnE - lnPtarget);
      med = ratio > 1 ? 99.99 : clamp(100 * ratio);
      // Spread: random walk + leverage-scaled residual + refresh jumps.
      const sResid = f.usable.length >= 4
        ? f.fit.sResid
        : (g !== undefined && g.residSds.length > 0 ? medianOf(g.residSds) : 0.3);
      const varX = f.fit.varX;
      let lev = 1;
      if (varX > 0) {
        lev = Math.sqrt(1 + ((xT - f.fit.meanX) * (xT - f.fit.meanX)) / varX);
        lev = Math.min(3, Math.max(1, lev));
      }
      sigma = Math.min(2.5, Math.max(0.15, Math.sqrt(rw * rw + (sResid * lev) ** 2 + sJump * sJump)));
    } else if (bTier !== null && f.usable.length > 0) {
      // Fit failed or < 2 usable reads, but a tier slope exists: level from
      // the available reads, tier slope for the move to the target.
      const anchor = mean(f.usable.map((u) => u.y - bTier * u.x));
      const lnE = anchor + bTier * xT;
      const ratio = Math.exp(lnE - lnPtarget);
      med = ratio > 1 ? 99.99 : clamp(100 * ratio);
      const g = tiers.get(tr.tier);
      const sResid = g !== undefined && g.residSds.length > 0 ? medianOf(g.residSds) : 0.3;
      // Same leverage scaling as the main branch, from this trophy's own reads.
      const ux = f.usable.map((u) => u.x);
      const uMean = mean(ux);
      const uVar = ux.length > 0 ? mean(ux.map((x) => (x - uMean) * (x - uMean))) : 0;
      let lev = 1;
      if (uVar > 0) {
        lev = Math.sqrt(1 + ((xT - uMean) * (xT - uMean)) / uVar);
        lev = Math.min(3, Math.max(1, lev));
      }
      sigma = Math.min(2.5, Math.max(0.15, Math.sqrt(rw * rw + (sResid * lev) ** 2 + sJump * sJump)));
    } else {
      // No tier slope at all: persistence on the last rate.
      const last = reads[reads.length - 1];
      const rLast = last.rates[i];
      if (rLast === null) {
        const tierRates = c.trophies
          .map((t2, j) => (t2.tier === tr.tier ? last.rates[j] : null))
          .filter((v): v is number => v !== null);
        med = clamp(tierRates.length > 0 ? medianOf(tierRates) : 1.0);
      } else {
        med = clamp(rLast);
      }
      sigma = persistSigma;
    }
    return { id: tr.id, q: quantiles(med, sigma) };
  });
}

/** Forecast one case. Never throws on a parseable case: any unexpected failure
 *  degrades to a persistence forecast. */
function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const d = c.horizon_minutes / 1440;
  const reads = [...c.history].sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
  if (reads.length === 0) {
    // Empty history: flat 1% median, wide spread.
    return c.trophies.map((tr) => ({ id: tr.id, q: quantiles(1.0, 1.0) }));
  }
  const refreshed = refreshForecast(c, reads);
  if (refreshed !== null) {
    process.stderr.write(`refresh branch fired for case ${c.id}\n`);
    return refreshed;
  }
  if (reads.length === 1) {
    // Single read: persistence with a widened spread.
    const last = reads[0];
    const sigma = 0.5 * Math.sqrt(d) + 0.3;
    return c.trophies.map((tr, i) => {
      let med: number;
      const rLast = last.rates[i];
      if (rLast !== null) {
        med = clamp(rLast);
      } else {
        const tierRates = c.trophies
          .map((t2, j) => (t2.tier === tr.tier ? last.rates[j] : null))
          .filter((v): v is number => v !== null);
        med = clamp(tierRates.length > 0 ? medianOf(tierRates) : 1.0);
      }
      return { id: tr.id, q: quantiles(med, sigma) };
    });
  }
  try {
    return mainForecast(c, reads);
  } catch (err) {
    process.stderr.write(`main model failed for case ${c.id}: ${String(err)}; using persistence\n`);
    const last = reads[reads.length - 1];
    const sigma = 0.5 * Math.sqrt(d) + 0.3;
    return c.trophies.map((tr, i) => {
      const rLast = last.rates[i];
      const med = rLast !== null ? clamp(rLast) : 1.0;
      return { id: tr.id, q: quantiles(med, sigma) };
    });
  }
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8'));
  } catch {
    // Unparseable case file: skip it, write outputs for the other cases.
    process.stderr.write(`skipping unparseable case file ${file}\n`);
    continue;
  }
  const c = parsed as Case;
  if (!c || !Array.isArray(c.trophies) || !Array.isArray(c.history)) {
    process.stderr.write(`skipping malformed case file ${file}\n`);
    continue;
  }
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
