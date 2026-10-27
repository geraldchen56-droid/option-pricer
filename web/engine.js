// Root T — pricing engine (port of option_pricer.gbm / monte_carlo / vectorized)
(function (global) {
  "use strict";

  // Seeded RNG so a given seed always gives the same "rolls of the dice".
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // n standard normal draws via Box–Muller, returned SORTED ascending.
  // S_T is increasing in Z, so sorting Z once sorts S_T for every sigma and T.
  function sortedNormals(n, seed) {
    const rand = mulberry32(seed);
    const z = new Float64Array(n);
    for (let i = 0; i < n; i += 2) {
      let u1 = rand();
      if (u1 < 1e-300) u1 = 1e-300;
      const u2 = rand();
      const rad = Math.sqrt(-2 * Math.log(u1));
      z[i] = rad * Math.cos(2 * Math.PI * u2);
      if (i + 1 < n) z[i + 1] = rad * Math.sin(2 * Math.PI * u2);
    }
    return z.sort();
  }

  // Standard normal CDF (Abramowitz & Stegun 7.1.26 via erf, |error| < 1.5e-7)
  function normCdf(x) {
    const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
      - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x / 2);
    return x >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
  }

  function bsCall(S0, K, T, r, sigma) {
    const sq = sigma * Math.sqrt(T);
    const d1 = (Math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / sq;
    const d2 = d1 - sq;
    return S0 * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d2);
  }

  // First index j with z[j] > x (binary search on sorted z)
  function upperBound(z, x) {
    let lo = 0, hi = z.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (z[mid] > x) hi = mid; else lo = mid + 1;
    }
    return lo;
  }

  // Monte Carlo call prices for one maturity across many strikes.
  // Uses suffix sums of S_T (and S_T^2 for the standard error), so each strike
  // costs a binary search instead of a pass over all n simulations.
  function mcSlice(z, S0, r, sigma, T, strikes, buf) {
    const n = z.length;
    const drift = (r - 0.5 * sigma * sigma) * T;
    const vol = sigma * Math.sqrt(T);
    const suf = buf.suf, suf2 = buf.suf2;
    suf[n] = 0; suf2[n] = 0;
    for (let j = n - 1; j >= 0; j--) {
      const s = S0 * Math.exp(drift + vol * z[j]);
      suf[j] = suf[j + 1] + s;
      suf2[j] = suf2[j + 1] + s * s;
    }
    const disc = Math.exp(-r * T);
    const prices = new Float64Array(strikes.length);
    const ses = new Float64Array(strikes.length);
    for (let k = 0; k < strikes.length; k++) {
      const K = strikes[k];
      const j = upperBound(z, (Math.log(K / S0) - drift) / vol); // first S_T > K
      const cnt = n - j;
      const sum = suf[j] - K * cnt;                               // Σ (S_T − K)+
      const sumSq = suf2[j] - 2 * K * suf[j] + K * K * cnt;      // Σ ((S_T − K)+)²
      const mean = sum / n;
      const variance = Math.max(sumSq / n - mean * mean, 0) * n / (n - 1);
      prices[k] = disc * mean;
      ses[k] = disc * Math.sqrt(variance / n);
    }
    return { prices, ses };
  }

  function makeBuffers(n) {
    return { suf: new Float64Array(n + 1), suf2: new Float64Array(n + 1) };
  }

  function linspace(a, b, m) {
    const out = new Float64Array(m);
    for (let i = 0; i < m; i++) out[i] = a + (b - a) * i / (m - 1);
    return out;
  }

  global.RootT = { sortedNormals, normCdf, bsCall, mcSlice, makeBuffers, linspace };
})(typeof window !== "undefined" ? window : globalThis);
