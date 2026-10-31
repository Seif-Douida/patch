# Phase 4 causal benchmark

Known effects injected into real quiet stretches of review history (design D12): each
method's error and 90% interval behaviour, measured where the truth is known.

| Run | |
|---|---|
| trials per cell | 200 |
| seed | 20261021 |
| snapshot (max run id) | 40 |
| effects version | e2 |
| tracked games | 12 |
| comparison games | 30 |
| quiet windows | 2302 |
| dry run | False |

## The rule (ADR-007, fixed before the first run)

Among methods whose 90% intervals contain the injected effect in 85% to 95% of the
positive-share trials (non-zero effects) and whose false-alarm rate at an injected 0
is at most 10%, the default is the one with the lower RMSE. With none, interrupted time
series, and the method note says its intervals weren't validated.

| Method | Coverage | False alarms at 0 | RMSE (points) |
|---|---|---|---|
| its | 93% | 7% | 3.35 |
| synth | 91% | 7% | 3.12 |

Default method: synth (eligible: its, synth; lowest RMSE: synth).

## Both methods on the same trials

The 2683 positive-share trials where both methods had an estimate (the rule above counts each method's own).

| Method | Trials | Coverage | False alarms at 0 | RMSE (points) |
|---|---|---|---|---|
| its | 2683 | 93% | 7% | 3.35 |
| synth | 2683 | 92% | 6% | 3.25 |

Synthetic control fell back to interrupted time series in 37% of its trials (another patch in its fit window, fewer than 5 qualifying comparison games, or a twin that didn't track).

Interrupted time series' intervals are calibrated on quiet dates in the game's own past (ADR-007 amendment 1); 15% of its trials had no estimate (too little quiet history to calibrate, or too few reviews).

## Windows with a volume spike

14% of the trials ran on a window with a volume spike: the site's `volume_spike` flag, a day in the horizon with reviews 3 SD above the 28 days before (log scale). Quiet windows avoid patches and Steam's seasonal sales, so these are events neither list holds: a game's own discount before 30 Sep 2026, an anniversary, an update the classifier didn't mark as a patch. The rule above pools them, as the nightly estimates meet them; the table shows how much less a flagged estimate can be trusted. Bias and RMSE in points (positive share) or percent (volume).

| Outcome | Method | Volume spike | Trials | Bias | RMSE | Coverage | False alarms at 0 |
|---|---|---|---|---|---|---|---|
| Positive share | its | no | 2279 | -0.12 | 2.94 | 95% | 6% |
| Positive share | its | yes | 404 | -0.62 | 5.08 | 78% | 10% |
| Positive share | synth | no | 2599 | -0.04 | 2.79 | 93% | 7% |
| Positive share | synth | yes | 438 | -0.75 | 4.61 | 83% | 9% |
| Volume | its | no | 1743 | +4.88 | 39.74 | 94% | 3% |
| Volume | its | yes | 316 | +172.67 | 262.97 | 66% | 28% |
| Volume | synth | no | 1855 | +3.29 | 37.73 | 92% | 5% |
| Volume | synth | yes | 323 | +161.53 | 257.47 | 66% | 29% |

Causal wording on volume-spike windows (Phase 5): allowed. The default method raised false alarms on 9% of 114 spike-window trials at an injected 0 (allowed: at most 10% over at least 20 trials).

## Positive share (points)

| Method | Horizon | Tier | Effect | Shape | n | Bias | RMSE | Coverage | False alarms | Power |
|---|---|---|---|---|---|---|---|---|---|---|
| its | 7 | 50-150 | -10 | ramp | 70 | +0.09 | +2.97 | 99% | - | 53% |
| its | 7 | 50-150 | -10 | step | 60 | +0.76 | +3.44 | 92% | - | 72% |
| its | 7 | 50-150 | -5 | ramp | 64 | -0.44 | +2.58 | 92% | - | 31% |
| its | 7 | 50-150 | -5 | step | 53 | +0.15 | +3.05 | 87% | - | 53% |
| its | 7 | 50-150 | -2 | ramp | 66 | +0.10 | +2.22 | 89% | - | 9% |
| its | 7 | 50-150 | -2 | step | 59 | -0.08 | +3.26 | 90% | - | 24% |
| its | 7 | 50-150 | +0 | ramp | 76 | -0.08 | +2.82 | 97% | 3% | - |
| its | 7 | 50-150 | +0 | step | 59 | +0.72 | +2.66 | 88% | 12% | - |
| its | 7 | over 150 | -10 | ramp | 57 | -0.42 | +3.15 | 95% | - | 60% |
| its | 7 | over 150 | -10 | step | 72 | -1.20 | +6.11 | 89% | - | 96% |
| its | 7 | over 150 | -5 | ramp | 62 | +0.23 | +1.32 | 95% | - | 24% |
| its | 7 | over 150 | -5 | step | 65 | -0.12 | +1.34 | 95% | - | 83% |
| its | 7 | over 150 | -2 | ramp | 65 | -0.19 | +3.82 | 91% | - | 18% |
| its | 7 | over 150 | -2 | step | 70 | -0.76 | +3.51 | 89% | - | 47% |
| its | 7 | over 150 | +0 | ramp | 53 | -0.12 | +1.50 | 92% | 8% | - |
| its | 7 | over 150 | +0 | step | 64 | -0.08 | +1.34 | 88% | 12% | - |
| its | 7 | under 50 | -10 | ramp | 46 | -0.32 | +4.32 | 100% | - | 17% |
| its | 7 | under 50 | -10 | step | 44 | -0.92 | +3.79 | 93% | - | 70% |
| its | 7 | under 50 | -5 | ramp | 43 | -0.45 | +3.94 | 93% | - | 9% |
| its | 7 | under 50 | -5 | step | 49 | +0.67 | +3.64 | 100% | - | 12% |
| its | 7 | under 50 | -2 | ramp | 47 | +0.27 | +3.54 | 91% | - | 6% |
| its | 7 | under 50 | -2 | step | 40 | -0.99 | +3.77 | 92% | - | 18% |
| its | 7 | under 50 | +0 | ramp | 42 | -0.53 | +4.17 | 95% | 5% | - |
| its | 7 | under 50 | +0 | step | 45 | -0.66 | +3.29 | 98% | 2% | - |
| its | 14 | 50-150 | -10 | ramp | 70 | +0.44 | +3.20 | 96% | - | 74% |
| its | 14 | 50-150 | -10 | step | 58 | +0.89 | +3.44 | 90% | - | 78% |
| its | 14 | 50-150 | -5 | ramp | 64 | -0.21 | +3.26 | 89% | - | 42% |
| its | 14 | 50-150 | -5 | step | 52 | +0.08 | +2.91 | 94% | - | 50% |
| its | 14 | 50-150 | -2 | ramp | 64 | +0.21 | +2.49 | 92% | - | 12% |
| its | 14 | 50-150 | -2 | step | 59 | +0.14 | +3.30 | 85% | - | 27% |
| its | 14 | 50-150 | +0 | ramp | 74 | +0.24 | +3.21 | 92% | 8% | - |
| its | 14 | 50-150 | +0 | step | 58 | +0.75 | +2.47 | 97% | 3% | - |
| its | 14 | over 150 | -10 | ramp | 54 | -0.28 | +3.74 | 93% | - | 74% |
| its | 14 | over 150 | -10 | step | 67 | -1.57 | +6.13 | 88% | - | 94% |
| its | 14 | over 150 | -5 | ramp | 54 | +0.12 | +1.48 | 100% | - | 44% |
| its | 14 | over 150 | -5 | step | 58 | -0.74 | +3.08 | 93% | - | 79% |
| its | 14 | over 150 | -2 | ramp | 58 | -0.34 | +4.21 | 90% | - | 33% |
| its | 14 | over 150 | -2 | step | 64 | -1.15 | +3.80 | 81% | - | 47% |
| its | 14 | over 150 | +0 | ramp | 52 | -0.88 | +2.93 | 90% | 10% | - |
| its | 14 | over 150 | +0 | step | 57 | -0.54 | +2.31 | 89% | 11% | - |
| its | 14 | under 50 | -10 | ramp | 45 | -0.18 | +4.22 | 100% | - | 31% |
| its | 14 | under 50 | -10 | step | 42 | -0.63 | +3.09 | 90% | - | 79% |
| its | 14 | under 50 | -5 | ramp | 43 | -0.15 | +2.59 | 100% | - | 19% |
| its | 14 | under 50 | -5 | step | 49 | +0.16 | +2.65 | 100% | - | 27% |
| its | 14 | under 50 | -2 | ramp | 47 | -0.04 | +3.11 | 94% | - | 9% |
| its | 14 | under 50 | -2 | step | 39 | -0.86 | +3.46 | 95% | - | 13% |
| its | 14 | under 50 | +0 | ramp | 40 | -0.32 | +3.86 | 100% | 0% | - |
| its | 14 | under 50 | +0 | step | 44 | -0.72 | +3.42 | 93% | 7% | - |
| synth | 7 | 50-150 | -10 | ramp | 75 | +0.01 | +2.95 | 91% | - | 65% |
| synth | 7 | 50-150 | -10 | step | 65 | +0.80 | +3.23 | 88% | - | 86% |
| synth | 7 | 50-150 | -5 | ramp | 73 | -0.14 | +2.51 | 92% | - | 36% |
| synth | 7 | 50-150 | -5 | step | 56 | +0.20 | +2.87 | 84% | - | 62% |
| synth | 7 | 50-150 | -2 | ramp | 68 | -0.10 | +2.05 | 90% | - | 16% |
| synth | 7 | 50-150 | -2 | step | 65 | -0.04 | +2.83 | 92% | - | 25% |
| synth | 7 | 50-150 | +0 | ramp | 79 | -0.07 | +2.74 | 96% | 4% | - |
| synth | 7 | 50-150 | +0 | step | 65 | +0.54 | +2.37 | 92% | 8% | - |
| synth | 7 | over 150 | -10 | ramp | 71 | +0.02 | +2.80 | 96% | - | 69% |
| synth | 7 | over 150 | -10 | step | 82 | -0.66 | +5.72 | 83% | - | 89% |
| synth | 7 | over 150 | -5 | ramp | 72 | +0.26 | +1.49 | 90% | - | 40% |
| synth | 7 | over 150 | -5 | step | 82 | +0.26 | +1.65 | 91% | - | 83% |
| synth | 7 | over 150 | -2 | ramp | 75 | -0.21 | +3.63 | 92% | - | 17% |
| synth | 7 | over 150 | -2 | step | 87 | -0.70 | +3.37 | 91% | - | 44% |
| synth | 7 | over 150 | +0 | ramp | 67 | +0.02 | +1.99 | 96% | 4% | - |
| synth | 7 | over 150 | +0 | step | 80 | -0.20 | +2.04 | 90% | 10% | - |
| synth | 7 | under 50 | -10 | ramp | 47 | -0.53 | +4.44 | 100% | - | 21% |
| synth | 7 | under 50 | -10 | step | 44 | -0.68 | +3.59 | 98% | - | 70% |
| synth | 7 | under 50 | -5 | ramp | 44 | -0.53 | +3.84 | 91% | - | 16% |
| synth | 7 | under 50 | -5 | step | 50 | +0.64 | +3.86 | 94% | - | 28% |
| synth | 7 | under 50 | -2 | ramp | 47 | +0.16 | +3.40 | 96% | - | 0% |
| synth | 7 | under 50 | -2 | step | 40 | -0.88 | +3.71 | 95% | - | 12% |
| synth | 7 | under 50 | +0 | ramp | 44 | -0.49 | +4.01 | 98% | 2% | - |
| synth | 7 | under 50 | +0 | step | 46 | -0.62 | +3.34 | 98% | 2% | - |
| synth | 14 | 50-150 | -10 | ramp | 75 | +0.12 | +2.81 | 89% | - | 61% |
| synth | 14 | 50-150 | -10 | step | 65 | +0.74 | +2.92 | 89% | - | 86% |
| synth | 14 | 50-150 | -5 | ramp | 73 | -0.31 | +2.61 | 89% | - | 55% |
| synth | 14 | 50-150 | -5 | step | 56 | +0.14 | +2.51 | 89% | - | 71% |
| synth | 14 | 50-150 | -2 | ramp | 68 | -0.10 | +1.95 | 87% | - | 18% |
| synth | 14 | 50-150 | -2 | step | 65 | +0.25 | +2.70 | 91% | - | 32% |
| synth | 14 | 50-150 | +0 | ramp | 79 | +0.27 | +2.76 | 90% | 10% | - |
| synth | 14 | 50-150 | +0 | step | 65 | +0.41 | +2.13 | 89% | 11% | - |
| synth | 14 | over 150 | -10 | ramp | 69 | -0.22 | +3.34 | 94% | - | 74% |
| synth | 14 | over 150 | -10 | step | 81 | -0.83 | +5.54 | 89% | - | 95% |
| synth | 14 | over 150 | -5 | ramp | 72 | +0.16 | +1.45 | 90% | - | 57% |
| synth | 14 | over 150 | -5 | step | 81 | -0.11 | +2.78 | 88% | - | 79% |
| synth | 14 | over 150 | -2 | ramp | 74 | -0.34 | +3.86 | 93% | - | 20% |
| synth | 14 | over 150 | -2 | step | 85 | -0.76 | +2.02 | 89% | - | 52% |
| synth | 14 | over 150 | +0 | ramp | 66 | -0.36 | +2.46 | 91% | 9% | - |
| synth | 14 | over 150 | +0 | step | 76 | -0.41 | +2.52 | 93% | 7% | - |
| synth | 14 | under 50 | -10 | ramp | 47 | -0.31 | +4.18 | 98% | - | 40% |
| synth | 14 | under 50 | -10 | step | 44 | -0.59 | +3.07 | 93% | - | 75% |
| synth | 14 | under 50 | -5 | ramp | 44 | -0.29 | +2.68 | 98% | - | 23% |
| synth | 14 | under 50 | -5 | step | 51 | +0.02 | +2.97 | 98% | - | 31% |
| synth | 14 | under 50 | -2 | ramp | 47 | -0.06 | +3.08 | 89% | - | 9% |
| synth | 14 | under 50 | -2 | step | 40 | -0.71 | +3.61 | 92% | - | 10% |
| synth | 14 | under 50 | +0 | ramp | 44 | -0.58 | +3.86 | 95% | 5% | - |
| synth | 14 | under 50 | +0 | step | 46 | -0.70 | +3.55 | 91% | 9% | - |

## Volume (percent)

| Method | Horizon | Tier | Effect | Shape | n | Bias | RMSE | Coverage | False alarms | Power |
|---|---|---|---|---|---|---|---|---|---|---|
| its | 7 | 50-150 | -25 | ramp | 62 | +37.52 | +120.01 | 84% | - | 16% |
| its | 7 | 50-150 | -25 | step | 62 | +24.44 | +83.18 | 92% | - | 16% |
| its | 7 | 50-150 | -10 | ramp | 71 | +62.54 | +230.97 | 83% | - | 14% |
| its | 7 | 50-150 | -10 | step | 64 | +51.84 | +159.44 | 81% | - | 16% |
| its | 7 | 50-150 | +0 | ramp | 83 | +43.87 | +150.31 | 88% | 12% | - |
| its | 7 | 50-150 | +0 | step | 61 | +17.81 | +78.39 | 93% | 7% | - |
| its | 7 | over 150 | -25 | ramp | 61 | +17.50 | +71.16 | 90% | - | 10% |
| its | 7 | over 150 | -25 | step | 80 | +27.27 | +100.51 | 88% | - | 34% |
| its | 7 | over 150 | -10 | ramp | 60 | +15.75 | +54.84 | 93% | - | 5% |
| its | 7 | over 150 | -10 | step | 68 | +40.20 | +142.71 | 93% | - | 12% |
| its | 7 | over 150 | +0 | ramp | 61 | +29.84 | +98.24 | 92% | 8% | - |
| its | 7 | over 150 | +0 | step | 69 | +56.63 | +140.92 | 88% | 12% | - |
| its | 7 | under 50 | -25 | ramp | 52 | +6.40 | +23.52 | 92% | - | 4% |
| its | 7 | under 50 | -25 | step | 36 | +10.80 | +39.70 | 94% | - | 19% |
| its | 7 | under 50 | -10 | ramp | 41 | +8.97 | +21.24 | 93% | - | 0% |
| its | 7 | under 50 | -10 | step | 45 | +10.37 | +31.26 | 98% | - | 0% |
| its | 7 | under 50 | +0 | ramp | 36 | +11.11 | +25.98 | 100% | 0% | - |
| its | 7 | under 50 | +0 | step | 39 | +17.78 | +54.46 | 92% | 8% | - |
| its | 14 | 50-150 | -25 | ramp | 62 | +48.30 | +90.39 | 82% | - | 16% |
| its | 14 | 50-150 | -25 | step | 61 | +28.02 | +73.25 | 92% | - | 20% |
| its | 14 | 50-150 | -10 | ramp | 70 | +64.08 | +230.33 | 89% | - | 11% |
| its | 14 | 50-150 | -10 | step | 62 | +42.08 | +103.01 | 89% | - | 10% |
| its | 14 | 50-150 | +0 | ramp | 83 | +54.41 | +165.23 | 88% | 12% | - |
| its | 14 | 50-150 | +0 | step | 61 | +25.75 | +75.00 | 97% | 3% | - |
| its | 14 | over 150 | -25 | ramp | 55 | +14.60 | +53.65 | 93% | - | 18% |
| its | 14 | over 150 | -25 | step | 74 | +20.38 | +62.81 | 92% | - | 20% |
| its | 14 | over 150 | -10 | ramp | 54 | +21.42 | +82.09 | 87% | - | 13% |
| its | 14 | over 150 | -10 | step | 61 | +27.01 | +79.22 | 95% | - | 15% |
| its | 14 | over 150 | +0 | ramp | 58 | +37.60 | +113.97 | 90% | 10% | - |
| its | 14 | over 150 | +0 | step | 61 | +49.99 | +104.34 | 89% | 11% | - |
| its | 14 | under 50 | -25 | ramp | 52 | +7.83 | +23.45 | 96% | - | 4% |
| its | 14 | under 50 | -25 | step | 36 | +13.18 | +33.25 | 89% | - | 17% |
| its | 14 | under 50 | -10 | ramp | 40 | +13.28 | +27.01 | 90% | - | 2% |
| its | 14 | under 50 | -10 | step | 44 | +11.14 | +27.79 | 98% | - | 0% |
| its | 14 | under 50 | +0 | ramp | 36 | +17.53 | +34.08 | 94% | 6% | - |
| its | 14 | under 50 | +0 | step | 38 | +17.79 | +48.00 | 97% | 3% | - |
| synth | 7 | 50-150 | -25 | ramp | 63 | +34.34 | +117.48 | 84% | - | 22% |
| synth | 7 | 50-150 | -25 | step | 64 | +20.12 | +80.40 | 94% | - | 38% |
| synth | 7 | 50-150 | -10 | ramp | 72 | +60.27 | +229.34 | 86% | - | 21% |
| synth | 7 | 50-150 | -10 | step | 64 | +49.71 | +159.00 | 84% | - | 20% |
| synth | 7 | 50-150 | +0 | ramp | 83 | +40.47 | +149.88 | 89% | 11% | - |
| synth | 7 | 50-150 | +0 | step | 63 | +15.90 | +76.99 | 90% | 10% | - |
| synth | 7 | over 150 | -25 | ramp | 65 | +16.72 | +68.73 | 89% | - | 29% |
| synth | 7 | over 150 | -25 | step | 90 | +26.30 | +94.61 | 86% | - | 49% |
| synth | 7 | over 150 | -10 | ramp | 66 | +9.33 | +45.80 | 95% | - | 9% |
| synth | 7 | over 150 | -10 | step | 77 | +35.01 | +133.95 | 90% | - | 16% |
| synth | 7 | over 150 | +0 | ramp | 67 | +26.52 | +92.93 | 91% | 9% | - |
| synth | 7 | over 150 | +0 | step | 76 | +43.98 | +122.42 | 89% | 11% | - |
| synth | 7 | under 50 | -25 | ramp | 52 | +7.27 | +23.93 | 87% | - | 6% |
| synth | 7 | under 50 | -25 | step | 36 | +7.44 | +38.42 | 94% | - | 36% |
| synth | 7 | under 50 | -10 | ramp | 41 | +4.03 | +18.47 | 98% | - | 0% |
| synth | 7 | under 50 | -10 | step | 45 | +7.93 | +31.12 | 89% | - | 9% |
| synth | 7 | under 50 | +0 | ramp | 36 | +9.09 | +25.57 | 97% | 3% | - |
| synth | 7 | under 50 | +0 | step | 39 | +17.62 | +54.48 | 90% | 10% | - |
| synth | 14 | 50-150 | -25 | ramp | 63 | +43.20 | +87.04 | 79% | - | 22% |
| synth | 14 | 50-150 | -25 | step | 63 | +24.59 | +70.54 | 95% | - | 35% |
| synth | 14 | 50-150 | -10 | ramp | 71 | +60.62 | +228.21 | 82% | - | 18% |
| synth | 14 | 50-150 | -10 | step | 62 | +40.64 | +102.52 | 89% | - | 15% |
| synth | 14 | 50-150 | +0 | ramp | 83 | +51.88 | +164.76 | 87% | 13% | - |
| synth | 14 | 50-150 | +0 | step | 63 | +23.53 | +73.18 | 95% | 5% | - |
| synth | 14 | over 150 | -25 | ramp | 65 | +8.38 | +47.29 | 95% | - | 29% |
| synth | 14 | over 150 | -25 | step | 87 | +14.41 | +52.07 | 85% | - | 45% |
| synth | 14 | over 150 | -10 | ramp | 66 | +15.66 | +73.95 | 88% | - | 17% |
| synth | 14 | over 150 | -10 | step | 74 | +21.19 | +70.59 | 97% | - | 15% |
| synth | 14 | over 150 | +0 | ramp | 65 | +33.64 | +107.07 | 85% | 15% | - |
| synth | 14 | over 150 | +0 | step | 68 | +42.21 | +98.64 | 84% | 16% | - |
| synth | 14 | under 50 | -25 | ramp | 52 | +7.08 | +22.87 | 88% | - | 8% |
| synth | 14 | under 50 | -25 | step | 36 | +9.55 | +31.38 | 89% | - | 31% |
| synth | 14 | under 50 | -10 | ramp | 41 | +7.72 | +24.75 | 95% | - | 5% |
| synth | 14 | under 50 | -10 | step | 45 | +7.05 | +26.42 | 91% | - | 9% |
| synth | 14 | under 50 | +0 | ramp | 36 | +14.03 | +33.71 | 100% | 0% | - |
| synth | 14 | under 50 | +0 | step | 39 | +17.47 | +47.35 | 92% | 8% | - |

## Placebo-in-space (comparison games at real patch dates, no injection)

| Method | Trials | False alarms |
|---|---|---|
| its | 1840 | 8% |
| synth | 2006 | 7% |

## Limitations

- The effect-0 trials on tracked games are the placebo-in-time runs: fake patch dates where none happened.
- Aspect shares among negative reviews use the same interrupted time series but aren't benchmarked separately; their counts are thinner, so their intervals are wider.
- Patch dates come from a rules classifier (rules-v1): a missed patch can leave a "quiet" window or a comparison game contaminated.
- Game-specific discounts are known only since 30 Sep 2026; quiet windows avoid Steam's seasonal sales but not earlier game discounts (the volume-spike table shows their weight).
- Injected effects are level shifts or 7-day ramps; effects that fade are not tested.
