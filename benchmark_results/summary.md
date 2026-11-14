# Benchmark results

Runs executed on 2026-11-08 on one machine, sequentially: the original suite ran 14:30-16:22 (+05:30). After the mode-D solver was improved (continue from best weights, gentle continuation, R3 point density/budget), all 27 D runs and the oscillator A-C runs (whose point budget changed) were re-run 20:45-21:20; all other rows are from the original run and are unaffected by that change. Previous results are kept in results_v1.csv.
Hardware/software: macOS-26.0-arm64-arm-64bit, arm64; Python 3.10.11, PyTorch 2.14.1, NumPy 2.2.6, SciPy 1.15.3, SymPy 1.14.0; 4 torch threads, float64 on CPU.
Seeds: [0, 1, 2]. Equal wall-clock budget per problem for every configuration (decay: 30 s, logistic: 30 s, oscillator: 120 s, bvp: 30 s, bvp_neumann: 30 s, poisson: 60 s, heat: 120 s, stiff: 30 s, burgers: 240 s) and equal collocation-point budget (A/B use it uniformly from the start; C/D grow towards it adaptively).
Thresholds were fixed before the runs: relative L2 <= 1e-3 (ODEs/BVPs) and <= 1e-2 (PDEs); IC/BC error <= 1e-4 (ODE/BVP) and <= 1e-3 (PDE).
Error metrics are for the PINN solution (in D, the best attempt) even when D then displayed the numerical fallback; 'fallback used' counts those runs.
For E the rows report the numerical solver compared with the exact solution where one exists (no PINN is trained).

| problem | mode | runs | PINN met threshold | fallback used | rel. L2 error (mean ± sd) | RMS residual | training time s (mean ± sd) | points | attempts |
|---|---|---|---|---|---|---|---|---|---|
| decay | A | 3 | 3/3 | 0/3 | 2.62e-04 ± 1.4e-04 | 6.38e-04 ± 4.6e-05 | 2.54e+01 ± 1.5e+00 | 512 | 1.00e+00 ± 0.0e+00 |
| decay | B | 3 | 3/3 | 0/3 | 3.24e-04 ± 2.2e-04 | 1.49e-03 ± 8.9e-04 | 2.05e+01 ± 2.7e-01 | 512 | 1.00e+00 ± 0.0e+00 |
| decay | C | 3 | 3/3 | 0/3 | 3.50e-04 ± 2.4e-04 | 1.01e-03 ± 6.3e-04 | 1.66e+01 ± 1.7e+00 | 512 | 1.00e+00 ± 0.0e+00 |
| decay | D | 3 | 3/3 | 0/3 | 9.66e-05 ± 5.0e-05 | 3.63e-04 ± 1.9e-04 | 4.89e+00 ± 3.0e-01 | 128 | 1.00e+00 ± 0.0e+00 |
| decay | E | 3 | n/a (no PINN) | 0/3 | 5.50e-11 ± 0.0e+00 | n/a | 1.00e-02 ± 1.7e-03 | 0 | 0.00e+00 ± 0.0e+00 |
| logistic | A | 3 | 0/3 | 0/3 | 1.00e+00 ± 3.1e-03 | 3.80e-02 ± 1.0e-03 | 1.98e+01 ± 3.2e-01 | 512 | 1.00e+00 ± 0.0e+00 |
| logistic | B | 3 | 1/3 | 0/3 | 6.64e-01 ± 5.7e-01 | 2.95e-02 ± 2.5e-02 | 2.05e+01 ± 2.5e+00 | 512 | 1.00e+00 ± 0.0e+00 |
| logistic | C | 3 | 1/3 | 0/3 | 6.64e-01 ± 5.7e-01 | 6.67e-02 ± 6.6e-02 | 1.84e+01 ± 1.9e+00 | 512 | 1.00e+00 ± 0.0e+00 |
| logistic | D | 3 | 3/3 | 0/3 | 5.04e-04 ± 3.5e-04 | 4.54e-04 ± 2.4e-04 | 9.74e+00 ± 5.0e-01 | 192 | 3.00e+00 ± 0.0e+00 |
| logistic | E | 3 | n/a (no PINN) | 0/3 | 3.33e-11 ± 0.0e+00 | n/a | 3.27e-02 ± 2.9e-03 | 0 | 0.00e+00 ± 0.0e+00 |
| oscillator | A | 3 | 0/3 | 0/3 | 6.97e-03 ± 2.1e-03 | 1.21e-02 ± 2.4e-03 | 1.20e+02 ± 7.4e-03 | 2048 | 1.00e+00 ± 0.0e+00 |
| oscillator | B | 3 | 0/3 | 0/3 | 8.39e-01 ± 8.1e-02 | 6.64e-01 ± 5.2e-02 | 1.20e+02 ± 1.2e-02 | 2048 | 1.00e+00 ± 0.0e+00 |
| oscillator | C | 3 | 0/3 | 0/3 | 6.22e-01 ± 1.0e-01 | 3.57e-01 ± 4.9e-02 | 1.20e+02 ± 1.5e-03 | 2048 | 1.00e+00 ± 0.0e+00 |
| oscillator | D | 3 | 2/3 | 1/3 | 9.87e-04 ± 9.1e-04 | 6.03e-03 ± 2.2e-03 | 8.32e+01 ± 3.4e+01 | 1024, 1280 | 4.00e+00 ± 1.0e+00 |
| oscillator | E | 3 | n/a (no PINN) | 0/3 | 1.06e-10 ± 0.0e+00 | n/a | 4.43e-02 ± 7.6e-03 | 0 | 0.00e+00 ± 0.0e+00 |
| bvp | A | 3 | 1/3 | 0/3 | 1.09e-04 ± 8.7e-05 | 1.06e-03 ± 5.1e-04 | 3.00e+01 ± 1.5e-03 | 512 | 1.00e+00 ± 0.0e+00 |
| bvp | B | 3 | 3/3 | 0/3 | 6.68e-06 ± 6.1e-06 | 5.20e-04 ± 2.9e-04 | 3.00e+01 ± 2.6e-03 | 512 | 1.00e+00 ± 0.0e+00 |
| bvp | C | 3 | 3/3 | 0/3 | 4.96e-06 ± 2.8e-06 | 5.01e-04 ± 2.1e-04 | 2.91e+01 ± 1.5e+00 | 512 | 1.00e+00 ± 0.0e+00 |
| bvp | D | 3 | 3/3 | 0/3 | 3.85e-06 ± 2.7e-06 | 3.65e-04 ± 1.6e-04 | 8.06e+00 ± 4.9e-01 | 128 | 1.00e+00 ± 0.0e+00 |
| bvp | E | 3 | n/a (no PINN) | 0/3 | 9.58e-13 ± 0.0e+00 | n/a | 6.33e-03 ± 7.5e-03 | 0 | 0.00e+00 ± 0.0e+00 |
| bvp_neumann | A | 3 | 2/3 | 0/3 | 5.05e-05 ± 7.8e-05 | 1.34e-03 ± 4.6e-04 | 3.00e+01 ± 2.9e-03 | 512 | 1.00e+00 ± 0.0e+00 |
| bvp_neumann | B | 3 | 3/3 | 0/3 | 5.08e-06 ± 3.2e-06 | 1.38e-03 ± 5.8e-04 | 3.00e+01 ± 9.0e-03 | 512 | 1.00e+00 ± 0.0e+00 |
| bvp_neumann | C | 3 | 2/3 | 0/3 | 9.90e-05 ± 1.4e-04 | 9.56e-04 ± 4.7e-05 | 3.00e+01 ± 1.0e-03 | 512 | 1.00e+00 ± 0.0e+00 |
| bvp_neumann | D | 3 | 3/3 | 0/3 | 9.98e-07 ± 5.0e-07 | 3.55e-04 ± 1.4e-04 | 8.83e+00 ± 1.6e+00 | 128 | 1.00e+00 ± 0.0e+00 |
| bvp_neumann | E | 3 | n/a (no PINN) | 0/3 | 9.37e-13 ± 0.0e+00 | n/a | 6.00e-03 ± 6.9e-03 | 0 | 0.00e+00 ± 0.0e+00 |
| poisson | A | 3 | 0/3 | 0/3 | 4.05e-04 ± 1.6e-04 | 3.61e-02 ± 4.0e-03 | 3.13e+01 ± 2.1e-01 | 512 | 1.00e+00 ± 0.0e+00 |
| poisson | B | 3 | 3/3 | 0/3 | 2.27e-05 ± 2.3e-05 | 3.30e-02 ± 1.4e-02 | 3.07e+01 ± 6.1e-01 | 512 | 1.00e+00 ± 0.0e+00 |
| poisson | C | 3 | 3/3 | 0/3 | 4.96e-05 ± 4.8e-05 | 3.04e-02 ± 4.6e-03 | 2.92e+01 ± 3.5e-01 | 512 | 1.00e+00 ± 0.0e+00 |
| poisson | D | 3 | 3/3 | 0/3 | 2.68e-05 ± 2.2e-05 | 1.49e-02 ± 6.6e-03 | 8.12e+00 ± 7.5e-02 | 128 | 1.00e+00 ± 0.0e+00 |
| poisson | E | 3 | n/a (no PINN) | 0/3 | 2.11e-12 ± 0.0e+00 | n/a | 4.33e-02 ± 1.0e-02 | 0 | 0.00e+00 ± 0.0e+00 |
| heat | A | 3 | 1/3 | 0/3 | 4.66e-04 ± 2.8e-04 | 2.66e-03 ± 1.3e-03 | 1.20e+02 ± 1.5e-02 | 4096 | 1.00e+00 ± 0.0e+00 |
| heat | B | 3 | 3/3 | 0/3 | 1.12e-03 ± 1.7e-03 | 4.16e-03 ± 5.5e-03 | 1.20e+02 ± 7.5e-03 | 4096 | 1.00e+00 ± 0.0e+00 |
| heat | C | 3 | 3/3 | 0/3 | 6.96e-05 ± 1.3e-05 | 6.09e-04 ± 2.1e-04 | 1.20e+02 ± 1.0e-02 | 4096 | 1.00e+00 ± 0.0e+00 |
| heat | D | 3 | 3/3 | 0/3 | 1.60e-05 ± 6.1e-06 | 2.36e-04 ± 8.2e-05 | 6.01e+01 ± 1.6e+00 | 2048 | 1.00e+00 ± 0.0e+00 |
| heat | E | 3 | n/a (no PINN) | 0/3 | 1.25e-06 ± 0.0e+00 | n/a | 4.93e-02 ± 9.2e-03 | 0 | 0.00e+00 ± 0.0e+00 |
| stiff | A | 3 | 0/3 | 0/3 | 1.15e-01 ± 1.2e-04 | 1.17e-01 ± 2.4e-03 | 2.51e+01 ± 5.0e-01 | 1024 | 1.00e+00 ± 0.0e+00 |
| stiff | B | 3 | 0/3 | 0/3 | 4.06e-02 ± 5.6e-03 | 2.29e+00 ± 3.1e-01 | 2.34e+01 ± 1.8e-01 | 1024 | 1.00e+00 ± 0.0e+00 |
| stiff | C | 3 | 0/3 | 0/3 | 3.21e-03 ± 1.3e-03 | 3.14e-01 ± 1.1e-01 | 2.13e+01 ± 2.7e-01 | 1024 | 1.00e+00 ± 0.0e+00 |
| stiff | D | 3 | 3/3 | 0/3 | 1.83e-04 ± 1.3e-04 | 7.27e-03 ± 5.3e-03 | 5.97e+00 ± 3.5e-02 | 512 | 1.00e+00 ± 0.0e+00 |
| stiff | E | 3 | n/a (no PINN) | 0/3 | 5.77e-12 ± 0.0e+00 | n/a | 3.05e-01 ± 1.1e-02 | 0 | 0.00e+00 ± 0.0e+00 |
| burgers | A | 3 | 0/3 | 0/3 | 5.50e-02 ± 4.1e-02 | 2.84e-02 ± 5.9e-04 | 2.40e+02 ± 3.4e-02 | 8192 | 1.00e+00 ± 0.0e+00 |
| burgers | B | 3 | 2/3 | 0/3 | 8.51e-03 ± 1.4e-03 | 3.14e-02 ± 8.6e-03 | 2.40e+02 ± 1.0e-02 | 8192 | 1.00e+00 ± 0.0e+00 |
| burgers | C | 3 | 1/3 | 0/3 | 1.45e-02 ± 7.6e-03 | 2.11e-02 ± 1.5e-03 | 2.40e+02 ± 2.6e-03 | 5632 | 1.00e+00 ± 0.0e+00 |
| burgers | D | 3 | 3/3 | 0/3 | 2.85e-03 ± 1.2e-03 | 6.83e-03 ± 1.6e-03 | 1.45e+02 ± 3.1e-01 | 4608 | 1.00e+00 ± 0.0e+00 |
| burgers | E | 3 | n/a (no PINN) | 0/3 | n/a | n/a | 3.38e-01 ± 2.5e-03 | 0 | 0.00e+00 ± 0.0e+00 |

### Runs in which the PINN did not meet its thresholds (41)

A run passes only if BOTH the relative L2 error and the maximum IC/BC error are within their limits; the reason column says which limit was missed.

| problem | mode | seed | status | final method | rel. L2 (limit) | max IC/BC error (limit) | reason |
|---|---|---|---|---|---|---|---|
| logistic | A | 0 | failed | pinn | 1.00e+00 (1e-03) | 1.4e-02 (1e-04) | relative L2 error and IC/BC error |
| logistic | A | 1 | failed | pinn | 1.00e+00 (1e-03) | 1.6e-02 (1e-04) | relative L2 error and IC/BC error |
| logistic | A | 2 | failed | pinn | 9.97e-01 (1e-03) | 1.6e-02 (1e-04) | relative L2 error and IC/BC error |
| logistic | B | 1 | failed | pinn | 9.95e-01 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| logistic | B | 2 | failed | pinn | 9.96e-01 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| logistic | C | 1 | failed | pinn | 9.99e-01 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| logistic | C | 2 | failed | pinn | 9.92e-01 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| oscillator | A | 0 | failed | pinn | 8.94e-03 (1e-03) | 3.6e-03 (1e-04) | relative L2 error and IC/BC error |
| oscillator | A | 1 | failed | pinn | 7.26e-03 (1e-03) | 1.1e-03 (1e-04) | relative L2 error and IC/BC error |
| oscillator | A | 2 | failed | pinn | 4.72e-03 (1e-03) | 2.1e-03 (1e-04) | relative L2 error and IC/BC error |
| oscillator | B | 0 | failed | pinn | 9.20e-01 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| oscillator | B | 1 | failed | pinn | 8.39e-01 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| oscillator | B | 2 | failed | pinn | 7.58e-01 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| oscillator | C | 0 | failed | pinn | 7.27e-01 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| oscillator | C | 1 | failed | pinn | 6.12e-01 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| oscillator | C | 2 | failed | pinn | 5.26e-01 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| oscillator | D | 2 | fallback | numerical_fallback | 2.03e-03 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| bvp | A | 0 | failed | pinn | 1.66e-04 (1e-03) | 1.9e-04 (1e-04) | IC/BC error |
| bvp | A | 2 | failed | pinn | 1.52e-04 (1e-03) | 1.9e-04 (1e-04) | IC/BC error |
| bvp_neumann | A | 0 | failed | pinn | 1.40e-04 (1e-03) | 2.2e-04 (1e-04) | IC/BC error |
| bvp_neumann | C | 1 | failed | pinn | 2.61e-04 (1e-03) | 4.0e-04 (1e-04) | IC/BC error |
| poisson | A | 0 | failed | pinn | 3.66e-04 (1e-03) | 3.7e-04 (1e-04) | IC/BC error |
| poisson | A | 1 | failed | pinn | 2.65e-04 (1e-03) | 3.4e-04 (1e-04) | IC/BC error |
| poisson | A | 2 | failed | pinn | 5.83e-04 (1e-03) | 7.2e-04 (1e-04) | IC/BC error |
| heat | A | 0 | failed | pinn | 4.69e-04 (1e-02) | 2.3e-03 (1e-03) | IC/BC error |
| heat | A | 2 | failed | pinn | 7.47e-04 (1e-02) | 1.5e-03 (1e-03) | IC/BC error |
| stiff | A | 0 | failed | pinn | 1.15e-01 (1e-03) | 9.9e-01 (1e-04) | relative L2 error and IC/BC error |
| stiff | A | 1 | failed | pinn | 1.15e-01 (1e-03) | 9.9e-01 (1e-04) | relative L2 error and IC/BC error |
| stiff | A | 2 | failed | pinn | 1.15e-01 (1e-03) | 9.9e-01 (1e-04) | relative L2 error and IC/BC error |
| stiff | B | 0 | failed | pinn | 3.47e-02 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| stiff | B | 1 | failed | pinn | 4.58e-02 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| stiff | B | 2 | failed | pinn | 4.14e-02 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| stiff | C | 0 | failed | pinn | 1.73e-03 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| stiff | C | 1 | failed | pinn | 4.22e-03 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| stiff | C | 2 | failed | pinn | 3.68e-03 (1e-03) | 0.0e+00 (1e-04) | relative L2 error |
| burgers | A | 0 | failed | pinn | 3.55e-02 (1e-02) | 5.7e-02 (1e-03) | relative L2 error and IC/BC error |
| burgers | A | 1 | failed | pinn | 2.71e-02 (1e-02) | 5.6e-02 (1e-03) | relative L2 error and IC/BC error |
| burgers | A | 2 | failed | pinn | 1.02e-01 (1e-02) | 5.9e-02 (1e-03) | relative L2 error and IC/BC error |
| burgers | B | 1 | failed | pinn | 1.01e-02 (1e-02) | 1.2e-16 (1e-03) | relative L2 error |
| burgers | C | 1 | failed | pinn | 1.24e-02 (1e-02) | 1.2e-16 (1e-03) | relative L2 error |
| burgers | C | 2 | failed | pinn | 2.29e-02 (1e-02) | 1.2e-16 (1e-03) | relative L2 error |
