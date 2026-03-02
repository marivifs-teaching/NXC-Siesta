#!/usr/bin/env python3
"""
compare_descriptor_distributions.py
=================================

Utility script to analyze and compare the distributions of density-based
NN input descriptors on real-space grids.

WHAT THIS SCRIPT DOES
---------------------
Given a real-space grid file with columns:

  rho, gx, gy, gz, w

(this is the standard format dumped from PySCF or SIESTA), the script:

  • optionally subsamples the grid for efficiency,
  • computes the same intermediate quantities and NN features used in the
    NN-GGA functional implementation:
      - |grad rho|
      - x0 = log10(rho^(1/3) + eps)
      - reduced gradient s = |grad rho| / (2 kF rho)
      - x1 = log10(1+s) * (1 - exp(-s^2))
  • prints min/max values and selected quantiles of each quantity.

WHAT IT IS USED FOR
-------------------
This script is a *diagnostic and analysis tool*. It is used to:

  • compare descriptor distributions between different codes
    (e.g. PySCF vs SIESTA),
  • compare descriptor distributions between different XC functionals
    (e.g. PBE density vs NN density),
  • detect extrapolation regimes where NN descriptors (especially x1 or s)
    extend beyond the range seen during training,
  • understand density-driven errors in self-consistent NN calculations.

In particular, it was used to demonstrate that SIESTA densities produce
systematically larger reduced gradients (and NN features x1) than PySCF
for the same physical systems, explaining large density-driven effects.

USAGE
-----
Example (sample 500k points for speed):

  python compare_descriptor_distributions.py \
      --grid pbe_rho_grad_w.dat \
      --sample 500000 \
      --seed 0

Use multiple grids to compare outputs side by side:

  python compare_descriptor_distributions.py --grid pyscf_pbe_rho_grad_w.dat
  python compare_descriptor_distributions.py --grid pyscf_nn_rho_grad_w.dat
  python compare_descriptor_distributions.py --grid pbe_rho_grad_w.dat
  python compare_descriptor_distributions.py --grid nn_rho_grad_w.dat

NOTES
-----
• This script does not evaluate energies or potentials.
• Integration weights are ignored; only descriptor values are analyzed.
• Vacuum points (rho ≈ 0) are included unless filtered externally; this is
  intentional to reveal how grids differ near low-density regions.

This script is intended for analysis and debugging, not for production runs.
"""

import argparse
import numpy as np

def load_grid_sample(path, max_rows=None, seed=0):
    # Fast-ish load: np.loadtxt is slow for huge files; if needed we can switch to np.memmap/binary later.
    data = np.loadtxt(path)
    if max_rows is not None and len(data) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(data), size=max_rows, replace=False)
        data = data[idx]
    rho = data[:,0]
    gx  = data[:,1]
    gy  = data[:,2]
    gz  = data[:,3]
    w   = data[:,4]
    return rho, gx, gy, gz, w

def compute_features(rho, gx, gy, gz, eps_rho=1e-14, eps_x0=1e-5):
    rho = np.asarray(rho)
    gx = np.asarray(gx); gy = np.asarray(gy); gz = np.asarray(gz)

    g = np.sqrt(gx*gx + gy*gy + gz*gz)

    # Guard to avoid divide-by-zero in s
    rho_safe = np.maximum(rho, eps_rho)

    # x0 = log10(rho^(1/3) + eps_x0)  (same as your NN path)
    x0 = np.log10(rho_safe**(1/3) + eps_x0)

    # reduced gradient s = g / (2*kF*rho), kF=(3*pi^2*rho)^(1/3)
    pi = np.pi
    kF = (3.0*pi*pi*rho_safe)**(1/3)
    s  = g / (2.0*kF*rho_safe)

    # your x1 transform
    x1 = np.log10(1.0 + s) * (1.0 - np.exp(-s*s))

    return g, x0, s, x1

def summarize(name, arr):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    arr = arr[finite]
    qs = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0]
    qv = np.quantile(arr, qs)
    print(f"\n{name}:")
    print(f"  n (finite) = {arr.size}")
    print(f"  min/max    = {arr.min():.6e}  {arr.max():.6e}")
    for q, v in zip(qs, qv):
        print(f"  q={q:>7g}: {v:.6e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="grid file: rho gx gy gz w")
    ap.add_argument("--sample", type=int, default=500000, help="random sample size (None for full)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rho, gx, gy, gz, w = load_grid_sample(args.grid, max_rows=args.sample, seed=args.seed)
    g, x0, s, x1 = compute_features(rho, gx, gy, gz)

    print(f"\n=== {args.grid} ===")
    summarize("rho", rho)
    summarize("|grad rho|", g)
    summarize("x0", x0)
    summarize("s", s)
    summarize("x1", x1)

if __name__ == "__main__":
    main()