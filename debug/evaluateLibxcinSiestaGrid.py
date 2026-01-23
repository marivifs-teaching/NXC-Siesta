#!/usr/bin/env python3
"""
evaluateLibxcinSiestaGrid.py
============================

Reference post-processing script to evaluate PBE exchange–correlation energies
on a real-space density grid using LibXC (via PySCF).

WHAT THIS SCRIPT DOES
---------------------
This script reads a real-space grid dumped from SIESTA with columns:

  rho, gx, gy, gz, w

where:
  rho        = electron density
  gx,gy,gz   = components of the density gradient
  w          = integration weight (dVol)

It then:

  • builds a LibXC-compatible GGA input array of shape (4, N):
      [rho, d rho/dx, d rho/dy, d rho/dz]
  • evaluates the PBE exchange–correlation energy density using
    LibXC through PySCF,
  • integrates the total XC energy as:

      E_xc = sum_i w_i * rho_i * eps_xc(rho_i, ∇rho_i)

WHAT IT IS USED FOR
-------------------
This script is used as a *reference and validation tool* to:

  • reproduce SIESTA-reported PBE Exc values from dumped grids,
  • verify that grid dumping and integration weights are correct,
  • provide a clean PBE baseline when comparing against NN-GGA
    evaluations on the same SIESTA grid.

Because LibXC is used directly, this script serves as an independent
check on SIESTA’s PBE implementation.

USAGE
-----
Run in a directory containing a SIESTA grid dump:

  python evaluateLibxcinSiestaGrid.py

The script currently expects the file:

  pbe_rho_grad_w.dat

and prints the integrated XC energy in Hartree and eV.

NOTES
-----
• This script performs no self-consistent calculations.
• It evaluates energies only (no potentials or derivatives).
• Gradient information is passed explicitly (xctype = GGA).

This script is intended for validation and analysis, not production use.
"""

import numpy as np
import jax.numpy as jnp  # only for convenience, not strictly needed
from pyscf.dft import libxc

# Load SIESTA grid dump: rho, gx, gy, gz, dVol
data = np.loadtxt("pbe_rho_grad_w.dat")
rho  = data[:, 0]
gx   = data[:, 1]
gy   = data[:, 2]
gz   = data[:, 3]
w    = data[:, 4]      # integration weights dVol

# Build LibXC-style GGA rho array: shape (4, N) = [rho, dx, dy, dz]
rho_gga = np.stack([rho, gx, gy, gz], axis=0)  # (4, N)

# Evaluate PBE XC with LibXC via PySCF
# xctype='GGA': LibXC expects gradient components, not sigma
exc, vxc, fxc, kxc = libxc.eval_xc('PBE', rho_gga, spin=0, deriv=0)

# exc is per-electron xc energy density (in Ha)
exc = np.asarray(exc)  # shape (N,)

# Integrated E_xc = sum_i w_i * rho_i * exc_i
Exctot = np.sum(w * rho * exc)
print("Python PBE E_xc [Ha]:", Exctot)
print("Python PBE E_xc [eV]:", Exctot * 27.211386245988)