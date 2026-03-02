#!/usr/bin/env python3
"""
xc_on_grid.py
===============

Post-processing utility to evaluate exchange–correlation (XC) energies on a
precomputed real-space grid of densities and gradients.

WHAT THIS SCRIPT DOES
---------------------
Given a grid file with columns:

  rho, gx, gy, gz, w

(where rho is the electron density, (gx,gy,gz) its gradient components, and w
is the integration weight), this script:

  • evaluates either PBE (via LibXC) or a neural-network GGA functional
    (Fx/Fc models trained in PySCF/JAX),
  • computes the per-electron XC energy density eps_xc(r),
  • integrates the total XC energy:

        E_xc = sum_i w_i * rho_i * eps_xc(r_i)

This is a *post-processing* tool: it does NOT perform SCF and does not modify
any electronic-structure code.

WHAT IT IS USED FOR
-------------------
This script was used to:

  • compare PBE vs NN functionals on the *same density* (functional-driven error),
  • compare PBE vs NN self-consistent densities (density-driven error),
  • validate NN-GGA implementations between PySCF and SIESTA,
  • diagnose NaN/Inf issues arising from vacuum grid points,
  • reproduce SIESTA-reported Exc values from dumped grids.

It enables a clean 2×2 decomposition:

  PBE[rho_PBE],  NN[rho_PBE]
  PBE[rho_NN ],  NN[rho_NN ]

USAGE
-----
Evaluate PBE on a grid:

  python xc_on_grid.py --grid pbe_rho_grad_w.dat --xc PBE

Evaluate NN on a grid (recommended LDA reference = LibXC):

  python xc_on_grid.py --grid nn_rho_grad_w.dat \
                       --xc NN \
                       --model-dir ../pretrained_models \
                       --lda-ref libxc

Optional diagnostics:

  --nan-report N     Print up to N grid points where NaN/Inf appears
  --nan-save FILE   Save all problematic points to CSV for inspection

NOTES
-----
• For NN evaluation, vacuum points (rho ≈ 0) are handled explicitly to avoid
  ill-defined reduced-gradient descriptors; these points contribute zero to
  the XC integral.

• LDA references for NN evaluation are taken from LibXC (LDA_X and LDA_C_PW),
  matching the references used internally by SIESTA.

• This script is intended for validation and analysis, not production runs.
"""

import os
import sys
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pyscf.dft import libxc

# Allow importing from project root (parent of SiestaPlugInn)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.models import GGA_Fx_G_transf_lin, GGA_Fc_G_transf_lin, load_eqx_model

HARTREE_TO_EV = 27.211386245988


def load_grid(path: str):
    data = np.loadtxt(path, comments="#")
    if data.shape[1] < 5:
        raise ValueError(f"Grid file must have 5 columns (rho gx gy gz w). Got shape {data.shape}")
    rho = data[:, 0]
    gx = data[:, 1]
    gy = data[:, 2]
    gz = data[:, 3]
    w = data[:, 4]
    return rho, gx, gy, gz, w


def eval_pbe_exc(rho, gx, gy, gz):
    # PySCF/libxc expects GGA rho array shape (4, N): [rho, dx, dy, dz]
    rho_gga = np.stack([rho, gx, gy, gz], axis=0)
    out = libxc.eval_xc("PBE", rho_gga, spin=0, deriv=0)
    exc = out[0] if isinstance(out, tuple) else out
    return np.asarray(exc).ravel()


def load_nn_models(model_dir="../pretrained_models", depth=3, nodes=16, seed=42, version_tag="v_20000"):
    jax.config.update("jax_enable_x64", True)

    name_fx = f"{GGA_Fx_G_transf_lin.name}_d{depth}_n{nodes}_s{seed}_{version_tag}"
    name_fc = f"{GGA_Fc_G_transf_lin.name}_d{depth}_n{nodes}_s{seed}_{version_tag}"

    fx_path = os.path.join(model_dir, name_fx)
    fc_path = os.path.join(model_dir, name_fc)

    model_fx = load_eqx_model(fx_path)
    model_fc = load_eqx_model(fc_path)
    return model_fx, model_fc


def eval_nn_exc(rho, gx, gy, gz, model_fx, model_fc, lda_ref="builtin", return_debug=False):
    """
    Evaluate your NN-GGA per-electron exc on a grid.
    Uses your model definitions which take inputs [rho, |grad rho|].

    lda_ref:
      - "builtin": use simple analytic LDA exchange and your PW92-unpolarized function (not implemented here)
      - "libxc": use LDA_X and LDA_C_PW from LibXC for references (recommended for SIESTA-grid consistency)
    """
    g = np.sqrt(gx * gx + gy * gy + gz * gz)

    # Guard against rho=0 (or extremely small) with nonzero gradient from finite-difference noise.
    # Such points contribute nothing to the integral (w*rho*exc = 0) but can produce NaNs in the NN descriptors.
    rho_cut = 1e-14
    mask = rho <= rho_cut
    # Force gradient magnitude to zero on masked points to keep descriptors finite.
    g = np.where(mask, 0.0, g)

    # JAX inputs
    rho_j = jnp.asarray(rho)
    g_j = jnp.asarray(g)
    inputs = jnp.stack([rho_j, g_j], axis=1)  # (N,2)

    Fx = jax.vmap(model_fx)(inputs)
    Fc = jax.vmap(model_fc)(inputs)
    Fx = np.asarray(Fx).ravel()
    Fc = np.asarray(Fc).ravel()

    if lda_ref.lower() == "libxc":
        rho_lda = rho[None, :]  # (1,N)
        epsx = libxc.eval_xc("LDA_X", rho_lda, spin=0, deriv=0)[0]
        epsc = libxc.eval_xc("LDA_C_PW", rho_lda, spin=0, deriv=0)[0]
        epsx = np.asarray(epsx).ravel()
        epsc = np.asarray(epsc).ravel()
    else:
        raise ValueError("For NN evaluation, please use --lda-ref libxc (recommended).")

    exc = Fx * epsx + Fc * epsc

    # Ensure masked points contribute exactly zero and do not propagate NaNs.
    exc = np.where(mask, 0.0, exc)

    integrand = rho * exc
    if return_debug:
        return exc, {
            "g": g,
            "Fx": Fx,
            "Fc": Fc,
            "epsx": epsx,
            "epsc": epsc,
            "integrand": integrand,
            "mask": mask,
        }
    return exc


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate XC energy on a rho/grad/weight grid file for PBE or NN."
    )
    parser.add_argument("--grid", required=True, help="Path to grid file with columns: rho gx gy gz weight")
    parser.add_argument("--xc", required=True, choices=["PBE", "NN"], help="XC to evaluate on the grid")
    parser.add_argument("--model-dir", default="../pretrained_models", help="Directory containing NN .eqx/.json")
    parser.add_argument("--lda-ref", default="libxc", choices=["libxc"], help="LDA reference for NN")
    parser.add_argument("--version-tag", default="v_20000", help="Model version tag used in filenames (default v_20000)")
    parser.add_argument("--nan-report", type=int, default=0,
                        help="If >0, print up to this many indices where NN exc or integrand is NaN/Inf.")
    parser.add_argument("--nan-save", default="",
                        help="Optional path to save all bad points as a CSV (index,rho,gx,gy,gz,w,g,Fx,Fc,epsx,epsc,exc,integrand).")
    args = parser.parse_args()

    rho, gx, gy, gz, w = load_grid(args.grid)

    if args.xc.upper() == "PBE":
        exc = eval_pbe_exc(rho, gx, gy, gz)
        dbg = None
    else:
        model_fx, model_fc = load_nn_models(model_dir=args.model_dir, version_tag=args.version_tag)
        want_debug = (args.nan_report > 0) or (args.nan_save != "")
        if want_debug:
            exc, dbg = eval_nn_exc(rho, gx, gy, gz, model_fx, model_fc, lda_ref=args.lda_ref, return_debug=True)
        else:
            exc = eval_nn_exc(rho, gx, gy, gz, model_fx, model_fc, lda_ref=args.lda_ref)
            dbg = None

    integrand_all = w * rho * exc
    Exc = np.nansum(integrand_all)

    if dbg is not None:
        g = dbg["g"]
        Fx = dbg["Fx"]
        Fc = dbg["Fc"]
        epsx = dbg["epsx"]
        epsc = dbg["epsc"]
        integrand = dbg["integrand"]
        mask = dbg.get("mask", None)

        bad = (~np.isfinite(exc)) | (~np.isfinite(integrand)) | (~np.isfinite(rho)) | (~np.isfinite(w))
        if mask is not None:
            bad = bad & (~mask)
        nbad = int(bad.sum())
        if nbad > 0:
            print(f"WARNING: Found {nbad} bad points (NaN/Inf) out of {len(rho)}")
            idx = np.where(bad)[0]
            nshow = min(args.nan_report, len(idx))
            if args.nan_report > 0:
                print("index        rho            g           Fx           Fc         epsx         epsc          exc     rho*exc")
                for ii in idx[:nshow]:
                    print(f"{ii:6d} {rho[ii]:12.5e} {g[ii]:12.5e} {Fx[ii]:12.5e} {Fc[ii]:12.5e} {epsx[ii]:12.5e} {epsc[ii]:12.5e} {exc[ii]:12.5e} {integrand[ii]:12.5e}")

            if args.nan_save:
                out = np.column_stack([
                    idx,
                    rho[idx], gx[idx], gy[idx], gz[idx], w[idx],
                    g[idx], Fx[idx], Fc[idx], epsx[idx], epsc[idx],
                    exc[idx], integrand[idx]
                ])
                header = "index,rho,gx,gy,gz,w,g,Fx,Fc,epsx,epsc,exc,integrand"
                np.savetxt(args.nan_save, out, delimiter=",", header=header, comments="")
                print(f"Saved bad points to {args.nan_save}")
        else:
            print("No NaN/Inf points found in NN exc/integrand.")

    print(f"Grid: {args.grid}")
    print(f"XC:   {args.xc}")
    print(f"E_xc [Ha]: {Exc:.15f}")
    print(f"E_xc [eV]: {Exc * HARTREE_TO_EV:.12f}")


if __name__ == "__main__":
    main()