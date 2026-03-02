#!/usr/bin/env python3
import argparse
import numpy as np
import jax.numpy as jnp
import test_xc_nn_jax_pyscf as t  # your notebook-consistent eval path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="nn_rho_grad_w.dat")
    ap.add_argument("--fx-model", required=True)
    ap.add_argument("--fc-model", required=True)
    ap.add_argument("--out", default="jax_grid.csv")
    args = ap.parse_args()

    model_fx = t.load_pretrained_model(args.fx_model)
    model_fc = t.load_pretrained_model(args.fc_model)
    my_eval = t.get_my_eval_xc_ft(model_fx=model_fx, model_fc=model_fc)

    data = np.loadtxt(args.grid, dtype=np.float64, comments="#")
    rho_u = data[:, 0]
    rho_d = data[:, 1]
    gxu, gyu, gzu = data[:, 2], data[:, 3], data[:, 4]
    gxd, gyd, gzd = data[:, 5], data[:, 6], data[:, 7]
    w = data[:, 8]

    rho_u_block = jnp.asarray(np.vstack([rho_u, gxu, gyu, gzu]))
    rho_d_block = jnp.asarray(np.vstack([rho_d, gxd, gyd, gzd]))
    rho_spin = (rho_u_block, rho_d_block)

    exc, vxc, _, _ = my_eval("GGA", rho_spin, spin=1, deriv=1)
    vrho, vsigma = vxc  # (N,2) and (N,3)

    exc = np.asarray(exc).reshape(-1)
    vrho = np.asarray(vrho)
    vsigma = np.asarray(vsigma)

    gu = np.stack([gxu, gyu, gzu], axis=1)
    gd = np.stack([gxd, gyd, gzd], axis=1)
    gtot = gu + gd
    grad_rho = np.linalg.norm(gtot, axis=1)

    rho = rho_u + rho_d
    edens = rho * exc

    out = np.column_stack([
        np.arange(1, len(rho) + 1),
        rho_u, rho_d,
        gxu, gyu, gzu,
        gxd, gyd, gzd,
        w,
        rho,
        grad_rho,
        exc,
        edens,
        vrho[:, 0], vrho[:, 1],
        vsigma[:, 0], vsigma[:, 1], vsigma[:, 2],
    ])

    np.savetxt(
        args.out,
        out,
        delimiter=",",
        header="i,rho_u,rho_d,gxu,gyu,gzu,gxd,gyd,gzd,w,rho,grad_rho,exc,edens,vrhou,vrhod,vsuu,vsud,vsdd",
        comments="",
    )

    print(f"Wrote {args.out}")
    print(f"JAX integrated E_xc = {np.sum(w * edens):.12f}")

if __name__ == "__main__":
    main()