#!/usr/bin/env python3
"""
Export Equinox MLP weights for GGA_Fx_G_transf_lin and GGA_Fc_G_transf_lin
to plain-text .dat files for use in Fortran.

Usage example:

    python export_eqx_to_fortran.py \
        --fx-model pretrained_models/GGA_Fx_G_transf_lin_d3_n16_s42_v_20000 \
        --fc-model pretrained_models/GGA_Fc_G_transf_lin_d3_n16_s42_v_20000 \
        --out-prefix xc_nn

This will create files like:
  xc_nn_fx_W1.dat, xc_nn_fx_b1.dat, ..., xc_nn_fc_W4.dat, xc_nn_fc_b4.dat
"""

import argparse
import numpy as np
import jax
import equinox as eqx

from nnxc_tools.models import load_eqx_model

def extract_mlp_params(mlp):
    """Return lists of (W, b) for each Linear layer in the Equinox MLP.

    Each W has shape (out_features, in_features), b has shape (out_features,).
    """
    weights = []
    biases = []
    # Equinox MLP stores Linear layers in `layers`
    for layer in mlp.layers:
        W = np.asarray(layer.weight)
        b = np.asarray(layer.bias)
        weights.append(W)
        biases.append(b)
    return weights, biases


def save_layer(prefix, idx, W, b):
    """Save a single layer's weights and biases to text files.

    Files:
      {prefix}_W{idx}.dat
      {prefix}_b{idx}.dat
    """
    w_file = f"{prefix}_W{idx}.dat"
    b_file = f"{prefix}_b{idx}.dat"
    np.savetxt(w_file, W)
    np.savetxt(b_file, b)
    print(f"  saved {w_file} (shape {W.shape}), {b_file} (shape {b.shape})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fx-model", required=True,
                        help="Path *without* .eqx/.json to Fx model "
                             "(e.g. pretrained_models/GGA_Fx_G_transf_lin_d3_n16_s42_v_20000)")
    parser.add_argument("--fc-model", required=True,
                        help="Path *without* .eqx/.json to Fc model "
                             "(e.g. pretrained_models/GGA_Fc_G_transf_lin_d3_n16_s42_v_20000)")
    parser.add_argument("--out-prefix", default="xc_nn",
                        help="Prefix for output files (default: xc_nn)")
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)

    print("Loading Fx model from", args.fx_model)
    fx_model = load_eqx_model(args.fx_model)
    fx_mlp = fx_model.net  # GGA_Fx_G_transf_lin.net is eqx.nn.MLP

    print("Loading Fc model from", args.fc_model)
    fc_model = load_eqx_model(args.fc_model)
    fc_mlp = fc_model.net  # GGA_Fc_G_transf_lin.net is eqx.nn.MLP

    print("\nExtracting Fx MLP parameters...")
    fx_weights, fx_biases = extract_mlp_params(fx_mlp)
    print(f"  Fx MLP: {len(fx_weights)} layers")

    print("Extracting Fc MLP parameters...")
    fc_weights, fc_biases = extract_mlp_params(fc_mlp)
    print(f"  Fc MLP: {len(fc_weights)} layers")

    # Sanity: For your current architecture, we expect 4 Linear layers for each model
    if len(fx_weights) != 4 or len(fc_weights) != 4:
        raise RuntimeError(
            f"Expected 4 layers for each MLP (depth=3), "
            f"got Fx={len(fx_weights)}, Fc={len(fc_weights)}. "
            "Update Fortran shapes or check the model definitions."
        )

    # Save Fx layers
    print("\nSaving Fx layers...")
    for i, (W, b) in enumerate(zip(fx_weights, fx_biases), start=1):
        save_layer(f"{args.out_prefix}_fx", i, W, b)

    # Save Fc layers
    print("\nSaving Fc layers...")
    for i, (W, b) in enumerate(zip(fc_weights, fc_biases), start=1):
        save_layer(f"{args.out_prefix}_fc", i, W, b)

    print("\nDone. You can now point the Fortran loader at prefix =", args.out_prefix)


if __name__ == "__main__":
    main()
