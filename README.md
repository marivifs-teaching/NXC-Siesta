# NXC – Neural-Network XC for SIESTA

This directory contains the neural-network GGA exchange–correlation (NNXC) functional for SIESTA, validated against a JAX/PySCF reference implementation.

- `models/` – JAX/Equinox trained models (.eqx, JSON).
- `weights/` – Exported ASCII parameters used by SIESTA.
- `validation/` – Fortran and Python tests for energies and first derivatives.
- `grids/` – Reference density/gradient grids used in validation.
- `legacy/` – Archived unpolarized tools and earlier versions.

The functional is validated to machine precision for NN outputs and ~1e-8 for first analytic derivatives.

SIESTA computes the divergence term numerically on the real-space grid; validation focuses on energy density and first derivatives.

See `validation/docs/NNXC_validation_report.md` for full technical details.


# Neural-Network GGA XC (NNXC)

## Usage & Validation Guide

This directory documents and supports the **NN-based GGA XC functional (NNXC)** used in SIESTA and validated against the JAX/PySCF reference implementation.

This README explains:

1. How to export NN parameters from JAX/PySCF to SIESTA
2. How to activate NNXC in a SIESTA input file
3. How to validate energies and first derivatives
4. Important numerical and conceptual notes

---

## 1. Exporting NN Parameters from PySCF/JAX

### Prerequisites

- Trained or pre-trained models saved as:

```
GGA_Fx_*.eqx
GGA_Fc_*.eqx
```

- You are inside the JAX/PySCF environment.

### Export Command

Use the export utility to generate ASCII parameter files compatible with SIESTA:

```bash
python export_nnxc_to_fortran.py \
    --fx-model model1/GGA_Fx_xcdiff_d3_n16_s42_v_20000 \
    --fc-model model1/GGA_Fc_xcdiff_d3_n16_s42_v_20000 \
    --prefix nn_params/xc_nn_spin
```

This creates files such as:

```
nn_params/xc_nn_spin_fx_W1.dat
nn_params/xc_nn_spin_fx_b1.dat
nn_params/xc_nn_spin_fc_W1.dat
nn_params/xc_nn_spin_fc_b1.dat
...
```

The prefix must match what is used in the SIESTA input file.

### Important Implementation Detail

The Fortran code uses **tanh-approximate GELU** to match JAX 0.7.x default behavior:

```
jax.nn.gelu(x, approximate=True)
```

---

## 2. Activating NNXC in a SIESTA `.fdf` Input File

Include the following block in your SIESTA input:

```text
# ---------- XC functional ----------
XC.Functional      GGA
XC.Authors         NNGGA

NNXC.Prefix        nn_params/xc_nn_spin

# Neural network architecture (must match exported model)
NNXC.FX_IN         1    # Number of inputs for Fx network
NNXC.FC_IN         3    # Number of inputs for Fc network
NNXC.H             16   # Hidden-layer width (neurons per layer)
```

### Notes

- `XC.Authors NNGGA` activates the NN-based GGA functional.
- `NNXC.Prefix` must match the exported parameter file prefix.
- `NNXC.FX_IN`, `NNXC.FC_IN`, and `NNXC.H` must match the architecture of the trained/exported model.

If using an unpolarized model:

```text
NNXC.Prefix        nn_params/xc_nn
```

---

## 3. Validation Workflow

### Step 1 — Validate NN Forward Pass (Fx, Fc Only)

```bash
./test_1d_fx_fc
```

Compare with Python reference.

**Expected tolerance:** ~1e-15 (machine precision)

---

### Step 2 — Validate First Analytic Derivatives (1D Analytic Density)

```bash
./test_1d_nngga
```

Compare with Python reference.

**Expected tolerance:**

- First derivatives: ~1e-8
- Energy density: ~1e-7 (LDA reference differences)

---

### Step 3 — Validate Real 3D Grid Test

```bash
./test_xc_nn
python dump_jax_grid_csv.py
```

Compare:

- `exc`
- `edens`
- `vrhou`, `vrhod`
- Gradient-vector derivatives (`Pu`, `Pd`)

⚠️ **Do NOT compare final KS potentials directly.**

---

## 4. Important Conceptual Note

The XC potential is defined as:

```
v_xc = ∂e/∂ρ − ∇·(∂e/∂∇ρ)
```

- **PySCF** handles the divergence term analytically via integration by parts inside matrix element construction.
- **SIESTA** evaluates the divergence numerically on the real-space grid (Balbás & Soler, PRB 64, 165110).

Therefore:

- Differences in final grid potentials are due to discretization, not functional inconsistency.
- Validation should focus on energy density and first derivatives.

---

## 5. Numerical Safety

Reduced gradients are defined with a tiny denominator to avoid NaNs when ρ → 0:

```
s = |∇ρ| / (2 k_F ρ + 1e-30)
```

This matches the JAX-side practice and prevents division-by-zero issues in vacuum regions.

---

**End of README**
