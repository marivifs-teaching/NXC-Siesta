# pbe_compare.py
# Utilities to compare PBE exchange-correlation energy density between libxc (via PySCF)
# and a reference Python implementation (spin-unpolarized).

from __future__ import annotations
import numpy as np

# IMport SARA

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from modules.custom_xc import eUEG_LDA_x, eUEG_LDA_c
from modules.reference_functionals import  pw92c_unpolarized_scalar
from jax import typing as jt
try:
    from pyscf import gto, dft
except Exception as e:
    raise RuntimeError(
        "This module requires PySCF. Please install pyscf>=2.0.0 in your environment.\n"
        f"Import error was: {e}"
    )

###############################################################################
# 1) Low-level helpers: density and gradient on the grid
###############################################################################

# NOTE: We now prefer using mf._numint.eval_rho(..., xctype='GGA') to build (rho, grad)
# to guarantee identical conventions with libxc/PySCF. The helper below is retained
# for reference and potential independent checks.

def dm_rho_sigma_from_ao(ao_deriv, dm):
    """
    Compute rho and sigma = |grad rho|^2 on the grid for a spin-restricted system.

    Parameters
    ----------
    ao_deriv : ndarray
        AO values and first derivatives on the grid as returned by
        ni.eval_ao(mol, coords, deriv=1). Shape is (4, ngrids, nao):
        [0] AO values, [1] d/dx, [2] d/dy, [3] d/dz.
    dm : ndarray, shape (nao, nao)
        One-particle density matrix (spin-restricted).

    Returns
    -------
    rho : ndarray, shape (ngrids,)
        Electron density on each grid point.
    sigma : ndarray, shape (ngrids,)
        |grad rho|^2 on each grid point (GGA invariant).
    grad : tuple(ndarray, ndarray, ndarray)
        The components (gx, gy, gz) of grad rho.
    """
    ao0, aox, aoy, aoz = ao_deriv  # shapes (ngrids, nao)
    rho = np.einsum("gi,ij,gj->g", ao0, dm, ao0, optimize=True)
    gx  = 2.0 * np.einsum("gi,ij,gj->g", aox, dm, ao0, optimize=True)
    gy  = 2.0 * np.einsum("gi,ij,gj->g", aoy, dm, ao0, optimize=True)
    gz  = 2.0 * np.einsum("gi,ij,gj->g", aoz, dm, ao0, optimize=True)
    sigma = gx*gx + gy*gy + gz*gz
    return rho, sigma, (gx, gy, gz)

###############################################################################
# 2) Reference PBE implementation (spin-unpolarized)
###############################################################################

# PBE constants
_PBE_KAPPA = 0.804
_PBE_MU    = 0.2195149727645171
_PBE_BETA  = 0.06672455060314922
_PBE_GAMMA = 0.031090690869654895
_PI = np.pi

def _rho_safe(rho, eps=np.finfo(float).tiny):
    return np.maximum(rho, eps)

def _rho_safe_jax(rho, eps=jnp.finfo(float).tiny):
    return jnp.maximum(rho, eps)

def _rs_from_rho(rho):
    rho_s = _rho_safe(rho)
    return (3.0 / (4.0 * _PI * rho_s)) ** (1.0/3.0)

def _lda_x_eps_unpol(rho):
    rho_s = _rho_safe(rho)
    return -0.75 * (3.0/_PI)**(1.0/3.0) * (rho_s**(1.0/3.0))

def _pbe_exchange_eps_unpol(rho, sigma):
    rho_s = _rho_safe(rho)
    kF = (3.0 * (_PI**2) * rho_s) ** (1.0/3.0)
    s2 = sigma / ((2.0*kF*rho_s)**2 + 1e-30)
    Fx = 1.0 + _PBE_KAPPA - _PBE_KAPPA / (1.0 + _PBE_MU * s2 / _PBE_KAPPA)
    return _lda_x_eps_unpol(rho_s) * Fx

# --- PW92 LDA correlation (vectorized over k=0,1,2) and return unpolarized column k=0 ---

def _pw92_c_eps_unpol_rho(rho):
    """
    Perdew-Wang 1992 LDA correlation (unpolarized) energy per particle, eps_c^LDA.
    Vectorized over the three PW92 parameter sets (k=0,1,2); returns unpolarized k=0.
    Ref: J.P. Perdew & Y. Wang, Phys. Rev. B 45, 13244 (1992).
    """
    rho_s = _rho_safe(rho)
    rs = (3.0 / (4.0 * _PI * rho_s)) ** (1.0/3.0)
    A      = np.array([0.031090690869654895, 0.015545, 0.016887])
    ALPHA1 = np.array([0.21370,  0.20548,  0.11125 ])
    BETA1  = np.array([7.5957,   14.1189,  10.357  ])
    BETA2  = np.array([3.5876,   6.1977,   3.6231  ])
    BETA3  = np.array([1.6382,   3.3662,   0.88026 ])
    BETA4  = np.array([0.49294,  0.62517,  0.49671 ])
    rs = rs.reshape(-1)
    sqrt_rs = np.sqrt(rs)
    B = (np.outer(sqrt_rs, BETA1)
         + np.outer(rs, BETA2)
         + np.outer(rs * sqrt_rs, BETA3)
         + np.outer(rs * rs, BETA4))
    C = 1.0 + 1.0 / (2.0 * (A[None, :]) * B)
    G = -2.0 * (A[None, :]) * (1.0 + (ALPHA1[None, :] * rs[:, None])) * np.log(C)
    return G[:, 0]


def _pw92_c_eps_unpol_rho_jax(rho):
    """
    Perdew-Wang 1992 LDA correlation (unpolarized) energy per particle, eps_c^LDA.
    Vectorized over the three PW92 parameter sets (k=0,1,2); returns unpolarized k=0.
    Ref: J.P. Perdew & Y. Wang, Phys. Rev. B 45, 13244 (1992).
    JAX version.
    """
    rho_s = _rho_safe_jax(rho)
    rs = (3.0 / (4.0 * jnp.pi * rho_s)) ** (1.0/3.0)
    A      = jnp.array([0.031090690869654895, 0.015545, 0.016887])
    ALPHA1 = jnp.array([0.21370,  0.20548,  0.11125 ])
    BETA1  = jnp.array([7.5957,   14.1189,  10.357  ])
    BETA2  = jnp.array([3.5876,   6.1977,   3.6231  ])
    BETA3  = jnp.array([1.6382,   3.3662,   0.88026 ])
    BETA4  = jnp.array([0.49294,  0.62517,  0.49671 ])
    rs = rs.reshape(-1)
    sqrt_rs = jnp.sqrt(rs)
    B = (jnp.outer(sqrt_rs, BETA1)
         + jnp.outer(rs, BETA2)
         + jnp.outer(rs * sqrt_rs, BETA3)
         + jnp.outer(rs * rs, BETA4))
    C = 1.0 + 1.0 / (2.0 * (A[None, :]) * B)
    G = -2.0 * (A[None, :]) * (1.0 + (ALPHA1[None, :] * rs[:, None])) * jnp.log(C)
    return G[:, 0]

def _pbe_correlation_eps_unpol(rho, sigma):
    """PBE correlation energy per particle (unpolarized)."""
    rho_s = _rho_safe(rho)
    eps_c_lda = _pw92_c_eps_unpol_rho(rho_s)
    kF = (3.0 * (_PI**2) * rho_s) ** (1.0/3.0)
    k_s = np.sqrt(4.0 * kF / _PI)
    t2 = sigma / ((2.0*k_s*rho_s)**2 + 1e-30)
    # Correct PBE A(rs) denominator: gamma * (exp(-eps_c_lda/gamma) - 1)
    denom = _PBE_GAMMA * (np.exp(-eps_c_lda / _PBE_GAMMA) - 1.0)
    denom_sign = np.where(denom >= 0.0, 1.0, -1.0)
    denom = np.where(np.abs(denom) < 1e-14, denom_sign * 1e-14, denom)
    A_rs = _PBE_BETA / denom
    num = (_PBE_BETA / _PBE_GAMMA) * t2 * (1.0 + A_rs * t2)
    den = 1.0 + A_rs * t2 + (A_rs*A_rs) * (t2*t2)
    den_sign = np.where(den >= 0.0, 1.0, -1.0)
    den = np.where(np.abs(den) < 1e-18, den_sign * 1e-18, den)
    # Use log1p and guard against hitting -1 from below (machine epsilon nudge)
    arg = num / den
    arg = np.where(arg > -1.0 + np.finfo(float).eps, arg, -1.0 + np.finfo(float).eps)
    H = _PBE_GAMMA * np.log1p(arg)
    return eps_c_lda + H, eps_c_lda, t2, H, rho_s

def pbe_corr_terms_unpolarized(rho, sigma):
    """
    Return correlation terms separately for diagnostics:
    returns (rs, t2, eps_c_lda, H, eps_c_total)
    """
    eps_c_total, eps_c_lda, t2, H, rs = _pbe_correlation_eps_unpol(rho, sigma)
    return rs, t2, eps_c_lda, H, eps_c_total


def pbe_exc_unpolarized_reference(rho, sigma):
    ex = _pbe_exchange_eps_unpol(rho, sigma)
    ec, _, _, _, _ = _pbe_correlation_eps_unpol(rho, sigma)
    return ex + ec, ex, ec


## FUNCIONES SARA


def Fc_packedSIG(inputs) -> jt.ArrayLike:

    rho, sigma = inputs
    grad_rho = jnp.sqrt(sigma)

    return Fc_scalar(rho, grad_rho)

def Fx_packedSIG(inputs) -> jt.ArrayLike:
    """Exact reference exchange enhancement factor.

    As a function of n and ∇n.
    Args:
        inputs (tuple): (rho, grad_rho)

    Returns:
        jt.ArrayLike: Fx
    """
    rho, sigma = inputs

    return FxSIG(rho, sigma)

def FxSIG(rho: jt.ArrayLike, sigma: jt.ArrayLike) -> jt.ArrayLike:
    """Exact reference exchange enhancement factor.

    As a function of n and sigma.
    Args:
        rho (jt.ArrayLike): 
        sigma (jt.ArrayLike):

    Returns:
        jt.ArrayLike: Fx
    """
    # ADDED - AFFECTS RESULTS
    _PBE_KAPPA = 0.804
    _PBE_MU    = 0.2195149727645171
    # ADDED - DOES NOT AFFECT RESULT
    # rho = _rho_safe_jax(rho)

    kf = (3 * jnp.pi**2 * rho)**(1/3)

    # ADDED - CHANGE: s^2 instead of s
    s = jnp.sqrt(sigma) / (2 * kf * rho)
    # s2 = sigma / ((2.0*kf*rho)**2 + 1e-30)  - DOES NOT AFFECT RESULT
    kappa, mu = _PBE_KAPPA, _PBE_MU
    # Exchange enhancement factor
    # Fx = 1 + kappa - kappa / (1 + mu * s2 / kappa)  #  Before: s**2 = s2
    Fx = 1 + kappa - kappa / (1 + mu * s * s / kappa)  #  After: s**2 = s2

    return Fx


def Fc_scalar(rho: jt.ArrayLike, grad_rho: jt.ArrayLike) -> jt.ArrayLike:
    """Exact reference correlation enhancement function.

    The energy per unit electron is taken for PW LDA (see pw92c_unpolarized),
    assuming zeta=0.

    Args:
        rho (jt.ArrayLike): 
        grad_rho (jt.ArrayLike): 

    Returns:
        jt.ArrayLike: Fc
    """
    # ADDED
    _PBE_BETA  = 0.06672455060314922
    # ADDED
    rho = _rho_safe_jax(rho)

    pi = jnp.pi
    k_F = (3 * pi**2 * rho)**(1/3)
    k_s = jnp.sqrt((4 * k_F) / pi)
    t = jnp.abs(grad_rho) / (2 * k_s * rho)
    # beta = 0.066725
    beta = _PBE_BETA
    gamma = (1 - jnp.log(2)) / (pi**2)

    # Use pw92c_unpolarized for e_heg_c
    e_heg_c = pw92c_unpolarized_scalar(rho)
    # ADDED - CHANGE
    # e_heg_c = _pw92_c_eps_unpol_rho_jax(rho)

    A = (beta / gamma) / (jnp.exp(-e_heg_c / (gamma)) - 1)

    H = gamma * jnp.log(1 + (beta / gamma) * t**2 *
                        ((1 + A * t**2) / (1 + A * t**2 + A**2 * t**4)))

    Fc = 1 + (H / e_heg_c)  # correlation enhancement factor

    return Fc
def evaluate_customxc_justoutput_SIG(rho0, sigma):
    eps = 1e-14  # As in PySCF
    def custom_potential_ft_ex_sc_SIG(rho0, sigma):
        """
        ft = from trained
        SIG = network expects sigma as input
        """
        inputs = jnp.array([rho0, sigma])

        # Evaluate our model
        fx = Fx_packedSIG(inputs)

        return fx*eUEG_LDA_x(rho0)

    def custom_potential_ft_ec_sc_SIG(rho0, sigma):
        """
        ft = from trained
        SIG = network expects sigma as input
        """
        inputs = jnp.array([rho0, sigma])

        # Evaluate our model
        fc = Fc_packedSIG(inputs)

        return fc*eUEG_LDA_c(rho0)

    rho0 = jnp.asarray(rho0)
    exc_x = jax.vmap(custom_potential_ft_ex_sc_SIG)(rho0, sigma)
    exc_c = jax.vmap(custom_potential_ft_ec_sc_SIG)(rho0, sigma)
    exc_x = np.where(rho0 < eps, 0.0, exc_x)
    exc_c = np.where(rho0 < eps, 0.0, exc_c)
    exc = exc_x + exc_c

    return exc, exc_x, exc_c


# From libxc




###############################################################################
# 3) High-level: build RKS(PBE), SCF, grids, and compare per-grid eps_xc
###############################################################################

def rks_pbe_scf_and_grids(mol, xc="PBE", grid_level=4):
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = grid_level
    mf.grids.build(with_non0tab=True)
    mf.kernel()
    coords = mf.grids.coords
    weights = mf.grids.weights
    ao_deriv = mf._numint.eval_ao(mol, coords, deriv=1)
    return mf, coords, weights, ao_deriv


def libxc_exc_on_grid(mf, rho, grad):
    ni = mf._numint
    gx, gy, gz = grad
    # PySCF/libxc expects for GGA (spin=0): a density array shaped (nvar, ngrids)
    # with variables [rho, grad_x, grad_y, grad_z]. It forms sigma internally.
    rho_gga = np.vstack((rho, gx, gy, gz))  # shape (4, ngrids)
    exc_x, vxc_x, _, _ = ni.eval_xc("GGA_X_PBE", rho_gga, spin=0, deriv=1)
    exc_c, vxc_c, _, _ = ni.eval_xc("GGA_C_PBE", rho_gga, spin=0, deriv=1)
    return exc_x + exc_c, exc_x, exc_c


def integrate_exc(weights, rho, exc):
    return float(np.dot(weights, exc * rho))


def compare_2(dict1, dict2, weights, rho, verbose=True):
    type_1 = dict1['type']
    type_2 = dict2['type']

    print(f"Comparison between {type_1} and {type_2}:")

    E_x_1 = dict1['E_x']
    E_x_2 = dict2['E_x']
    E_c_1 = dict1['E_c']
    E_c_2 = dict2['E_c']
    E_xc_1 = dict1['E_xc']
    E_xc_2 = dict2['E_xc']

    ex_1 = dict1['ex']
    ex_2 = dict2['ex']
    ec_1 = dict1['ec']
    ec_2 = dict2['ec']
    exc_1 = dict1['exc']
    exc_2 = dict2['exc']
    # Per-point differences
    diff_ex  = ex_1  - ex_2
    diff_ec  = ec_1  - ec_2
    diff_exc = exc_1 - exc_2

    # BORRAR
    print("[debug]", type(diff_ex), type(diff_ec), type(diff_exc))


    # Weighted norms
    l2_x  = float(np.sqrt(np.dot(weights, (diff_ex**2)  * rho)))
    l2_c  = float(np.sqrt(np.dot(weights, (diff_ec**2)  * rho)))
    l2    = float(np.sqrt(np.dot(weights, (diff_exc**2) * rho)))
    max_x = float(np.max(np.abs(diff_ex)))
    max_c = float(np.max(np.abs(diff_ec)))
    max_t = float(np.max(np.abs(diff_exc)))

    wr = weights * rho
    mask = wr > 1e-14
    if np.any(mask):
        max_x_sig = float(np.max(np.abs(diff_ex[mask])))
        max_c_sig = float(np.max(np.abs(diff_ec[mask])))
        max_t_sig = float(np.max(np.abs(diff_exc[mask])))
    else:
        max_x_sig = max_x
        max_c_sig = max_c
        max_t_sig = max_t

    if verbose:
        print("Grid points:", len(weights))
        print(f"E_x  ({type_1})     = {E_x_1: .12f} Ha")
        print(f"E_x  ({type_2}) = {E_x_2: .12f} Ha   | abs diff = {abs(E_x_2 - E_x_1):.6e} Ha")
        print(f"   per-point ex diff: L2~{l2_x:.6e}, max|Δ|={max_x:.6e} (Ha/e)")
        print(f"                  max|Δ| (significant points)={max_x_sig:.6e} (Ha/e)")
        print(f"E_c  ({type_1})     = {E_c_1: .12f} Ha")
        print(f"E_c  ({type_2}) = {E_c_2: .12f} Ha   | abs diff = {abs(E_c_2 - E_c_1):.6e} Ha")
        print(f"   per-point ec diff: L2~{l2_c:.6e}, max|Δ|={max_c:.6e} (Ha/e)")
        print(f"                  max|Δ| (significant points)={max_c_sig:.6e} (Ha/e)")
        print(f"E_xc ({type_1})     = {E_xc_1: .12f} Ha")
        print(f"E_xc ({type_2}) = {E_xc_2: .12f} Ha   | abs diff = {abs(E_xc_2 - E_xc_1):.6e} Ha")
        print(f"   per-point exc diff: L2~{l2:.6e}, max|Δ|={max_t:.6e} (Ha/e)")
        print(f"                  max|Δ| (significant points)={max_t_sig:.6e} (Ha/e)")


    return {
        "info_dicts": (type_1, type_2),
        "ex_diff": diff_ex,
        "ec_diff": diff_ec,
        "exc_diff": diff_exc,
    }



def compare_pbe_exc_per_grid(mol, grid_level=4, verbose=True, compare=True):
    mf, coords, weights, ao_deriv = rks_pbe_scf_and_grids(mol, xc="PBE", grid_level=grid_level)
    dm = mf.make_rdm1()

    # Use PySCF's own evaluator to ensure rho and grad match libxc conventions
    vals = mf._numint.eval_rho(mf.mol, ao_deriv, dm, xctype='GGA')
    if vals.shape[0] == 4:
        rho_pscf, gx, gy, gz = vals
    else:
        rho_pscf, gx, gy, gz = vals.T
    sigma = gx*gx + gy*gy + gz*gz
    rho = rho_pscf
    grad = (gx, gy, gz)

    # Evaluate libxc and reference implementations
    exc_libxc_pyscf, ex_libxc_pyscf, ec_libxc_pyscf = libxc_exc_on_grid(mf, rho, grad)
    exc_ref,   ex_ref,   ec_ref   = pbe_exc_unpolarized_reference(rho, sigma)
    exc_sara, ex_sara, ec_sara = evaluate_customxc_justoutput_SIG(rho, sigma)

    # Integrate X and C separately as well as total XC
    # E_x = ∫ rho(r) * eps_x(r) dr
    E_x_libxc_pyscf = integrate_exc(weights, rho, ex_libxc_pyscf)
    E_x_ref   = integrate_exc(weights, rho, ex_ref)
    E_x_sara  = integrate_exc(weights, rho, ex_sara)

    # E_c = ∫ rho(r) * eps_c(r) dr
    E_c_libxc_pyscf = integrate_exc(weights, rho, ec_libxc_pyscf)
    E_c_ref   = integrate_exc(weights, rho, ec_ref)
    E_c_sara  = integrate_exc(weights, rho, ec_sara)

    # E_xc = ∫ rho(r) * eps_xc(r) dr
    E_xc_libxc_pyscf = integrate_exc(weights, rho, exc_libxc_pyscf)
    E_xc_ref   = integrate_exc(weights, rho, exc_ref)
    E_xc_sara  = integrate_exc(weights, rho, exc_sara)

    dict_all_info = {
        "coords": coords,
        "weights": weights,
        "rho": rho,
        "sigma": sigma,
        "exc_libxc_pyscf": exc_libxc_pyscf,
        "ex_libxc_pyscf": ex_libxc_pyscf,
        "ec_libxc_pyscf": ec_libxc_pyscf,
        "exc_ref": exc_ref,
        "ex_ref": ex_ref,
        "ec_ref": ec_ref,
        "exc_sara": exc_sara,
        "ex_sara": ex_sara,
        "ec_sara": ec_sara,
        "E_x_libxc_pyscf": E_x_libxc_pyscf,
        "E_x_ref": E_x_ref,
        "E_x_sara": E_x_sara,
        "E_c_libxc_pyscf": E_c_libxc_pyscf,
        "E_c_ref": E_c_ref,
        "E_c_sara": E_c_sara,
        "E_xc_libxc_pyscf": E_xc_libxc_pyscf,
        "E_xc_ref": E_xc_ref,
        "E_xc_sara": E_xc_sara,
    }

    if compare:
        info_ref = {'type': 'ref', 'E_x': E_x_ref, 'E_c': E_c_ref, 'E_xc': E_xc_ref,
                    'ex': ex_ref, 'ec': ec_ref, 'exc': exc_ref}
        info_libxc = {'type': 'libxc_pyscf', 'E_x': E_x_libxc_pyscf, 'E_c': E_c_libxc_pyscf, 'E_xc': E_xc_libxc_pyscf,
                    'ex': ex_libxc_pyscf, 'ec': ec_libxc_pyscf, 'exc': exc_libxc_pyscf}
        info_sara = {'type': 'sara', 'E_x': E_x_sara, 'E_c': E_c_sara, 'E_xc': E_xc_sara, 
                    'ex': ex_sara, 'ec': ec_sara, 'exc': exc_sara}

        print("[info] Calculating differences between libxc (via PySCF) and reference PBE implementation:")
        out1 = compare_2(info_libxc, info_ref, weights, rho, verbose=verbose)
        print("\n[info] Calculating differences between libxc (via PySCF) and PBE SARA implementation:")
        out2 = compare_2(info_libxc, info_sara, weights, rho, verbose=verbose)
        print("\n[info] Calculating differences between reference PBE and PBE SARA implementation:")
        out3 = compare_2(info_ref, info_sara, weights, rho, verbose=verbose)

        return dict_all_info, out1, out2, out3
    else:
        return dict_all_info
