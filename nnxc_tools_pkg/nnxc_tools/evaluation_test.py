from functools import partial
import jax
jax.config.update("jax_enable_x64", True)  # Enable 64 bit precision
import jax.numpy as jnp
import numpy as np
from pyscf import gto, dft
from pyscf.data import elements
from pathlib import Path
import re
import warnings
from typing import Tuple
from modules.custom_xc import custom_potential_ft_ec_sc, custom_potential_ft_ex_sc,\
    custom_potential_ft_ec_sc_SIG, custom_potential_ft_ex_sc_SIG


PREFER_XYZ = ("struc.xyz", "struct.xyz", "geometry.xyz")

    
def get_my_custom_xc(model_fx, model_fc, sigma=False):
    if sigma:
    
        return custom_potential_ft_ex_sc_SIG, custom_potential_ft_ec_sc_SIG
    else:
        try:
            model_name_lower_fx = model_fx.name.lower()
            model_name_lower_fc = model_fc.name.lower()
            if 'sig' in model_name_lower_fx or 'sig' in model_name_lower_fc:
                warnings.warn("""CHECK YOUR NETWORKS! The model is trained
with sigma as input. Put sigma=True""")
        except AttributeError:
            print('Your functional has not atribute name. If it is a function, \
you may continue safely. If it is a trained functional, be carefull as a lot \
of functions expect a .name attribute.')

        return custom_potential_ft_ex_sc, custom_potential_ft_ec_sc

def evaluate_customxc_justoutput_SIG(model_fx, model_fc, rho0, sigma):
    custom_ex, custom_ec = get_my_custom_xc(model_fx, model_fc, sigma=True)

    custom_ex = partial(custom_ex, model_fx=model_fx)
    custom_ec = partial(custom_ec, model_fc=model_fc)
    rho0 = jnp.asarray(rho0)
    # Calculate the "custom" energy with rho -- THIS IS e
    exc = jax.vmap(custom_ex)(rho0, sigma) + jax.vmap(custom_ec)(rho0, sigma)

    def rho_times_func(rho0, sigma):
        return rho0*custom_ex(rho0, sigma) + rho0*custom_ec(rho0, sigma)
    vrho, vsigma = jax.vmap(
        jax.grad(rho_times_func, argnums=(0, 1)))(rho0, sigma)
    # The outputs are expected to be numpy arrays
    eps = 1e-14
    exc = jnp.where(jnp.abs(rho0) < eps, 0.0, exc)
    vrho = jnp.where(jnp.abs(rho0) < eps, 0.0, vrho)
    vsigma = jnp.where(jnp.abs(rho0) < eps, 0.0, vsigma)
    exc = np.asarray(exc)
    vrho = np.asarray(vrho)
    vsigma = np.asarray(vsigma)

    vxc = (vrho, vsigma)
    return exc, vxc

def evaluate_customxc_justoutput_only_e(model_fx, model_fc, rho0, sigma, sigma_input=None):
    """ Only returns exc, ex and ec on grid, no potentials."""

    if sigma_input is None:
        warnings.warn("PUTING SIGMA TO TRUE")
        sigma_input = True
    custom_ex, custom_ec = get_my_custom_xc(model_fx, model_fc, sigma_input)

    custom_ex = partial(custom_ex, model_fx=model_fx)
    custom_ec = partial(custom_ec, model_fc=model_fc)
    rho0 = jnp.asarray(rho0)
    # Calculate the "custom" energy with rho -- THIS IS e
    # REMEMBER: custom_exc ALWAYS expects sigma as input
    ex = jax.vmap(custom_ex)(rho0, sigma)
    ec = jax.vmap(custom_ec)(rho0, sigma)

    exc = ex + ec
    # The outputs are expected to be numpy arrays
    eps = 1e-14
    exc = jnp.where(jnp.abs(rho0) < eps, 0.0, exc)

    return exc, ex, ec


def lib_xc_obtain_all(sdir, idx, subidx, basis='def2-TZVP', grid_level=4):
    mol = return_mol_from_sdir(sdir, idx, subidx, BASIS=basis)
    if mol is None:
        return None, None, None, None, None # We skip spin-polarized molecules for now
    mf, coords, weights, ao_deriv = rks_pbe_scf_and_grids_from_mol(mol, xc="PBE", grid_level=grid_level)
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
    return rho, sigma, grad, mf, weights


def lib_xc_obtain_all_from_mf(mf):

    mf, coords, weights, ao_deriv = rks_pbe_scf_and_grids_from_mf(mf)
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
    return rho, sigma, grad, mf, weights


def libxc_exc_on_grid(mf, rho, grad):
    """ Returns exc, ex, ec on the grid for a given mf object and density+grad.
    
    Args:
        mf: pyscf dft.RKS object (after mf.kernel())
        rho: array, density on the grid
        grad: tuple of arrays, (grad_x, grad_y, grad_z) on the grid
    Returns:
        exc, ex, ec: exchange-correlation energy density and components on the grid
    """
    ni = mf._numint
    gx, gy, gz = grad
    # PySCF/libxc expects for GGA (spin=0): a density array shaped (nvar, ngrids)
    # with variables [rho, grad_x, grad_y, grad_z]. It forms sigma internally.
    rho_gga = np.vstack((rho, gx, gy, gz))  # shape (4, ngrids)
    exc_x, vxc_x, _, _ = ni.eval_xc("GGA_X_PBE", rho_gga, spin=0, deriv=1)
    exc_c, vxc_c, _, _ = ni.eval_xc("GGA_C_PBE", rho_gga, spin=0, deriv=1)
    return exc_x + exc_c, exc_x, exc_c



def xyz_to_mol(xyz: Path, basis: str, charge: int) -> gto.Mole:
    """Convert an .xyz file to a pyscf.gto.Mole with given basis and charge.
    
    Infers multiplicity from total electrons (even=RHF, odd=UHF).

    Args:
        xyz: Path to .xyz file (may have natom line and/or blank lines)
        basis: Basis set name (e.g., "def2-TZVP")
        charge: Integer molecular charge
    Returns:
        mol: pyscf.gto.Mole object.
    """
    lines = xyz.read_text().strip().splitlines()
    try:
        nat = int(lines[0].strip()); start = 2
    except:
        nat = None; start = 0
    syms, coords = [], []
    for ln in lines[start:]:
        if not ln.strip(): continue
        t = ln.split()
        if len(t) < 4: continue
        syms.append(t[0]); coords.append([float(t[1]),float(t[2]),float(t[3])])
    if nat is not None:
        syms, coords = syms[:nat], coords[:nat]

    ne = sum(elements.charge(s) for s in syms) - charge
    spin = ne % 2  # 0 (even) -> restricted; 1 (odd) -> unrestricted

    mol = gto.Mole()
    mol.atom   = "\n".join(f"{s} {x} {y} {z}" for s,(x,y,z) in zip(syms, coords))
    mol.unit   = "Angstrom"
    mol.basis  = basis
    mol.charge = charge
    mol.spin   = spin
    mol.build(verbose=0)
    return mol

def pick_xyz(subset_dir: Path, idx: int, subidx: str) -> Path:
    cands = sorted(subset_dir.glob(f"{idx:03d}__{subidx}__*.xyz"))
    if not cands:
        raise FileNotFoundError(f"No .xyz for {subset_dir.name}:{idx:03d}")
    # prefer struc/struct/etc
    for p in cands:
        tail = p.name.split("__",2)[-1]
        if tail in PREFER_XYZ: return p
    return cands[0]


def read_charge(subset_dir: Path, idx: int, subidx: str) -> int:
    """Read integer charge from matching CHRG (if present)."""
    for pat in (f"{idx:03d}__{subidx}__.CHRG", f"{idx:03d}__{subidx}__.chrg"):
        for p in sorted(subset_dir.glob(pat)):
            txt = p.read_text().strip()
            toks = re.findall(r"[-+]?\d+", txt)
            if toks:
                try: return int(toks[0])
                except: pass
    return 0  # default neutral
def rks_pbe_scf_and_grids_from_mol(mol, xc="PBE", grid_level=4):
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = grid_level
    mf.grids.build(with_non0tab=True)
    mf.kernel()
    coords = mf.grids.coords
    weights = mf.grids.weights
    ao_deriv = mf._numint.eval_ao(mol, coords, deriv=1)
    return mf, coords, weights, ao_deriv

def rks_pbe_scf_and_grids_from_mf(mf):
    mf.grids.build(with_non0tab=True)
    mf.kernel()
    coords = mf.grids.coords
    weights = mf.grids.weights
    ao_deriv = mf._numint.eval_ao(mf.mol, coords, deriv=1)
    return mf, coords, weights, ao_deriv

def return_mol_from_sdir(sdir: Path, idx: int, subidx: int, BASIS="def2-SVP"):
    """ Return a mean-field object for a given molecule in the slim05 dataset."""
    charge = read_charge(sdir, idx, subidx)
    xyz_file = pick_xyz(sdir, idx, subidx)
    mol = xyz_to_mol(xyz_file, BASIS, charge)

    if mol.spin == 0:
        return mol
    else:
        return None  # We skip spin-polarized molecules for now


def integrate_exc(weights, rho, exc):
    return float(np.dot(weights, exc * rho))