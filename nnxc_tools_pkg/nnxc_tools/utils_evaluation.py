from pathlib import Path
import re
from pyscf import gto, dft
from pyscf.lib.exceptions import BasisNotFoundError
from pyscf.data import elements
import numpy as np
import matplotlib.pyplot as plt
import json
import traceback
import pyscf.lib.exceptions as pyscf_ex
# -------------------------- MOLECULE HANDLING --------------------------

PREFER_XYZ = ("struc.xyz", "struct.xyz", "geometry.xyz")


def get_mol_from_index(dir_slim, subset: str, idx: int, subidx: str) -> gto.Mole:
    """Get a pyscf.gto.Mole from slim subset, index, and subindex.

    Args:
        dir_slim: Path to the slim dataset directory.
        subset: Subset name (e.g., "WATER27").
        idx: Molecule index (integer).
        subidx: Molecule subindex (string).
    Returns:
        mol: pyscf.gto.Mole object.
    """
    info_runs = json.load(open(dir_slim / "run_info.json", "r"))
    try:
        basis = info_runs["basis"]
    except KeyError:
        print(f"Warning: basis not found in {dir_slim}/run_info.json; defaulting to def2-TZVP")
        basis = "def2-TZVP"
    subset_dir = dir_slim / subset
    xyz_path = pick_xyz(subset_dir, idx, subidx)
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ file {xyz_path} does not exist.")
    charge = read_charge(subset_dir, idx, subidx)
    mol = xyz_to_mol(xyz_path, basis=basis, charge=charge)
    if mol is None:
        raise ValueError(f"Could not create molecule from {xyz_path} with basis {basis} and charge {charge}.")
    return mol


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
    try:
        mol.build(verbose=0)
    except Exception as e:
        # now re-raise as a ValueError if it's the basis-not-found case
        if isinstance(e, pyscf_ex.BasisNotFoundError):
            raise ValueError(f"ERROR: Basis {basis} not found for molecule in {xyz}.")
    return mol

def pick_xyz(subset_dir: Path, idx: int, subidx: str) -> Path:
    cands = sorted(subset_dir.glob(f"{idx:03d}__{subidx}__*.xyz"))
    if not cands:
        raise FileNotFoundError(f"No .xyz for {subset_dir.name}:{idx:03d}:{subidx} in {subset_dir}")
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


# -------------- EVALUATION UTILS ---------------

def evaluate_mol_pyscf_return_all(mol, xc_code, GRID_LEVEL=4, CONVERGE_TOL=1e-9):
    """ Returns everything, including sigma and grad_rho_tuple.
    
    Args:
        mol: pyscf molecule
        xc_code: str, e.g. 'PBE'
    Returns:
        e_tot, rho0, grad_rho_tuple, sigma, ao, mf"""

    mf = dft.RKS(mol)
    mf.grids.level = GRID_LEVEL
    mf.conv_tol = CONVERGE_TOL
    mf.xc = xc_code
    mf.kernel()
    COORDS = mf.grids.coords
    ao = dft.numint.eval_ao(mol, COORDS, deriv=1)
    dm = mf.make_rdm1()
    inputs = dft.numint.eval_rho(mol, ao, dm, xctype='GGA')
    rho0, dx, dy, dz = inputs
    sigma = dx**2+dy**2+dz**2
    grad_rho_tuple = (dx, dy, dz)

    return mf.e_tot, rho0, grad_rho_tuple, sigma, ao, mf


# -------------------------- PLOTTING --------------------------
HARTREE_TO_EV = 27.2114


def get_npz_files_from_directory(filepath, run_type,  f_type=None, tail_info=""):
    """Extract dataset names from filenames in a given directory.

    Example filename: 'results/Fcnet_sigma_d3_n16_s42/errors_WATER27_PBE_Fc_epoch8000.npz'
    Args:
        filepath (Path): Directory path containing the files.
        f_type (str): Type of the file, e.g., 'Fc' or 'Fx'.
        run_type (str): Type of the run, e.g., 'PBE' or 'HF'.
        tail_info (str): Additional information in the filename, e.g., 'epoch8000'.
    Returns:
        list: Sorted list of dataset names extracted from filenames.
    """
    filepath = Path(filepath)
    print(f"Looking for files in {filepath} with run_type={run_type}, f_type={f_type}, tail_info={tail_info}")
    if f_type is not None:
        tail_info = f"_{f_type}_{tail_info}"
    if tail_info and not tail_info.startswith('_'):
        tail_info = f"_{tail_info}"

    if not filepath.exists():
        raise ValueError(f"The provided filepath {filepath} does not exist.")
    all_files = sorted(filepath.glob(f"errors_*_{run_type}{tail_info}.npz"))
    if not all_files:
        raise ValueError(f"No files found in {filepath} matching the pattern")
    return [file.name for file in all_files]


def plot_histogram_subset(npz_file, logscale=True, abs=True):
    """Load and plot information from a given npz file."""
    data = np.load(npz_file)
    subset = npz_file.stem.split('_')[1]
    # Example: print the keys and shapes of the arrays in the npz file

    tags = data['tags']
    mol_names = [tag.split('_')[1] for tag in tags]
    errors_e_tot = data['errors_exc']*HARTREE_TO_EV
    fde = data[f'fde_xc']*HARTREE_TO_EV
    dde = data[f'dde_xc']*HARTREE_TO_EV
    if abs:
        errors_e_tot = np.abs(errors_e_tot)
        fde = np.abs(fde)
        dde = np.abs(dde)
    if not abs and logscale:
        print("Warning: logscale is not compatible with non-absolute values.\
               Disabling logscale.")
        logscale = False

    width = 0.35  # the width of the bars
    x = np.arange(len(mol_names))  # the label locations

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f'XC energy Errors for subset {subset}')
    ax.bar(x - width/2, fde, width, label='FDE', alpha=0.7)
    ax.bar(x - width/2, dde, width, label='DDE', bottom=fde, alpha=0.7)
    ax.bar(x + width/2, errors_e_tot, width, label='TEE')

    ax.set_xticks(x, mol_names, rotation=90)
    ax.set_ylabel('Error (eV)')
    if logscale:
        ax.set_yscale('log')

    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_histogram_subset_allreps_sep(npz_files, logscale=True):
    """Load and plot information from some given npz files.

    This npz files are expected to be from different repetitions in training of the same subset
    (i.e., trained the same way but with different random molecules)

    It creates a subplot for each repetition.

    Args:
        npz_files (list of Path): List of paths to the npz files.
        logscale (bool): Whether to use logarithmic scale for y-axis.
    """
    fig, ax = plt.subplots(1, len(npz_files), figsize=(3*len(npz_files), 6), sharey=True)
    for i, npz_file in enumerate(npz_files):
        data = np.load(npz_file)
        subset = npz_file.stem.split('_')[1]

        tags = data['tags']
        mol_names = [tag.split('_')[1] for tag in tags]
        errors_e_tot = data['errors_exc']*HARTREE_TO_EV
        fde = np.abs(data[f'fde_xc']*HARTREE_TO_EV)
        dde = np.abs(data[f'dde_xc']*HARTREE_TO_EV)

        width = 0.35  # the width of the bars
        x = np.arange(len(mol_names))  # the label locations

        fig.suptitle(f'Errors for subset {subset} (XC energies)')
        ax[i].bar(x - width/2, fde, width, label='Functional-driven error (fde)')
        ax[i].bar(x - width/2, dde, width, label='Density-driven error (dde)', bottom=fde)
        ax[i].bar(x + width/2, np.abs(errors_e_tot), width, label='XC energy error')

        ax[i].set_xticks(x, mol_names, rotation=90)
        ax[i].set_ylabel('Error (eV)')
        if logscale:
            ax[i].set_yscale('log')

    ax[-1].legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize='small')
    plt.tight_layout()
    plt.show()


def plot_histogram_all_subsets(filepath, files,
                               log_scale=True, title=None, ylim=None,
                               ignore_pol=True, ignore_nonconv=True, slim_set=5):
    """Plot mean errors (TEE, FDE, DDE) across subsets from given files.

    The mean is in absolute value. The files for the different subsets are
    expected to be provided. See get_npz_files_from_directory function.

    Args:
        filepath (Path): Directory path containing the files.
        files (list): List of filenames to process.
        log_scale (bool): Whether to use logarithmic scale for y-axis. Default: True.
        title (str): Title for the plot. If None, a default title is used.
        ylim (tuple): y-axis limits. If None, no limits are set."""
    subsets = []
    mean_etot_err = []
    mean_fde = []
    mean_dde = []
    if ignore_pol:
        print("Ignoring non-polarized molecules in the analysis.")
    if ignore_nonconv:
        print("Ignoring non-converged molecules in the analysis.")

    if ignore_pol or ignore_nonconv:
        with open(f'../DataBases/GMTKN55/slim{slim_set:02d}/polarization_convergence_info.json', 'r') as f:
            pol_conv_info = json.load(f)
    for file in files:
        full_path = Path(filepath / file)
        if not full_path.exists():
            raise ValueError(f"The file {full_path} does not exist.")
        subset = file.split('_')[1]
        if subset in subsets:
            raise ValueError(f"Subset {subset} already processed.")
        subsets.append(subset)
        data = np.load(full_path)
        # Take the tags depending if we want pol molecules or conv molecules
        tags = [f'{subset}_{tag[:-4]}' for tag in data['tags']] # remove _PBE from tag
        nonpolarised_flags = [True]*len(tags)
        converged_flags = [True]*len(tags)
        if ignore_pol:
            # Only take polarised = False
            nonpolarised_flags = [not pol_conv_info[tag]['polarized'] for tag in tags]
        if ignore_nonconv:
            # Only take converged = True
            converged_flags = [pol_conv_info[tag]['converged'] for tag in tags]
        take_idx = [x and y for x, y in zip(nonpolarised_flags, converged_flags)]
        
        errors_e_tot = (data['errors_exc'][take_idx])*HARTREE_TO_EV
        fde = (data['fde_xc'][take_idx])*HARTREE_TO_EV
        dde = (data['dde_xc'][take_idx])*HARTREE_TO_EV
        mean_etot_err.append(np.mean(np.abs(errors_e_tot)))
        mean_fde.append(np.mean(np.abs(fde)))
        mean_dde.append(np.mean(np.abs(dde)))
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(subsets))
    width = 0.4
    ax.bar(x - width/2, mean_fde, width, label='Mean Functional-driven error (fde)')
    ax.bar(x - width/2, mean_dde, width, label='Mean Density-driven error (dde)', bottom=mean_fde)
    ax.bar(x + width/2, mean_etot_err, width, label='Mean Total energy error')
    ax.set_xticks(x, subsets, rotation=90)
    ax.set_ylabel('Mean Error (eV)')
    ax.plot([x.min(), x.max()+1], [0.043, 0.043], 'k--', lw=0.3)
    ax.set_xlim(x.min()-0.5, x.max()+1)
    ax.text(x.max()-0.5, 0.05, 'Ch.Acc: 0.043 eV', rotation=0, color='black')
    ax.grid(ls="--", linewidth=0.5, axis='y')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f'Mean Errors across subsets in {filepath.name}')
    if log_scale:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_histogram_all_subsets_reps(filepath, log_scale=True, title=None,
                                   ignore_pol=True, ignore_nonconv=True,
                                   slim_set=5, ylim=None, tail_info="_rep"):
    """Plot mean errors across subsets for different
    repetitions.

    A repetition is defined as a different training of the same subsets
    (i.e., trained the same way but with different random molecules).

    The errors are TEE, FDE, DDE. This mean is calculated in absolute
    value over the different repetitions.

    All the files in the given filepath are expected to be from different
    repetitions of the same subsets."""
    mean_etot_err = {}
    mean_fde = {}
    mean_dde = {}
    if ignore_pol:
        print("Ignoring non-polarized molecules in the analysis.")
    if ignore_nonconv:
        print("Ignoring non-converged molecules in the analysis.")

    if ignore_pol or ignore_nonconv:
        path_pol_conv_info = (
            f'../DataBases/GMTKN55/slim{slim_set:02d}/'
            'polarization_convergence_info.json'
        )
        with open(path_pol_conv_info, 'r') as f:
            pol_conv_info = json.load(f)

    all_files = sorted(filepath.glob(f"errors_*_PBE{tail_info}*.npz"))
    all_files = [file.name for file in all_files]

    all_subsets = set()
    for file in all_files:
        subset = file.split('_')[1]
        all_subsets.add(subset)
    all_subsets = sorted(list(all_subsets))
    if not all_subsets:
        raise ValueError(f"No subsets found in {filepath}.")

    for subset in all_subsets:
        mean_etot_err[subset] = []
        mean_fde[subset] = []
        mean_dde[subset] = []
        files = [file for file in all_files if f"_{subset}_" in file]
        for file in files:
            full_path = Path(filepath / file)
            if not full_path.exists():
                raise ValueError(f"The file {full_path} does not exist.")
            data = np.load(full_path)

            # Take the tags depending if we do not want polarized molecules
            # or molecules whose PBE minimisation did not converge
            tags = [f'{subset}_{tag[:-4]}'
                    for tag in data['tags']]  # remove _PBE from tag
            nonpolarised_flags = [True]*len(tags)
            converged_flags = [True]*len(tags)
            if ignore_pol:
                # Only take polarised = False
                nonpolarised_flags = [not pol_conv_info[tag]['polarized']
                                      for tag in tags]
            if ignore_nonconv:
                # Only take converged = True
                converged_flags = [pol_conv_info[tag]['converged']
                                   for tag in tags]
            take_idx = [x and y for x, y in
                        zip(nonpolarised_flags, converged_flags)]
            errors_e_tot = data['errors_exc']*HARTREE_TO_EV
            errors_e_tot = errors_e_tot[take_idx]
            fde = data['fde_xc']*HARTREE_TO_EV
            fde = fde[take_idx]
            dde = data['dde_xc']*HARTREE_TO_EV
            dde = dde[take_idx]
            mean_etot_err[subset].append(np.mean(np.abs(errors_e_tot)))
            mean_fde[subset].append(np.mean(np.abs(fde)))
            mean_dde[subset].append(np.mean(np.abs(dde)))
    # Now compute the overall mean for each subset
    for subset in all_subsets:
        mean_etot_err[subset] = np.mean(mean_etot_err[subset])
        mean_fde[subset] = np.mean(mean_fde[subset])
        mean_dde[subset] = np.mean(mean_dde[subset])
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_subsets))
    mean_etot_toplot = np.array(list(mean_etot_err.values()))
    mean_fde_toplot = np.array(list(mean_fde.values()))
    mean_dde_toplot = np.array(list(mean_dde.values()))
    width = 0.4
    ax.bar(x - width/2, mean_fde_toplot, width,
           label='Mean Functional-driven error (fde)')
    ax.bar(x - width/2, mean_dde_toplot, width,
           label='Mean Density-driven error (dde)', bottom=mean_fde_toplot)
    ax.bar(x + width/2, mean_etot_toplot, width,
           label='Mean Total energy error')
    ax.set_xticks(x, all_subsets, rotation=90)
    ax.set_ylabel('Mean Error (eV)')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Mean Errors across subsets using PBE')
    ax.plot([x.min(), x.max()+1], [0.043, 0.043], 'k--', lw=0.3)
    ax.set_xlim(x.min()-0.5, x.max()+1)
    ax.text(x.max()-0.5, 0.05, 'Ch.Acc: 0.043 eV', rotation=0, color='black')
    ax.grid(ls="--", linewidth=0.5, axis='y')
    if log_scale:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
  
    plt.tight_layout()
    plt.show()


def plot_histogram_MAE_all_subsets_diff_func_reps(
        filepaths, name_tags, title=None, cmap_name='viridis',
        ignore_pol=True, ignore_nonconv=True, slim_set=5, ylim=None,
        tail_infos=None, log_scale=True, figsize=(10,6),
        colors=None):
    """Calculate the mean errors for each subset for different functionals.

    We only plot the absolute mean error in energies, as we plot the MAE for
    each subset and for an arbitrary number of different functionals
    (or different trainings).
    We perform also the MAE between repetitions (i.e., the training in
    different sets of random molecules)

    All the files of the repetitions are stored in the `filepaths` list.
    """
    if tail_infos is None:
        tail_infos = ['']*len(filepaths)
    N = len(filepaths)
    if colors is None:
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i/(N+0.2)) for i in range(N)]  # avoid yellow
    elif len(colors) != N:
        print("Warning: number of colors provided does not match number of filepaths.")
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i/(N+0.2)) for i in range(N)]  # avoid yellow

    if N != len(name_tags):
        raise ValueError("Number of filepaths must match number of name_tags.")
    all_results = []
    if ignore_pol:
        print("Ignoring non-polarized molecules in the analysis.")
    if ignore_nonconv:
        print("Ignoring non-converged molecules in the analysis.")
    if ignore_pol or ignore_nonconv:
        path_pol_conv_info = (
            f'../DataBases/GMTKN55/slim{slim_set:02d}/'
            'polarization_convergence_info.json'
        )
        with open(path_pol_conv_info, 'r') as f:
            pol_conv_info = json.load(f)

    for i, filepath in enumerate(filepaths):
        mean_etot_err = {}

        all_files = sorted(filepath.glob(f"errors_*_PBE{tail_infos[i]}*.npz"))
        all_files = [file.name for file in all_files]

        all_subsets = set()
        for file in all_files:
            subset = file.split('_')[1]
            all_subsets.add(subset)
        all_subsets = sorted(list(all_subsets))
        if not all_subsets:
            raise ValueError(f"No subsets found in {filepath}.")

        for subset in all_subsets:
            mean_etot_err[subset] = []

            files = [file for file in all_files if f"_{subset}_" in file]
            for file in files:
                full_path = Path(filepath / file)
                if not full_path.exists():
                    raise ValueError(f"The file {full_path} does not exist.")
                data = np.load(full_path)
                # Take the tags depending if we do not want polarized molecules
                # or molecules whose PBE minimisation did not converge
                tags_take = [f'{subset}_{tag[:-4]}'
                             for tag in data['tags']]  # remove _PBE from tag
                nonpolarised_flags = [True]*len(tags_take)
                converged_flags = [True]*len(tags_take)
                if ignore_pol:
                    # Only take polarised = False
                    nonpolarised_flags = [not pol_conv_info[tag]['polarized']
                                          for tag in tags_take]
                if ignore_nonconv:
                    # Only take converged = True
                    converged_flags = [pol_conv_info[tag]['converged']
                                       for tag in tags_take]
                take_idx = [x and y for x, y in
                            zip(nonpolarised_flags, converged_flags)]
                errors_e_tot = data['errors_exc']*HARTREE_TO_EV
                errors_e_tot = errors_e_tot[take_idx]
                mean_etot_err[subset].append(np.mean(np.abs(errors_e_tot)))
        # Now compute the overall mean for each subset
        for subset in all_subsets:
            mean_etot_err[subset] = np.mean(mean_etot_err[subset])
        mean_etot_err = np.array(list(mean_etot_err.values()))

        all_results.append(mean_etot_err)
    # Plotting
    all_results = np.array(all_results)
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(all_subsets))
    width = 0.9/N
    for i in range(N):
        ax.bar(x + i*width, all_results[i], width=width, label=name_tags[i], color=colors[i])
    ax.set_xticks(x + width*(N-1)/2, all_subsets, rotation=90)
    ax.set_ylabel('Mean XC Energy Error (eV)')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Mean XC Energy Errors across subsets using PBE')
    if log_scale:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    ax.plot([x.min(), x.max()+1], [0.043, 0.043], 'k--', lw=0.3)
    ax.set_xlim(x.min()-0.5, x.max()+1)
    ax.text(x.max()-0.5, 0.05, 'Ch.Acc: 0.043 eV', rotation=0, color='black')
    ax.grid(ls="--", linewidth=0.5, axis='y')
    plt.tight_layout()
    plt.show()


def compare_histograms_all_subsets_2funcs(
        filepath_1, filepath_2, files_1, files_2,
        logscale=True, title=None):
    """Plot histograms for 2 different functionals
    
    We plot the comparison of all the energies (TEE, FDE and DDE)
    for 2 functionals
    """
    subsets_1 = []
    subsets_2 = []
    mean_etot_err_1 = []
    mean_etot_err_2 = []
    mean_fde_1 = []
    mean_fde_2 = []
    mean_dde_1 = []
    mean_dde_2 = []
    tail_info = ''
    energy_tag = 'errors_e_tot'

    for file in files_1:
        data = np.load(filepath_1 / file)
        subset = file.split('_')[1]
        if subset in subsets_1:
            raise ValueError(f"Subset {subset} already processed.")
        subsets_1.append(subset)
        errors_e_tot = data['errors_exc']*HARTREE_TO_EV
        fde = data['fde_xc']*HARTREE_TO_EV
        dde = data['dde_xc']*HARTREE_TO_EV
        mean_etot_err_1.append(np.mean(np.abs(errors_e_tot)))
        mean_fde_1.append(np.mean(np.abs(fde)))
        mean_dde_1.append(np.mean(np.abs(dde)))
    for file in files_2:
        data = np.load(filepath_2 / file)
        subset = file.split('_')[1]
        if subset in subsets_2:
            raise ValueError(f"Subset {subset} already processed.")
        subsets_2.append(subset)
        errors_e_tot = data[energy_tag]*HARTREE_TO_EV
        fde = data[f'fde{tail_info}']*HARTREE_TO_EV
        dde = data[f'dde{tail_info}']*HARTREE_TO_EV
        mean_etot_err_2.append(np.mean(np.abs(errors_e_tot)))
        mean_fde_2.append(np.mean(np.abs(fde)))
        mean_dde_2.append(np.mean(np.abs(dde)))
    if subsets_1 != subsets_2:
        raise ValueError("Subsets in both file lists do not match.")
    subsets = subsets_1
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(subsets))
    width = 0.2
    ax.bar(x - 3*width/2, mean_fde_1, width,
           label='Mean Functional-driven error (fde) - Method 1')
    ax.bar(x - 3*width/2, mean_dde_1, width,
           label='Mean Density-driven error (dde) - Method 1',
           bottom=mean_fde_1)
    ax.bar(x - width/2, mean_etot_err_1, width,
           label='Mean Total energy error - Method 1')
    ax.bar(x + width/2, mean_fde_2, width,
           label='Mean Functional-driven error (fde) - Method 2')
    ax.bar(x + width/2, mean_dde_2, width,
           label='Mean Density-driven error (dde) - Method 2',
           bottom=mean_fde_2)
    ax.bar(x + 3*width/2, mean_etot_err_2, width,
           label='Mean Total energy error - Method 2')

    ax.set_xticks(x, subsets, rotation=90)
    ax.set_ylabel('Mean Error (eV)')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f'Mean Errors across subsets - Comparing')
    ax.legend()
    if logscale:
        ax.set_yscale('log')
    plt.tight_layout()
    plt.show()


# -----------------OLD FUNCTIONS: DEPRECATED-----------------
def __plot_info_for_subset(npz_file, method, xc= True, logscale=True):
    """Load and plot information from a given npz file."""
    data = np.load(npz_file)
    subset = npz_file.stem.split('_')[1]
    # Example: print the keys and shapes of the arrays in the npz file
    tail_info = ''
    energy_tag = 'errors_e_tot'
    label_energy = 'Total energy error'
    if xc:
        tail_info = '_xc'
        energy_tag = 'errors_exc'
        label_energy = 'XC energy error'

    tags = data['tags']
    mol_names = [tag.split('_')[1] for tag in tags]
    errors_e_tot = data[energy_tag]*HARTREE_TO_EV
    fde = data[f'fde{tail_info}']*HARTREE_TO_EV
    dde = data[f'dde{tail_info}']*HARTREE_TO_EV

    width = 0.35  # the width of the bars
    x = np.arange(len(mol_names))  # the label locations
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f'Errors for subset {subset} using {method} method {"(XC energies)" if xc else ""}')
    ax.bar(x - width/2, fde, width, label='Functional-driven error (fde)')
    ax.bar(x - width/2, dde, width, label='Density-driven error (dde)', bottom=fde)
    ax.bar(x + width/2, np.abs(errors_e_tot), width, label=label_energy)

    ax.set_xticks(x, mol_names, rotation=90)
    ax.set_ylabel('Error (eV)')
    if logscale:
        ax.set_yscale('log')

    ax.legend()
    plt.tight_layout()
    plt.show()


def __mean_error_same_func(filepath, files, xc=False, method='PBE', log_scale=True):
    """Plot mean errors across subsets from given files."""
    if xc and method=='HF':
        raise ValueError("HF method does not use xc functionals.")
    subsets = []
    mean_etot_err = []
    mean_fde = []
    mean_dde = []
    tail_info = ''
    energy_tag = 'errors_e_tot'
    if xc:
        tail_info = '_xc'
        energy_tag = 'errors_exc'

    for file in files:
        full_path = Path(filepath / file)
        if not full_path.exists():
            raise ValueError(f"The file {full_path} does not exist.")
        data = np.load(full_path)
        subset = file.split('_')[1]
        if subset in subsets:
            raise ValueError(f"Subset {subset} already processed.")
        subsets.append(subset)
        errors_e_tot = data[energy_tag]*HARTREE_TO_EV
        fde = data[f'fde{tail_info}']*HARTREE_TO_EV
        dde = data[f'dde{tail_info}']*HARTREE_TO_EV
        mean_etot_err.append(np.mean(np.abs(errors_e_tot)))
        mean_fde.append(np.mean(np.abs(fde)))
        mean_dde.append(np.mean(np.abs(dde)))
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(subsets))
    width = 0.4
    ax.bar(x - width/2, mean_fde, width, label='Mean Functional-driven error (fde)')
    ax.bar(x - width/2, mean_dde, width, label='Mean Density-driven error (dde)', bottom=mean_fde)
    ax.bar(x + width/2, mean_etot_err, width, label='Mean Total energy error')
    ax.set_xticks(x, subsets, rotation=90)
    ax.set_ylabel('Mean Error (eV)')
    ax.set_title(f'Mean Errors across subsets using {method} method')
    if log_scale:
        ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.show()


def __mean_error_same_func_avoiding_errors(filepath, files, error_file, xc=False, method='PBE', log_scale=True):
    if xc and method=='HF':
        raise ValueError("HF method does not use xc functionals.")
    # Get tags with non-converged calculations
    all_tags_nc = __get_tags_for_non_converged(error_file, method)
    subsets = []
    mean_etot_err = []
    mean_fde = []
    mean_dde = []
    tail_info = ''
    energy_tag = 'errors_e_tot'
    if xc:
        tail_info = '_xc'
        energy_tag = 'errors_exc'

    for file in files:
        full_path = Path(filepath / file)
        if not full_path.exists():
            raise ValueError(f"The file {full_path} does not exist.")
        data = np.load(full_path)
        subset = file.split('_')[1]
        if subset in subsets:
            raise ValueError(f"Subset {subset} already processed.")
        subsets.append(subset)
        subset_tags_nc = all_tags_nc[subset] if subset in all_tags_nc else []
        tags_this_file = data['tags']
        mask = ~np.isin(tags_this_file, subset_tags_nc)

        # Apply the mask to the errors
        errors_e_tot = data[energy_tag][mask]*HARTREE_TO_EV
        fde = data[f'fde{tail_info}'][mask]*HARTREE_TO_EV
        dde = data[f'dde{tail_info}'][mask]*HARTREE_TO_EV

        if len(subset_tags_nc) > 0:
            print(f"Mean before: {subset}: {np.mean(np.abs(data[energy_tag]*HARTREE_TO_EV)):.4f} eV, after removing {len(subset_tags_nc)} non-converged calculations: {np.mean(np.abs(errors_e_tot)):.4f} eV")

        mean_etot_err.append(np.mean(np.abs(errors_e_tot)))
        mean_fde.append(np.mean(np.abs(fde)))
        mean_dde.append(np.mean(np.abs(dde)))
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(subsets))
    width = 0.4
    ax.bar(x - width/2, mean_fde, width, label='Mean Functional-driven error (fde)')
    ax.bar(x - width/2, mean_dde, width, label='Mean Density-driven error (dde)', bottom=mean_fde)
    ax.bar(x + width/2, mean_etot_err, width, label='Mean Total energy error')
    ax.set_xticks(x, subsets, rotation=90)
    ax.set_ylabel('Mean Error (eV)')
    ax.set_title(f'Mean Errors across subsets using {method} method')
    if log_scale:
        ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.show()

def __get_tags_for_non_converged(error_file, method):
    # We will want the info between []
    tags_with_nonconverged = {}
    pattern = r"\[(.*?)\]"
    analised_subsets = []
    with open(error_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('[Warning'):
            info = re.findall(pattern, line)[0]
            info = info.split(' ')[1]
            subset, idx, subidx = info.split(':')
            if subset not in analised_subsets:
                analised_subsets.append(subset)
                tags_with_nonconverged[subset] = []
            tag = f"{idx}_{subidx}_{method}"
            print(f"Warning found: {tag}")
            tags_with_nonconverged[subset].append(tag)
    return tags_with_nonconverged