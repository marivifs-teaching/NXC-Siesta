import numpy as np
from pathlib import Path
from modules.reference_functionals import S_prevent0

def read_geometry_in(filename="geometry.in"):
    lattice = []
    atoms = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "lattice_vector":
                lattice.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom_frac":
                atoms.append((parts[4], np.array([float(x) for x in parts[1:4]]), "frac"))
            elif parts[0] == "atom":
                atoms.append((parts[4], np.array([float(x) for x in parts[1:4]]), "cart"))
    return np.array(lattice), atoms


def convert_to_pyscf(lattice, atoms):
    lines = []
    for species, coord, mode in atoms:
        if mode == "frac":
            cart = coord @ lattice
        else:
            cart = coord
        lines.append(f"{species} {cart[0]:.6f} {cart[1]:.6f} {cart[2]:.6f}")
    return lines


def write_pyscf_input(lattice, atom_lines, filename="geometry_pyscf.py"):
    with open(filename, "w") as out:
        out.write("from pyscf.pbc import gto\n\n")
        out.write("cell = gto.Cell()\n")
        out.write("cell.a = [\n")
        for vec in lattice:
            out.write(f"    [{vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f}],\n")
        out.write("]\n")
        out.write("cell.atom = '''\n")
        for line in atom_lines:
            out.write(line + "\n")
        out.write("'''\n")
        out.write("cell.unit = 'Angstrom'\n")
        out.write("cell.basis = 'gth-szv'  # or e.g. 'def2-svp'\n")
        out.write("cell.pseudo = 'gth-pade'\n")
        out.write("cell.build()\n\n")
        out.write("print(cell)\n")
    print(f"Wrote PySCF input: {filename}")


def read_dat_file_AIMS(path):
    file = Path(path) / 'rho_and_derivs_spin_1.dat'
    data = np.loadtxt(file)
    print(f"Read {data.shape} entries from {file}")
    grid_points = data[:, :3]
    weights = data[:, 3]
    rho = data[:, 4]
    gx, gy, gz = data[:, 5], data[:, 6], data[:, 7]
    tau = data[:, 8]
    lapl = data[:, 9]
    gradrho = np.sqrt(gx**2 + gy**2 + gz**2)
    s = S_prevent0(rho, gradrho)
    all_data = {
        'grid_points': grid_points,
        'rho': rho,
        'grho': gradrho,
        's': s,
        'tau': tau,
        'lapl': lapl,
        'weights': weights
    }
    
    return all_data