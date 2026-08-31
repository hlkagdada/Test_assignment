import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_CSV = "/app/data/DFT_ENERGY_STRAINS.csv"
POSCAR_PATH = "/app/data/POSCAR_UNITCELL"
OUTPUT_DIR = "/app/output"

EV_TO_JOULE = 1.602176634e-19
ANG3_TO_M3 = 1.0e-30
AMU_TO_KG = 1.66053906660e-27
GPA_TO_PA = 1.0e9

ATOMIC_MASSES = {
    "H": 1.008, "He": 4.002602,
    "Li": 6.94, "Be": 9.0121831, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998403163, "Ne": 20.1797,
    "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815385, "Si": 28.085,
    "P": 30.973761998, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.0983, "Ca": 40.078, "Sc": 44.955908, "Ti": 47.867,
    "V": 50.9415, "Cr": 51.9961, "Mn": 54.938044, "Fe": 55.845,
    "Co": 58.933194, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.921595, "Se": 78.971,
    "Br": 79.904, "Kr": 83.798,
    "Rb": 85.4678, "Sr": 87.62, "Y": 88.90584, "Zr": 91.224,
    "Nb": 92.90637, "Mo": 95.95, "Tc": 98.0, "Ru": 101.07,
    "Rh": 102.90550, "Pd": 106.42, "Ag": 107.8682, "Cd": 112.414,
    "In": 114.818, "Sn": 118.710, "Sb": 121.760, "Te": 127.60,
    "I": 126.90447, "Xe": 131.293,
    "Cs": 132.90545196, "Ba": 137.327, "La": 138.90547, "Ce": 140.116,
    "Pr": 140.90766, "Nd": 144.242, "Pm": 145.0, "Sm": 150.36,
    "Eu": 151.964, "Gd": 157.25, "Tb": 158.92535, "Dy": 162.500,
    "Ho": 164.93033, "Er": 167.259, "Tm": 168.93422, "Yb": 173.045,
    "Lu": 174.9668, "Hf": 178.49, "Ta": 180.94788, "W": 183.84,
    "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084,
    "Au": 196.966569, "Hg": 200.592, "Tl": 204.38, "Pb": 207.2,
    "Bi": 208.98040, "Po": 209.0, "At": 210.0, "Rn": 222.0,
    "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.0377,
    "Pa": 231.03588, "U": 238.02891,
}


def read_poscar(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    scale = float(lines[1])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)])

    #in some of DFT codes such as vasp gives negative scale
    if scale > 0:
        lattice = scale * lattice
    elif scale < 0:
        target_volume = abs(scale)
        raw_volume = abs(np.linalg.det(lattice))
        lattice = lattice * (target_volume / raw_volume) ** (1.0 / 3.0)
    else:
        raise ValueError("POSCAR scale factor cannot be zero.")

    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    if len(species) != len(counts):
        raise ValueError("POSCAR species/count entries are inconsistent.")

    return lines, lattice, species, counts


def unit_cell_volume(lattice):
    return float(abs(np.linalg.det(lattice)))


def mass_density(species, counts, volume_a3):
    try:
        total_mass_amu = sum(
            ATOMIC_MASSES[element] * count
            for element, count in zip(species, counts)
        )
    except KeyError as exc:
        raise ValueError(f"No atomic mass available for element {exc.args[0]}.")

    return total_mass_amu * AMU_TO_KG / (volume_a3 * ANG3_TO_M3)


def load_dft_dataset(path):
    df = pd.read_csv(path)
    required = {"strain_pattern", "eta", "energy_ev"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    expected = {"e1", "e2", "e3", "e4", "e5", "e6", "e1_e2", "e1_e3", "e2_e3"}
    found = set(df["strain_pattern"])
    missing_patterns = expected - found
    if missing_patterns:
        raise ValueError(f"CSV missing strain patterns: {sorted(missing_patterns)}")
    return df


def fit_curvatures(df):
    curvatures = {}
    coefficients = {}
    for pattern, group in df.groupby("strain_pattern"):
        x = group["eta"].to_numpy(dtype=float)
        y = group["energy_ev"].to_numpy(dtype=float)
        if len(x) < 5:
            raise ValueError(f"Need at least five points for a fourth-order fit: {pattern}")
        coeff = np.polyfit(x, y, 4)
        coefficients[pattern] = coeff.tolist()
        # np.polyfit returns [a4, a3, a2, a1, a0].
        curvatures[pattern] = float(2.0 * coeff[2])
    return curvatures, coefficients


def build_orthorhombic_cij(curvatures, volume_a3):
    # E(eta) curvature is in eV. For E = E0 + 1/2*V*C*eta^2,
    # C[GPa] = curvature[eV] * (eV/Ang^3 -> GPa) / V[Ang^3].
    ev_per_a3_to_gpa = (EV_TO_JOULE / ANG3_TO_M3) / GPA_TO_PA
    factor = ev_per_a3_to_gpa / volume_a3

    c11 = curvatures["e1"] * factor
    c22 = curvatures["e2"] * factor
    c33 = curvatures["e3"] * factor
    c44 = curvatures["e4"] * factor
    c55 = curvatures["e5"] * factor
    c66 = curvatures["e6"] * factor

    c12 = 0.5 * (
        curvatures["e1_e2"] - curvatures["e1"] - curvatures["e2"]
    ) * factor
    c13 = 0.5 * (
        curvatures["e1_e3"] - curvatures["e1"] - curvatures["e3"]
    ) * factor
    c23 = 0.5 * (
        curvatures["e2_e3"] - curvatures["e2"] - curvatures["e3"]
    ) * factor

    C = np.array([
        [c11, c12, c13, 0.0, 0.0, 0.0],
        [c12, c22, c23, 0.0, 0.0, 0.0],
        [c13, c23, c33, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c44, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, c55, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, c66],
    ])
    return C


def check_stability(C):
    eigenvalues = np.linalg.eigvalsh(C)
    if np.min(eigenvalues) <= 0:
        raise ValueError(
            f"Elastic stiffness matrix is not positive definite. "
            f"Minimum eigenvalue = {np.min(eigenvalues):.8g} GPa."
        )
    return eigenvalues


def vrh_properties(C):
    S = np.linalg.inv(C)

    c11, c22, c33 = C[0, 0], C[1, 1], C[2, 2]
    c12, c13, c23 = C[0, 1], C[0, 2], C[1, 2]
    c44, c55, c66 = C[3, 3], C[4, 4], C[5, 5]

    b_v = (
        c11 + c22 + c33
        + 2.0 * (c12 + c13 + c23)
    ) / 9.0

    g_v = (
        c11 + c22 + c33
        - c12 - c13 - c23
        + 3.0 * (c44 + c55 + c66)
    ) / 15.0

    b_r = 1.0 / (
        S[0, 0] + S[1, 1] + S[2, 2]
        + 2.0 * (S[0, 1] + S[0, 2] + S[1, 2])
    )

    g_r = 15.0 / (
        4.0 * (S[0, 0] + S[1, 1] + S[2, 2])
        - 4.0 * (S[0, 1] + S[0, 2] + S[1, 2])
        + 3.0 * (S[3, 3] + S[4, 4] + S[5, 5])
    )

    b_vrh = 0.5 * (b_v + b_r)
    g_vrh = 0.5 * (g_v + g_r)
    nu = (3.0 * b_vrh - 2.0 * g_vrh) / (
        2.0 * (3.0 * b_vrh + g_vrh)
    )
    anisotropy_au = 5.0 * g_v / g_r + b_v / b_r - 6.0

    return {
        "S_ij": S,
        "B_V": b_v,
        "B_R": b_r,
        "B_VRH": b_vrh,
        "G_V": g_v,
        "G_R": g_r,
        "G_VRH": g_vrh,
        "nu_VRH": nu,
        "Anisotropy_AU": anisotropy_au,
    }


def acoustic_properties(b_vrh, g_vrh, rho, volume_a3, num_atoms):
    vl = np.sqrt((b_vrh + (4.0 / 3.0) * g_vrh) * GPA_TO_PA / rho)
    vt = np.sqrt(g_vrh * GPA_TO_PA / rho)
    vm = (
        (1.0 / 3.0) * (1.0 / vl**3 + 2.0 / vt**3)
    ) ** (-1.0 / 3.0)

    number_density = num_atoms / (volume_a3 * ANG3_TO_M3)
    theta_d = (
        (6.62607015e-34 / 1.380649e-23)
        * vm
        * ((3.0 * number_density) / (4.0 * np.pi)) ** (1.0 / 3.0)
    )

    return {
        "v_l": float(vl),
        "v_t": float(vt),
        "v_m": float(vm),
        "Debye_temperature": float(theta_d),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _, lattice, species, counts = read_poscar(POSCAR_PATH)
    volume_a3 = unit_cell_volume(lattice)
    rho = mass_density(species, counts, volume_a3)
    df = load_dft_dataset(DATA_CSV)

    curvatures, coefficients = fit_curvatures(df)
    C = build_orthorhombic_cij(curvatures, volume_a3)
    eigenvalues = check_stability(C)
    props = vrh_properties(C)
    acoustic = acoustic_properties(
        props["B_VRH"],
        props["G_VRH"],
        rho,
        volume_a3,
        sum(counts),
    )

    result = {
        "V0_A3": volume_a3,
        "density_kg_m3": rho,
        "C_ij": C.tolist(),
        "S_ij": props["S_ij"].tolist(),
        "B_V": props["B_V"],
        "B_R": props["B_R"],
        "B_VRH": props["B_VRH"],
        "G_V": props["G_V"],
        "G_R": props["G_R"],
        "G_VRH": props["G_VRH"],
        "nu_VRH": props["nu_VRH"],
        "Anisotropy_AU": props["Anisotropy_AU"],
        "v_l": acoustic["v_l"],
        "v_t": acoustic["v_t"],
        "v_m": acoustic["v_m"],
        "Debye_temperature": acoustic["Debye_temperature"],
        "fit_curvatures_eV": curvatures,
        "fit_coefficients": coefficients,
        "C_eigenvalues_GPa": eigenvalues.tolist(),
        "num_atoms": int(sum(counts)),
    }

    output_path = os.path.join(OUTPUT_DIR, "elastic_properties.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
