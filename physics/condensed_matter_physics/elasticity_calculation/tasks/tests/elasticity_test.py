import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/app/data/DFT_ENERGY_STRAINS.csv")
POSCAR = Path("/app/data/POSCAR_UNITCELL")
OUTPUT = Path("/app/output/elastic_properties.json")

MASSES = {
    "Mg": 24.305,
    "Si": 28.085,
    "O": 15.999,
}
EV_TO_J = 1.602176634e-19
ANG3_TO_M3 = 1e-30
AMU_TO_KG = 1.66053906660e-27


def read_poscar():
    lines = [x.strip() for x in POSCAR.read_text().splitlines() if x.strip()]
    scale = float(lines[1])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)])
    lattice *= scale
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    return lattice, species, counts


def independent_reference():
    lattice, species, counts = read_poscar()
    volume = abs(np.linalg.det(lattice))

    total_mass = sum(MASSES[s] * n for s, n in zip(species, counts))
    rho = total_mass * AMU_TO_KG / (volume * ANG3_TO_M3)

    df = pd.read_csv(DATA)
    curv = {}
    for pattern, g in df.groupby("strain_pattern"):
        coeff = np.polyfit(g["eta"].to_numpy(), g["energy_ev"].to_numpy(), 4)
        curv[pattern] = 2.0 * coeff[2]

    factor = (EV_TO_J / ANG3_TO_M3) / 1e9 / volume

    c11 = curv["e1"] * factor
    c22 = curv["e2"] * factor
    c33 = curv["e3"] * factor
    c44 = curv["e4"] * factor
    c55 = curv["e5"] * factor
    c66 = curv["e6"] * factor
    c12 = 0.5 * (curv["e1_e2"] - curv["e1"] - curv["e2"]) * factor
    c13 = 0.5 * (curv["e1_e3"] - curv["e1"] - curv["e3"]) * factor
    c23 = 0.5 * (curv["e2_e3"] - curv["e2"] - curv["e3"]) * factor

    C = np.array([
        [c11, c12, c13, 0, 0, 0],
        [c12, c22, c23, 0, 0, 0],
        [c13, c23, c33, 0, 0, 0],
        [0, 0, 0, c44, 0, 0],
        [0, 0, 0, 0, c55, 0],
        [0, 0, 0, 0, 0, c66],
    ])

    S = np.linalg.inv(C)
    b_v = (c11 + c22 + c33 + 2 * (c12 + c13 + c23)) / 9
    g_v = (c11 + c22 + c33 - c12 - c13 - c23 + 3 * (c44 + c55 + c66)) / 15
    b_r = 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
    g_r = 15 / (
        4 * (S[0, 0] + S[1, 1] + S[2, 2])
        - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
        + 3 * (S[3, 3] + S[4, 4] + S[5, 5])
    )
    b = 0.5 * (b_v + b_r)
    g = 0.5 * (g_v + g_r)
    nu = (3 * b - 2 * g) / (2 * (3 * b + g))
    au = 5 * g_v / g_r + b_v / b_r - 6

    vl = math.sqrt((b + 4 * g / 3) * 1e9 / rho)
    vt = math.sqrt(g * 1e9 / rho)
    vm = ((1 / 3) * (1 / vl**3 + 2 / vt**3)) ** (-1 / 3)
    n_atoms = sum(counts)
    number_density = n_atoms / (volume * ANG3_TO_M3)
    theta = (6.62607015e-34 / 1.380649e-23) * vm * ((3 * number_density) / (4 * math.pi)) ** (1 / 3)

    return {
        "volume": volume,
        "rho": rho,
        "C": C,
        "B_VRH": b,
        "G_VRH": g,
        "nu": nu,
        "AU": au,
        "vm": vm,
        "theta": theta,
    }


def test_output_file_exists():
    assert OUTPUT.exists(), "elastic_properties.json was not created"


def test_output_matches_independent_reference():
    ref = independent_reference()
    result = json.loads(OUTPUT.read_text())

    assert math.isclose(result["V0_A3"], ref["volume"], rel_tol=1e-6, abs_tol=1e-8)
    assert math.isclose(result["density_kg_m3"], ref["rho"], rel_tol=1e-6)

    C = np.asarray(result["C_ij"], dtype=float)
    assert C.shape == (6, 6)
    assert np.allclose(C, ref["C"], rtol=2e-5, atol=2e-5)

    assert math.isclose(result["B_VRH"], ref["B_VRH"], rel_tol=2e-5)
    assert math.isclose(result["G_VRH"], ref["G_VRH"], rel_tol=2e-5)
    assert math.isclose(result["nu_VRH"], ref["nu"], rel_tol=2e-5)
    assert math.isclose(result["Anisotropy_AU"], ref["AU"], rel_tol=2e-5)
    assert math.isclose(result["v_m"], ref["vm"], rel_tol=2e-5)
    assert math.isclose(result["Debye_temperature"], ref["theta"], rel_tol=2e-5)


def test_stiffness_matrix_is_symmetric_and_stable():
    result = json.loads(OUTPUT.read_text())
    C = np.asarray(result["C_ij"], dtype=float)
    assert np.allclose(C, C.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(C) > 0)


def test_required_output_keys():
    result = json.loads(OUTPUT.read_text())
    required = {
        "V0_A3", "density_kg_m3", "C_ij",
        "B_VRH", "G_VRH", "nu_VRH",
        "Anisotropy_AU", "v_m", "Debye_temperature",
    }
    assert required.issubset(result.keys())
