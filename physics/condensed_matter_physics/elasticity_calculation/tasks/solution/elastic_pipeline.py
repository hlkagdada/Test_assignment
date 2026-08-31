import os
import numpy as np
import pandas as pd

# PATHS
DATA_CSV = "/app/data/DFT_ENERGY_STRAINS.csv"
POSCAR_PATH = "/app/data/POSCAR_UNITCELL"
OUTPUT_DIR = "/app/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# STAGE 1: PARSING AND PREPARATION OF THE DATA (Steps 1–5)
# ==============================================================================

def step_01_read_poscar(poscar_path):
    """Step 1: Read POSCAR lines."""
    with open(poscar_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def step_02_extract_lattice_vectors(poscar_lines):
    """Step 2: Parse 3x3 lattice vectors and scale factor."""
    scale = float(poscar_lines[1])
    vecs = np.array([[float(x) for x in poscar_lines[i].split()] for i in range(2, 5)])
    return scale * vecs

def step_03_calculate_unit_cell_volume(lattice_matrix):
    """Step 3: Calculate unit cell volume V0 in Angstrom^3."""
    v0 = float(np.abs(np.dot(lattice_matrix[0], np.cross(lattice_matrix[1], lattice_matrix[2]))))
    assert v0 > 0, "Volume must be positive."
    return v0

def step_04_load_dft_dataset(csv_path):
    """Step 4: Load energy vs. strain CSV data."""
    df = pd.read_csv(csv_path)
    required = {'distortion_type', 'strain', 'energy_ev'}
    assert required.issubset(df.columns), f"CSV missing required columns: {required - set(df.columns)}"
    return df

def step_05_convert_units_ev_to_gpa(v0_ang3):
    """Step 5: Compute conversion factor from eV/Angstrom^3 to GPa."""
    EV_TO_JOULE = 1.602176634e-19
    ANG3_TO_M3 = 1e-30
    GPA_CONVERSION = (EV_TO_JOULE / ANG3_TO_M3) / 1e9
    return GPA_CONVERSION / v0_ang3

# ==============================================================================
# STAGE 2: STRAIN-ENERGY POLYNOMIAL FITTING (Steps 6–10)
# ==============================================================================

def step_06_filter_distortion_data(df, distortion_name):
    """Step 6: Extract subset dataframe for specific distortion pattern."""
    sub_df = df[df['distortion_type'] == distortion_name].sort_values('strain')
    assert len(sub_df) >= 3, f"Insufficient data points for {distortion_name}"
    return sub_df['strain'].values, sub_df['energy_ev'].values

def step_07_fit_quadratic_energy(strains, energies):
    """Step 7: Fit E(delta) = E0 + A*delta + B*delta^2 and extract curvature 2B."""
    coeffs = np.polyfit(strains, energies, 2)
    curvature = 2.0 * coeffs[0]  # Second derivative d^2E/d(delta)^2
    return curvature

def step_08_extract_c11_plus_c12_curvature(df):
    """Step 8: Compute 2nd derivative for hydrostatic/triaxial strain mode."""
    strains, energies = step_06_filter_distortion_data(df, 'hydrostatic')
    return step_07_fit_quadratic_energy(strains, energies)

def step_09_extract_c11_minus_c12_curvature(df):
    """Step 9: Compute 2nd derivative for volume-conserving orthorhombic strain mode."""
    strains, energies = step_06_filter_distortion_data(df, 'orthorhombic')
    return step_07_fit_quadratic_energy(strains, energies)

def step_10_extract_c44_curvature(df):
    """Step 10: Compute 2nd derivative for monoclinic shear strain mode."""
    strains, energies = step_06_filter_distortion_data(df, 'shear')
    return step_07_fit_quadratic_energy(strains, energies)

# ==============================================================================
# STAGE 3: ELASTIC STIFFNESS MATRIX ASSEMBLY & STABILITY (Steps 11–15)
# ==============================================================================

def step_11_compute_raw_elastic_constants(curv_c11_c12, curv_c11_sub_c12, curv_c44, conv_factor):
    """Step 11: Convert second derivatives to GPa elastic constants C11, C12, C44."""
    # Example scaling for cubic symmetry
    c11_plus_c12 = curv_c11_c12 * conv_factor
    c11_minus_c12 = curv_c11_sub_c12 * conv_factor
    
    c11 = 0.5 * (c11_plus_c12 + c11_minus_c12)
    c12 = 0.5 * (c11_plus_c12 - c11_minus_c12)
    c44 = curv_c44 * conv_factor
    return c11, c12, c44

def step_12_assemble_cubic_cij_matrix(c11, c12, c44):
    """Step 12: Build 6x6 Voigt stiffness matrix C_ij."""
    C = np.zeros((6, 6))
    C[0,0] = C[1,1] = C[2,2] = c11
    C[0,1] = C[0,2] = C[1,0] = C[1,2] = C[2,0] = C[2,1] = c12
    C[3,3] = C[4,4] = C[5,5] = c44
    return C

def step_13_verify_matrix_symmetry(C_matrix):
    """Step 13: Assert C_ij matrix is symmetric."""
    assert np.allclose(C_matrix, C_matrix.T, atol=1e-5), "Elastic matrix C_ij is not symmetric!"
    return True

def step_14_check_born_stability_criteria(c11, c12, c44):
    """Step 14: Check Born mechanical stability conditions for cubic systems."""
    cond1 = c11 - c12 > 0
    cond2 = c11 + 2 * c12 > 0
    cond3 = c44 > 0
    assert cond1 and cond2 and cond3, f"Born stability failed: C11-C12={c11-c12}, C11+2C12={c11+2*c12}, C44={c44}"
    return True

def step_15_compute_compliance_matrix(C_matrix):
    """Step 15: Invert stiffness matrix C_ij to obtain compliance matrix S_ij."""
    S = np.linalg.inv(C_matrix)
    return S

# ==============================================================================
# STAGE 4: ELASTIC MODULUS (Steps 16–20)
# ==============================================================================

def step_16_compute_voigt_bulk_modulus(c11, c12):
    """Step 16: Compute Voigt upper bound for Bulk Modulus K_V."""
    return (c11 + 2 * c12) / 3.0

def step_17_compute_reuss_bulk_modulus(S_matrix):
    """Step 17: Compute Reuss lower bound for Bulk Modulus K_R."""
    s11, s12 = S_matrix[0,0], S_matrix[0,1]
    return 1.0 / (3.0 * (s11 + 2 * s12))

def step_18_compute_voigt_shear_modulus(c11, c12, c44):
    """Step 18: Compute Voigt upper bound for Shear Modulus G_V."""
    return (c11 - c12 + 3 * c44) / 5.0

def step_19_compute_reuss_shear_modulus(S_matrix):
    """Step 19: Compute Reuss lower bound for Shear Modulus G_R."""
    s11, s12, s44 = S_matrix[0,0], S_matrix[0,1], S_matrix[3,3]
    return 5.0 / (4.0 * (s11 - s12) + 3.0 * s44)

def step_20_compute_vrh_averages(k_v, k_r, g_v, g_r):
    """Step 20: Compute Hill average for Bulk (K_VRH) and Shear (G_VRH) moduli."""
    k_vrh = 0.5 * (k_v + k_r)
    g_vrh = 0.5 * (g_v + g_r)
    return k_vrh, g_vrh

# ==============================================================================
# STAGE 5: SOUND VELOCITIES & DEBYE TEMPERATURE CALCULATIONS (Steps 21–25)
# ==============================================================================

def step_21_calculate_mass_density(poscar_lines, v0_ang3):
    """Step 21: Extract atomic masses and calculate mass density rho (kg/m^3)."""
    # Example placeholder mass calculation for target material
    total_mass_amu = 55.845 * 2  # e.g., 2 Fe atoms
    AMU_TO_KG = 1.66053906660e-27
    ANG3_TO_M3 = 1e-30
    rho = (total_mass_amu * AMU_TO_KG) / (v0_ang3 * ANG3_TO_M3)
    return rho

def step_22_compute_longitudinal_sound_velocity(k_vrh, g_vrh, rho):
    """Step 22: Compute longitudinal sound velocity v_l (m/s)."""
    # Convert GPa to Pa
    modulus_pa = (k_vrh + (4.0 / 3.0) * g_vrh) * 1e9
    vl = np.sqrt(modulus_pa / rho)
    return vl

def step_23_compute_transverse_sound_velocity(g_vrh, rho):
    """Step 23: Compute transverse sound velocity v_t (m/s)."""
    g_pa = g_vrh * 1e9
    vt = np.sqrt(g_pa / rho)
    return vt

def step_24_compute_average_sound_velocity(vl, vt):
    """Step 24: Compute average sound velocity v_m (m/s)."""
    v_m = ( (1.0 / 3.0) * ( (1.0 / (vl**3)) + (2.0 / (vt**3)) ) ) ** (-1.0 / 3.0)
    return v_m

def step_25_calculate_debye_temperature(v_m, v0_ang3, num_atoms=2, k_vrh=0.0, g_vrh=0.0):
    """Step 25: Calculate Debye Temperature Theta_D (K) and save final results."""
    H_PLANCK = 6.62607015e-34
    K_BOLTZMANN = 1.380649e-23
    
    v0_m3 = v0_ang3 * 1e-30
    number_density = num_atoms / v0_m3
    
    theta_d = (H_PLANCK / K_BOLTZMANN) * v_m * ((3.0 * number_density) / (4.0 * np.pi))**(1.0 / 3.0)
    
    # Save output artifacts
    output_path = os.path.join(OUTPUT_DIR, "elastic_properties.csv")
    results = pd.DataFrame([{
        "debye_temperature_K": theta_d,
        "average_sound_velocity_m_s": v_m,
        "bulk_modulus_vrh_GPa": k_vrh,
        "shear_modulus_vrh_GPa": g_vrh
    }])
    results.to_csv(output_path, index=False)
    return theta_d


# ==============================================================================
# PIPELINE EXECUTION COMMANDS
# ==============================================================================

if __name__ == "__main__":
    # Stage 1
    lines = step_01_read_poscar(POSCAR_PATH)
    lat_mat = step_02_extract_lattice_vectors(lines)
    v0 = step_03_calculate_unit_cell_volume(lat_mat)
    df_dft = step_04_load_dft_dataset(DATA_CSV)
    conv_factor = step_05_convert_units_ev_to_gpa(v0)

    # Stage 2
    curv_11_12 = step_08_extract_c11_plus_c12_curvature(df_dft)
    curv_11_sub_12 = step_09_extract_c11_minus_c12_curvature(df_dft)
    curv_44 = step_10_extract_c44_curvature(df_dft)

    # Stage 3
    c11, c12, c44 = step_11_compute_raw_elastic_constants(curv_11_12, curv_11_sub_12, curv_44, conv_factor)
    C_mat = step_12_assemble_cubic_cij_matrix(c11, c12, c44)
    step_13_verify_matrix_symmetry(C_mat)
    step_14_check_born_stability_criteria(c11, c12, c44)
    S_mat = step_15_compute_compliance_matrix(C_mat)

    # Stage 4
    kV = step_16_compute_voigt_bulk_modulus(c11, c12)
    kR = step_17_compute_reuss_bulk_modulus(S_mat)
    gV = step_18_compute_voigt_shear_modulus(c11, c12, c44)
    gR = step_19_compute_reuss_shear_modulus(S_mat)
    k_vrh, g_vrh = step_20_compute_vrh_averages(kV, kR, gV, gR)

    # Stage 5
    rho = step_21_calculate_mass_density(lines, v0)
    vl = step_22_compute_longitudinal_sound_velocity(k_vrh, g_vrh, rho)
    vt = step_23_compute_transverse_sound_velocity(g_vrh, rho)
    vm = step_24_compute_average_sound_velocity(vl, vt)
    theta_d = step_25_calculate_debye_temperature(vm, v0)
