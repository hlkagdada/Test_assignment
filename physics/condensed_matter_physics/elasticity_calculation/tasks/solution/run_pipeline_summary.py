import pandas as pd
import numpy as np
import os

from elastic_pipeline import (
    POSCAR_PATH, DATA_CSV, OUTPUT_DIR,
    step_01_read_poscar, step_02_extract_lattice_vectors, step_03_calculate_unit_cell_volume,
    step_04_load_dft_dataset, step_05_convert_units_ev_to_gpa,
    step_08_extract_c11_plus_c12_curvature, step_09_extract_c11_minus_c12_curvature, step_10_extract_c44_curvature,
    step_11_compute_raw_elastic_constants, step_12_assemble_cubic_cij_matrix, step_13_verify_matrix_symmetry,
    step_14_check_born_stability_criteria, step_15_compute_compliance_matrix,
    step_16_compute_voigt_bulk_modulus, step_17_compute_reuss_bulk_modulus,
    step_18_compute_voigt_shear_modulus, step_19_compute_reuss_shear_modulus,
    step_20_compute_vrh_averages, step_21_calculate_mass_density,
    step_22_compute_longitudinal_sound_velocity, step_23_compute_transverse_sound_velocity,
    step_24_compute_average_sound_velocity, step_25_calculate_debye_temperature
)

def main():
    lines = step_01_read_poscar(POSCAR_PATH)
    lat_mat = step_02_extract_lattice_vectors(lines)
    v0 = step_03_calculate_unit_cell_volume(lat_mat)
    df_dft = step_04_load_dft_dataset(DATA_CSV)
    conv_factor = step_05_convert_units_ev_to_gpa(v0)

    curv_11_12 = step_08_extract_c11_plus_c12_curvature(df_dft)
    curv_11_sub_12 = step_09_extract_c11_minus_c12_curvature(df_dft)
    curv_44 = step_10_extract_c44_curvature(df_dft)

    c11, c12, c44 = step_11_compute_raw_elastic_constants(curv_11_12, curv_11_sub_12, curv_44, conv_factor)
    C_mat = step_12_assemble_cubic_cij_matrix(c11, c12, c44)
    step_13_verify_matrix_symmetry(C_mat)
    is_stable = step_14_check_born_stability_criteria(c11, c12, c44)
    S_mat = step_15_compute_compliance_matrix(C_mat)

    kV = step_16_compute_voigt_bulk_modulus(c11, c12)
    kR = step_17_compute_reuss_bulk_modulus(S_mat)
    gV = step_18_compute_voigt_shear_modulus(c11, c12, c44)
    gR = step_19_compute_reuss_shear_modulus(S_mat)
    k_vrh, g_vrh = step_20_compute_vrh_averages(kV, kR, gV, gR)

    rho = step_21_calculate_mass_density(lines, v0)
    vl = step_22_compute_longitudinal_sound_velocity(k_vrh, g_vrh, rho)
    vt = step_23_compute_transverse_sound_velocity(g_vrh, rho)
    vm = step_24_compute_average_sound_velocity(vl, vt)
    theta_d = step_25_calculate_debye_temperature(vm, v0)

    # Print Summary
    summary = f"""
======================================================================
                           SUMMARY REPORT                 
======================================================================

[Crystallographic & Density Information]
  - Unit Cell Volume (V0)     : {v0:.4f} Å^3
  - Mass Density (rho)        : {rho:.2f} kg/m^3

[Elastic Stiffness Constants (C_ij)]
  - C11                       : {c11:.2f} GPa
  - C12                       : {c12:.2f} GPa
  - C44                       : {c44:.2f} GPa
  - Mechanical Stability Check : {'PASSED (Born Criteria Met)' if is_stable else 'FAILED'}

[Elastic Moduli (Voigt-Reuss-Hill)]
  - Bulk Modulus (K_V / K_R)   : {kV:.2f} GPa / {kR:.2f} GPa
  - Bulk Modulus (K_VRH)       : {k_vrh:.2f} GPa
  - Shear Modulus (G_V / G_R)  : {gV:.2f} GPa / {gR:.2f} GPa
  - Shear Modulus (G_VRH)      : {g_vrh:.2f} GPa

[Sound Velocities]
  - Longitudinal Velocity (vl) : {vl:.2f} m/s
  - Transverse Velocity (vt)   : {vt:.2f} m/s
  - Average Velocity (vm)      : {vm:.2f} m/s

[Final Debye Properties]
  - Debye Temperature (Theta_D): {theta_d:.2f} K
======================================================================
"""
    print(summary)

if __name__ == "__main__":
    main()
