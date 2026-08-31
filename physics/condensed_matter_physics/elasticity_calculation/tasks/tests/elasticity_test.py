import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solution.elastic_pipeline import (
    step_03_calculate_unit_cell_volume,
    step_12_assemble_cubic_cij_matrix,
    step_13_verify_matrix_symmetry,
    step_14_check_born_stability_criteria,
    step_20_compute_vrh_averages,
    step_25_calculate_debye_temperature
)

def test_unit_cell_volume():
    """Ensure unit cell volume calculation is strictly positive and matches expected scale."""
    lattice_matrix = np.array([[3.5, 0, 0], [0, 3.5, 0], [0, 0, 3.5]])
    v0 = step_03_calculate_unit_cell_volume(lattice_matrix)
    assert v0 == pytest.approx(42.875, rel=1e-3)
    assert v0 > 0

def test_born_stability():
    """Verify Born stability criteria for cubic systems (C11 - C12 > 0, C11 + 2C12 > 0, C44 > 0)."""
    c11, c12, c44 = 230.0, 135.0, 117.0  # Typical values for iron (GPa)
    assert step_14_check_born_stability_criteria(c11, c12, c44) is True

def test_cij_symmetry():
    """Verify assembled elastic stiffness matrix is symmetric."""
    C = step_12_assemble_cubic_cij_matrix(230.0, 135.0, 117.0)
    assert step_13_verify_matrix_symmetry(C) is True

def test_voigt_reuss_hill_bounds():
    """Verify Voigt upper bound is greater than or equal to Reuss lower bound."""
    kV, kR = 166.67, 166.67  # For cubic systems, KV == KR
    gV, gR = 82.5, 75.2     # GV >= GR always
    k_vrh, g_vrh = step_20_compute_vrh_averages(kV, kR, gV, gR)
    
    assert gV >= gR
    assert g_vrh == pytest.approx(78.85, rel=1e-2)

def test_debye_temperature_range():
    """Check that final calculated Debye temperature falls within physically realistic bounds."""
    # Assuming average sound velocity vm ~ 3500 m/s and V0 ~ 23 A^3
    theta_d = step_25_calculate_debye_temperature(3500.0, 23.5)
    assert 100.0 < theta_d < 1000.0  # Physical sanity check for solid metals/ceramics
