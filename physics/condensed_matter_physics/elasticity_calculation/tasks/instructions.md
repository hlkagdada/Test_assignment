# First-Principles Elasticity and Polycrystalline Property Extraction

You are given a ground-state VASP POSCAR and DFT total-energy calculations for an orthorhombic crystal.

## Input files

- `/app/data/POSCAR_UNITCELL`: ground-state VASP POSCAR containing the lattice vectors, atomic species, and atom counts.
- `/app/data/DFT_ENERGY_STRAINS.csv`: DFT energy-vs-strain data with columns `strain_pattern`, `eta`, and `energy_ev`.

The CSV contains the strain patterns `e1` through `e6` and the combined normal strains `e1_e2`, `e1_e3`, and `e2_e3`.

## Task

Write the required calculations into `/app/output/elastic_properties.json`.

1. Parse the POSCAR and calculate:
   - unit-cell volume `V0` in Å³;
   - mass density `rho` in kg/m³ using the atomic species and counts in the POSCAR.

2. For each of the nine strain patterns, fit the energy-strain data with a fourth-order polynomial
   `E(eta) = a4*eta^4 + a3*eta^3 + a2*eta^2 + a1*eta + a0`.
   Obtain the curvature at zero strain as `d²E/deta² = 2*a2`.

3. Use the energy-strain curvatures to construct the orthorhombic elastic stiffness matrix in GPa. With the strain conventions used by the supplied data:
   - `e1`, `e2`, `e3` provide `C11`, `C22`, `C33`;
   - `e4`, `e5`, `e6` provide `C44`, `C55`, `C66`;
   - the combined normal-strain curves provide the off-diagonal terms:
     - `C12 = (k12 - k1 - k2) / (2*V0)`,
     - `C13 = (k13 - k1 - k3) / (2*V0)`,
     - `C23 = (k23 - k2 - k3) / (2*V0)`,
     where the energy curvatures are converted from eV to GPa·Å³ consistently.
   The resulting matrix is
   ```
   [[C11, C12, C13, 0,   0,   0],
    [C12, C22, C23, 0,   0,   0],
    [C13, C23, C33, 0,   0,   0],
    [0,   0,   0,   C44, 0,   0],
    [0,   0,   0,   0,   C55, 0],
    [0,   0,   0,   0,   0,   C66]]
   ```

4. Invert the stiffness matrix to obtain the compliance matrix.

5. Calculate the orthorhombic Voigt and Reuss bounds and their Hill averages:
   - `B_VRH`;
   - `G_VRH`.

6. Calculate:
   - VRH Poisson's ratio `nu_VRH = (3B - 2G)/(2(3B + G))`;
   - Universal Elastic Anisotropy `Anisotropy_AU = 5*G_V/G_R + B_V/B_R - 6`.

7. Calculate longitudinal and transverse acoustic velocities from the VRH moduli and density, then the isotropic average acoustic velocity:
   `v_m = [ (1/3) * (1/v_l^3 + 2/v_t^3) ]^(-1/3)`.

8. Calculate the Debye temperature using the number of atoms in the POSCAR unit cell.

## Output

Save `/app/output/elastic_properties.json`.

It must contain at least these keys:

```json
{
  "V0_A3": 0.0,
  "density_kg_m3": 0.0,
  "C_ij": [[...]],
  "B_VRH": 0.0,
  "G_VRH": 0.0,
  "nu_VRH": 0.0,
  "Anisotropy_AU": 0.0,
  "v_m": 0.0,
  "Debye_temperature": 0.0
}
```

You may include additional diagnostic quantities such as the fitted strain curvatures, Voigt/Reuss bounds, compliance matrix, and longitudinal/transverse velocities.

Use the supplied input files rather than hard-coding material properties or final numerical answers.
