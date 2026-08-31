### First-Principles Elasticity & Polycrystalline Property Extraction ###

You are provided with ground-state crystal structure data and energy-vs-strain DFT calculations for an orthorhombic crystal.

### Input Files ###

- `/app/data/POSCAR_UNITCELL`: Ground-state VASP POSCAR file containing unit cell vectors and atomic species.
- `/app/data/DFT_ENERGY_STRAINS.csv`: CSV file containing columns `strain_pattern`, `eta` (strain magnitude), and `energy_ev` (total energy in eV).


### Task Requirements ###

1. Extract the ground-state unit cell volume ($V_0$ in Å³) and density ($\rho$ in kg/m³) from `/app/data/POSCAR_UNITCELL`.

2. Fit energy-strain curves to 4th-order polynomials to derive the second derivative $\frac{d^2E}{d\eta^2}$ at $\eta=0$ for each strain pattern (`e1` through `e6`, and combination strains `e1_e2`, `e1_e3`, `e2_e3`).

3. Construct the full $6 \times 6$ elastic stiffness matrix $C_{ij}$ in GPa.

4. Calculate compliance matrix $S_{ij} = C_{ij}^{-1}$.

5. Derive polycrystalline Voigt-Reuss-Hill (VRH) bulk modulus ($B_{\text{VRH}}$) and shear modulus ($G_{\text{VRH}}$) in GPa, Poisson's ratio ($\nu_{\text{VRH}}$), and Universal Elastic Anisotropy ($A^U$).

6. Compute average acoustic wave velocity ($v_m$ in m/s) and Debye temperature ($\Theta_D$ in K).

### Output Format ###

Save all output parameters in JSON format to `/app/output/elastic_properties.json` with the following key structure:

```json
{
  "C_ij": [[...]], 
  "B_VRH": 0.0,
  "G_VRH": 0.0,
  "nu_VRH": 0.0,
  "Anisotropy_AU": 0.0,
  "v_m": 0.0,
  "Debye_temperature": 0.0
}

