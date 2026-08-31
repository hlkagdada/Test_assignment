### Solution Approach ###

1. Cell Volume & Density Extraction: Parse `POSCAR_UNITCELL` to compute ground-state volume $V_0$ and density $\rho$.

2. Strain-Energy Fitting: For each strain pattern in `DFT_ENERGY_STRAINS.csv`, fit $E(\eta) = E_0 + a_1\eta + a_2\eta^2 + a_3\eta^3 + a_4\eta^4$. Extract $a_2 = \frac{1}{2} \frac{d^2E}{d\eta^2}$.

3. Elastic Tensor Construction: Convert second derivatives to GPa via $C = \frac{2 a_2}{V_0} \times \frac{1.602176634 \times 10^{-19} \text{ J}}{10^{-30} \text{ m}^3} \times 10^{-9}$. Assign $C_{11}, C_{22}, C_{33}, C_{44}, C_{55}, C_{66}, C_{12}, C_{13}, C_{23}$.

4. Compliance & VRH Bounds: Invert $C$ to get $S = C^{-1}$. Compute $B_V, G_V$ (Voigt) and $B_R, G_R$ (Reuss), then average them to find $B_{\text{VRH}}$ and $G_{\text{VRH}}$.

5. Acoustic Velocities & Debye Temperature: Calculate longitudinal ($v_l$) and transverse ($v_t$) sound speeds to yield mean velocity $v_m$. Derive Debye temperature $\Theta_D$.
