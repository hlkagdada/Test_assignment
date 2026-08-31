import json

OUTPUT_PATH = "/app/output/elastic_properties.json"


def main():
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        r = json.load(f)

    print("=" * 72)
    print("FIRST-PRINCIPLES ELASTICITY SUMMARY")
    print("=" * 72)
    print(f"Unit-cell volume        : {r['V0_A3']:.4f} Å^3")
    print(f"Mass density            : {r['density_kg_m3']:.2f} kg/m^3")
    print()
    print("Orthorhombic stiffness matrix C_ij (GPa):")
    for row in r["C_ij"]:
        print("  " + " ".join(f"{x:12.4f}" for x in row))
    print()
    print(f"B_V / B_R / B_VRH      : {r['B_V']:.4f} / {r['B_R']:.4f} / {r['B_VRH']:.4f} GPa")
    print(f"G_V / G_R / G_VRH      : {r['G_V']:.4f} / {r['G_R']:.4f} / {r['G_VRH']:.4f} GPa")
    print(f"Poisson ratio (VRH)     : {r['nu_VRH']:.6f}")
    print(f"Universal anisotropy AU : {r['Anisotropy_AU']:.6f}")
    print()
    print(f"Longitudinal velocity   : {r['v_l']:.2f} m/s")
    print(f"Transverse velocity     : {r['v_t']:.2f} m/s")
    print(f"Average velocity        : {r['v_m']:.2f} m/s")
    print(f"Debye temperature       : {r['Debye_temperature']:.2f} K")
    print("=" * 72)


if __name__ == "__main__":
    main()
