import cantera as ct
import numpy as np
import os

def generate_flamelet(T_inlet=300.0, P=ct.one_atm, mdot=0.12, fuel="CH4:1.0", oxidizer="O2:1.0, N2:3.76",
                      width=0.02, mech="gri30.yaml", save_path="data/raw"):
    gas = ct.Solution(mech)

    flame = ct.CounterflowDiffusionFlame(gas, width=width)
    flame.fuel_inlet.mdot = mdot
    flame.fuel_inlet.T = T_inlet
    flame.fuel_inlet.X = fuel

    flame.oxidizer_inlet.mdot = mdot
    flame.oxidizer_inlet.T = T_inlet
    flame.oxidizer_inlet.X = oxidizer

    flame.P = P
    flame.set_refine_criteria(ratio=3.0, slope=0.06, curve=0.12)

    try:
        flame.solve(loglevel=0, auto=True)
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/flamelet_T{int(T_inlet)}_P{int(P/1e5)}_M{int(mdot*1000)}.npz"
        np.savez(filename,
                 z=flame.grid,
                 T=flame.T,
                 species=flame.Y,
                 species_names=flame.gas.species_names)
        print(f"✅ Saved flamelet: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed: T={T_inlet}, P={P}, mdot={mdot} — {e}")
        return False

# Allow standalone use too
if __name__ == "__main__":
    generate_flamelet()
