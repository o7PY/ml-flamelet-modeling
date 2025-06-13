import cantera as ct
import numpy as np
import os

gas = ct.Solution("gri30.yaml")  # or your mechanism like nDodecane if available
width = 0.02  # 2 cm domain
n_points = 200

fuel = 'CH4:1.0'
oxidizer = 'O2:1.0, N2:3.76'
p = ct.one_atm
T_fuel = 300.0
T_oxidizer = 300.0

flame = ct.CounterflowDiffusionFlame(gas, width=width)
flame.fuel_inlet.mdot = 0.12
flame.fuel_inlet.T = T_fuel
flame.fuel_inlet.X = fuel
flame.oxidizer_inlet.mdot = 0.12
flame.oxidizer_inlet.T = T_oxidizer
flame.oxidizer_inlet.X = oxidizer

flame.set_refine_criteria(ratio=3.0, slope=0.06, curve=0.12)
flame.solve(loglevel=1, auto=True)

# Save the data
output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)

np.savez(f"{output_dir}/flamelet_case1.npz",
         z=flame.grid,
         T=flame.T,
         species=flame.Y,
         species_names=flame.gas.species_names)

print("Flamelet generated and saved.")
