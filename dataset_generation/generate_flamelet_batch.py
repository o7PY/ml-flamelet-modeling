# scripts/generate_flamelet_batch.py

from generate_flamelets import generate_flamelet
import numpy as np

temps = np.linspace(800, 1600, 5)         # 5 different temperatures
pressures = np.linspace(10, 50, 4) * 1e5  # 10–50 atm in Pa
mdots = np.linspace(0.05, 0.25, 10)       # 10 different mass flux values

total = 0
success = 0

for T in temps:
    for P in pressures:
        for mdot in mdots:
            total += 1
            if generate_flamelet(T_inlet=T, P=P, mdot=mdot):
                success += 1

print(f"\n🔥 DONE: {success}/{total} flamelets generated successfully!")
