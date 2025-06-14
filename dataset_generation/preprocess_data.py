import numpy as np
import pandas as pd
import cantera as ct

def compute_progress_variable(Y, species_names):
    # Pick weights for key species
    weights = {
        "CO": 1.0,
        "CO2": 1.0,
        "H2O": 1.0,
        "OH": 1.0
    }
    C = np.zeros(Y.shape[0])
    for species, w in weights.items():
        if species in species_names:
            idx = species_names.index(species)
            C += w * Y[:, idx]
    return C

def main():
    data = np.load("data/raw/flamelet_case1.npz", allow_pickle=True)
    z = data["z"]
    T = data["T"]
    Y = data["species"]
    if Y.shape[0] != len(z):
        Y = Y.T  # Transpose if needed
    species_names = list(data["species_names"])

    # Compute progress variable
    C = compute_progress_variable(Y, species_names)

    # Load gas object for property lookup
    gas = ct.Solution("gri30.yaml")

    Z = np.linspace(0, 1, len(z))  # placeholder, you'll compute real Z later
    wdot_C = np.gradient(C, z.flatten())  # simplistic ω̇C

    # Compute thermo properties
    Cp, mu, kappa = [], [], []
    for i in range(len(z)):
        gas.TPY = T[i], ct.one_atm, Y[i, :]
        Cp.append(gas.cp_mass)
        mu.append(gas.viscosity)
        kappa.append(gas.thermal_conductivity)

    df = pd.DataFrame({
        "Z": Z,
        "C": C,
        "omega_C": wdot_C,
        "Cp": Cp,
        "mu": mu,
        "kappa": kappa
    })

    df.to_csv("data/processed/flamelet_dataset.csv", index=False)
    print("✅ Processed data saved.")

if __name__ == "__main__":
    main()
