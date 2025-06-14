import os
import numpy as np
import pandas as pd
import cantera as ct

def compute_progress_variable(Y, species_names):
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

def preprocess_flamelet(file_path):
    data = np.load(file_path, allow_pickle=True)
    z = data["z"].squeeze()
    T = data["T"]
    Y = data["species"]
    species_names = list(data["species_names"])

    if Y.shape[0] != len(z):
        Y = Y.T

    C = compute_progress_variable(Y, species_names)
    if len(C) != len(z):
        min_len = min(len(C), len(z))
        C = C[:min_len]
        z = z[:min_len]
        Y = Y[:min_len]
        T = T[:min_len]

    # Calculate omega_C and apply log10 transform
    omega_C = np.gradient(C, z)
    omega_C_clipped = np.clip(omega_C, 1e-12, None)  # avoid log(0)
    log_omega_C = np.log10(omega_C_clipped)

    # Thermophysical properties
    gas = ct.Solution("gri30.yaml")
    Cp, mu, kappa = [], [], []

    for i in range(len(z)):
        gas.TPY = T[i], ct.one_atm, Y[i, :]
        Cp.append(gas.cp_mass)
        mu.append(gas.viscosity)
        kappa.append(gas.thermal_conductivity)

    # Extract flamelet parameters from filename
    name = os.path.basename(file_path)
    parts = name.replace(".npz", "").split("_")
    T_inlet = int(parts[1][1:])
    P_bar = int(parts[2][1:])
    mdot = int(parts[3][1:]) / 1000.0

    df = pd.DataFrame({
        "Z": np.linspace(0, 1, len(C)),  # placeholder
        "C": C,
        "log_omega_C": log_omega_C,
        "Cp": Cp,
        "mu": mu,
        "kappa": kappa,
        "T_inlet": T_inlet,
        "P_bar": P_bar,
        "mdot": mdot
    })

    return df

def main():
    flamelet_dir = "data/raw"
    output_csv = "data/processed/big_dataset.csv"
    os.makedirs("data/processed", exist_ok=True)

    flamelet_files = [os.path.join(flamelet_dir, f) for f in os.listdir(flamelet_dir) if f.endswith(".npz")]

    full_df = pd.DataFrame()

    for file in flamelet_files:
        try:
            df = preprocess_flamelet(file)
            full_df = pd.concat([full_df, df], ignore_index=True)
            print(f"✅ Processed: {file}")
        except Exception as e:
            print(f"❌ Failed: {file} — {e}")

    full_df.to_csv(output_csv, index=False)
    print(f"\n🔥 Done! Combined dataset saved to {output_csv}")
    print(f"🧮 Total rows: {len(full_df)}")

if __name__ == "__main__":
    main()
