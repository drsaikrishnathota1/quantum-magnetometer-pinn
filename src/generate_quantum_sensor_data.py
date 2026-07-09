import numpy as np
import pandas as pd
from config import SEED, RAW_DIR, N_SAMPLES, DT, GAMMA, SHOT_COUNT

np.random.seed(SEED)

def main():
    t = np.arange(N_SAMPLES) * DT

    # Hidden magnetic field B(t), normalized units
    b_true = (
        0.75 * np.sin(2 * np.pi * 0.35 * t)
        + 0.25 * np.sin(2 * np.pi * 1.10 * t + 0.5)
        + 0.10 * np.sin(2 * np.pi * 2.20 * t)
    )

    # Slow drift
    drift = 0.008 * t

    # Spoofing-like magnetic disturbance
    spoof = np.zeros_like(t)
    spoof_region = (t > 13.0) & (t < 16.0)
    spoof[spoof_region] = 0.65 * np.exp(-((t[spoof_region] - 14.5) ** 2) / 0.22)

    b_disturbed = b_true + drift + spoof

    # Quantum phase accumulation
    phi = np.cumsum(GAMMA * b_disturbed * DT)

    # Two-level measurement probability
    p_clean = 0.5 * (1 - np.cos(phi))

    # Shot noise using binomial sampling
    shot_counts = np.random.binomial(SHOT_COUNT, np.clip(p_clean, 0, 1))
    p_shot = shot_counts / SHOT_COUNT

    # Additional Gaussian sensor noise
    p_noisy = p_shot + np.random.normal(0, 0.025, size=N_SAMPLES)
    p_noisy = np.clip(p_noisy, 0, 1)

    df = pd.DataFrame({
        "t": t,
        "B_true": b_true,
        "B_disturbed": b_disturbed,
        "drift": drift,
        "spoof": spoof,
        "phi": phi,
        "P_clean": p_clean,
        "P_noisy": p_noisy
    })

    out_path = RAW_DIR / "quantum_magnetometer_data.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved dataset to {out_path}")
    print(df.head())

if __name__ == "__main__":
    main()
