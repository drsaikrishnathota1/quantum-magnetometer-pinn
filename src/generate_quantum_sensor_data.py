import numpy as np
import pandas as pd
from config import (
    SEED, RAW_DIR, N_SAMPLES, DT, GAMMA,
    SHOT_COUNT, CONTRAST, T2_STAR, OU_THETA, OU_SIGMA
)

np.random.seed(SEED)

def generate_ou_process(n, dt, theta, sigma):
    eta = np.zeros(n)
    for i in range(1, n):
        dW = np.sqrt(dt) * np.random.randn()
        eta[i] = eta[i-1] + (-theta * eta[i-1]) * dt + sigma * dW
    return eta

def main():
    t = np.arange(N_SAMPLES) * DT

    # Geomagnetic-like smooth field component, normalized units
    b_geo = (
        0.55 * np.sin(2 * np.pi * 0.18 * t)
        + 0.22 * np.sin(2 * np.pi * 0.71 * t + 0.40)
        + 0.08 * np.sin(2 * np.pi * 1.65 * t + 1.20)
    )

    # Slow sensor/environmental drift
    drift = 0.004 * t + 0.04 * np.sin(2 * np.pi * 0.035 * t)

    # Stochastic magnetic disturbance using Ornstein-Uhlenbeck physics
    ou_noise = generate_ou_process(N_SAMPLES, DT, OU_THETA, OU_SIGMA)

    # Spoofing-like localized magnetic transient
    spoof = np.zeros_like(t)
    spoof_window = (t > 24.0) & (t < 29.0)
    spoof[spoof_window] = (
        0.55 * np.exp(-((t[spoof_window] - 26.3) ** 2) / 0.85)
        - 0.22 * np.exp(-((t[spoof_window] - 27.6) ** 2) / 0.30)
    )

    # Hidden magnetic field to recover
    b_total = b_geo + drift + ou_noise + spoof

    # Quantum phase accumulation
    phi = np.cumsum(GAMMA * b_total * DT)

    # Ramsey quantum magnetometer probability with decoherence
    envelope = CONTRAST * np.exp(-t / T2_STAR)
    p_clean = 0.5 * (1.0 - envelope * np.cos(phi))
    p_clean = np.clip(p_clean, 1e-6, 1 - 1e-6)

    # Binomial shot noise from finite quantum measurement shots
    counts = np.random.binomial(SHOT_COUNT, p_clean)
    p_shot = counts / SHOT_COUNT

    # Additional small readout noise
    p_noisy = p_shot + np.random.normal(0.0, 0.012, size=N_SAMPLES)
    p_noisy = np.clip(p_noisy, 0.0, 1.0)

    df = pd.DataFrame({
        "t": t,
        "B_geo": b_geo,
        "B_drift": drift,
        "B_ou": ou_noise,
        "B_spoof": spoof,
        "B_total": b_total,
        "phi": phi,
        "decoherence_envelope": envelope,
        "P_clean": p_clean,
        "P_noisy": p_noisy,
        "shot_counts": counts
    })

    out_path = RAW_DIR / "quantum_magnetometer_data.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved upgraded quantum magnetometer dataset to {out_path}")
    print(df.head())

if __name__ == "__main__":
    main()
