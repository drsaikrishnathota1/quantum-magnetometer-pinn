import pandas as pd
import matplotlib.pyplot as plt
from config import PROCESSED_DIR, FIGURES_DIR, TABLES_DIR

def main():
    df = pd.read_csv(PROCESSED_DIR / "all_recovery_results.csv")
    metrics = pd.read_csv(TABLES_DIR / "baseline_comparison_metrics.csv")

    plt.figure(figsize=(10, 4))
    plt.plot(df["t"], df["P_clean"], label="Decohered clean quantum probability")
    plt.plot(df["t"], df["P_noisy"], label="Shot-noise-limited measurement", alpha=0.65)
    plt.xlabel("Time")
    plt.ylabel("Excited-state probability")
    plt.title("Ramsey Quantum Magnetometer Measurement with Decoherence and Shot Noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure1_quantum_measurement_probability.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["t"], df["B_total"], label="True hidden magnetic field", linewidth=2)
    plt.plot(df["t"], df["Kalman_Filter"], label="Kalman filter", alpha=0.70)
    plt.plot(df["t"], df["Ablation_PINN_recovered"], label="Ablation PINN without physics", alpha=0.75)
    plt.plot(df["t"], df["Bayesian_PINN_mean"], label="Bayesian physics-informed PINN", alpha=0.90)
    plt.xlabel("Time")
    plt.ylabel("Normalized magnetic field")
    plt.title("Hidden Magnetic-Field Recovery Under Stochastic Disturbance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure2_recovery_comparison.png", dpi=300)
    plt.close()

    lower = df["Bayesian_PINN_mean"] - 1.96 * df["Bayesian_PINN_std"]
    upper = df["Bayesian_PINN_mean"] + 1.96 * df["Bayesian_PINN_std"]

    plt.figure(figsize=(10, 4))
    plt.plot(df["t"], df["B_total"], label="True hidden field", linewidth=2)
    plt.plot(df["t"], df["Bayesian_PINN_mean"], label="Bayesian PINN mean")
    plt.fill_between(df["t"], lower, upper, alpha=0.25, label="Approx. 95% uncertainty band")
    plt.xlabel("Time")
    plt.ylabel("Normalized magnetic field")
    plt.title("Bayesian PINN Recovery with Uncertainty Band")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure3_uncertainty_band.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(metrics["Method"], metrics["RMSE"])
    plt.ylabel("RMSE")
    plt.title("Baseline and Ablation Recovery Error")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure4_rmse_ablation.png", dpi=300)
    plt.close()

    print("Saved upgraded paper figures in results/figures/")
    print(metrics)

if __name__ == "__main__":
    main()
