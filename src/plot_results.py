import pandas as pd
import matplotlib.pyplot as plt
from config import PROCESSED_DIR, FIGURES_DIR, TABLES_DIR

def main():
    df = pd.read_csv(PROCESSED_DIR / "all_recovery_results.csv")
    metrics = pd.read_csv(TABLES_DIR / "baseline_comparison_metrics.csv")

    # Figure 1: quantum measurement probability
    plt.figure(figsize=(10, 4))
    plt.plot(df["t"], df["P_clean"], label="Clean quantum probability")
    plt.plot(df["t"], df["P_noisy"], label="Noisy measured probability", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Measurement probability")
    plt.title("Quantum Magnetometer Measurement Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "quantum_measurement_probability.png", dpi=300)
    plt.close()

    # Figure 2: recovery comparison
    plt.figure(figsize=(10, 4))
    plt.plot(df["t"], df["B_disturbed"], label="True disturbed magnetic field", linewidth=2)
    plt.plot(df["t"], df["Kalman_Filter"], label="Kalman filter", alpha=0.75)
    plt.plot(df["t"], df["NN_recovered"], label="Standard NN", alpha=0.75)
    plt.plot(df["t"], df["PINN_recovered"], label="Physics-informed NN", alpha=0.85)
    plt.xlabel("Time")
    plt.ylabel("Normalized magnetic field")
    plt.title("Magnetic-Field Recovery Under Noise and Disturbance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "magnetic_field_recovery_comparison.png", dpi=300)
    plt.close()

    # Figure 3: RMSE bar chart
    plt.figure(figsize=(8, 4))
    plt.bar(metrics["Method"], metrics["RMSE"])
    plt.ylabel("RMSE")
    plt.title("Recovery Error Comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rmse_comparison.png", dpi=300)
    plt.close()

    print("Saved figures:")
    print(FIGURES_DIR / "quantum_measurement_probability.png")
    print(FIGURES_DIR / "magnetic_field_recovery_comparison.png")
    print(FIGURES_DIR / "rmse_comparison.png")

if __name__ == "__main__":
    main()
