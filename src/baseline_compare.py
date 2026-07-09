import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import PROCESSED_DIR, TABLES_DIR, DT, GAMMA

def simple_kalman(z, q=1e-4, r=1e-2):
    x = 0.0
    p = 1.0
    out = []

    for measurement in z:
        # Prediction
        p = p + q

        # Update
        k = p / (p + r)
        x = x + k * (measurement - x)
        p = (1 - k) * p

        out.append(x)

    return np.array(out)

def estimate_b_from_probability(p_noisy):
    # Approximate inverse of P = [1 - cos(phi)] / 2
    # This is intentionally imperfect and used as a classical baseline.
    phi_est = np.arccos(np.clip(1 - 2 * p_noisy, -1, 1))
    b_est = np.gradient(phi_est, DT) / GAMMA
    return b_est

def main():
    df = pd.read_csv(PROCESSED_DIR / "pinn_recovery_results.csv")

    b_true = df["B_disturbed"].values
    p_noisy = df["P_noisy"].values

    b_prob_est = estimate_b_from_probability(p_noisy)

    moving_average = uniform_filter1d(b_prob_est, size=25)
    kalman = simple_kalman(b_prob_est)

    df["Moving_Average"] = moving_average
    df["Kalman_Filter"] = kalman

    methods = {
        "Moving Average": moving_average,
        "Kalman Filter": kalman,
        "Standard Neural Network": df["NN_recovered"].values,
        "Physics-Informed Neural Network": df["PINN_recovered"].values
    }

    rows = []
    signal_power = np.mean(b_true ** 2)

    for name, pred in methods.items():
        error = b_true - pred
        rmse = np.sqrt(mean_squared_error(b_true, pred))
        mae = mean_absolute_error(b_true, pred)
        noise_power = np.mean(error ** 2)
        snr_db = 10 * np.log10(signal_power / noise_power)

        rows.append({
            "Method": name,
            "RMSE": rmse,
            "MAE": mae,
            "Recovered_SNR_dB": snr_db
        })

    metrics = pd.DataFrame(rows).sort_values("RMSE")
    metrics_path = TABLES_DIR / "baseline_comparison_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    out_path = PROCESSED_DIR / "all_recovery_results.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved full comparison results to {out_path}")
    print(f"Saved metrics to {metrics_path}")
    print(metrics)

if __name__ == "__main__":
    main()
