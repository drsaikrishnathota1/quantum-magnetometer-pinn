import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import PROCESSED_DIR, TABLES_DIR, DT, GAMMA

def estimate_b_from_probability(p_noisy):
    phi_est = np.arccos(np.clip(1 - 2 * p_noisy, -1, 1))
    phi_est = np.unwrap(phi_est)
    b_est = np.gradient(phi_est, DT) / GAMMA
    return b_est

def simple_kalman(z, q=5e-4, r=5e-2):
    x = 0.0
    p = 1.0
    out = []

    for measurement in z:
        p = p + q
        k = p / (p + r)
        x = x + k * (measurement - x)
        p = (1 - k) * p
        out.append(x)

    return np.array(out)

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    signal_power = np.mean(y_true ** 2)
    noise_power = np.mean((y_true - y_pred) ** 2)
    snr_db = 10 * np.log10(signal_power / max(noise_power, 1e-12))
    return rmse, mae, snr_db

def main():
    df = pd.read_csv(PROCESSED_DIR / "pinn_recovery_results.csv")
    b_true = df["B_total"].values
    p_noisy = df["P_noisy"].values

    probability_inversion = estimate_b_from_probability(p_noisy)
    moving_average = uniform_filter1d(probability_inversion, size=41)
    kalman = simple_kalman(probability_inversion)

    df["Probability_Inversion"] = probability_inversion
    df["Moving_Average"] = moving_average
    df["Kalman_Filter"] = kalman

    methods = {
        "Probability Inversion": probability_inversion,
        "Moving Average": moving_average,
        "Kalman Filter": kalman,
        "Standard Neural Network": df["NN_recovered"].values,
        "Ablation PINN Without Physics": df["Ablation_PINN_recovered"].values,
        "Hybrid Physics-Calibrated Bayesian PINN v3": df["Bayesian_PINN_mean"].values,
    }

    rows = []

    for name, pred in methods.items():
        rmse, mae, snr_db = evaluate(b_true, pred)

        coverage = np.nan
        if name == "Hybrid Physics-Calibrated Bayesian PINN v3":
            std = df["Bayesian_PINN_std"].values
            lower = pred - 1.96 * std
            upper = pred + 1.96 * std
            coverage = np.mean((b_true >= lower) & (b_true <= upper))

        rows.append({
            "Method": name,
            "RMSE": rmse,
            "MAE": mae,
            "Recovered_SNR_dB": snr_db,
            "Coverage_95": coverage
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
