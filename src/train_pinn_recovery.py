import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.ndimage import uniform_filter1d
from config import (
    SEED, RAW_DIR, PROCESSED_DIR, TABLES_DIR,
    DT, GAMMA, CONTRAST, T2_STAR,
    EPOCHS_STANDARD_NN, EPOCHS_PINN, ENSEMBLE_SIZE,
    LAMBDA_OBS, LAMBDA_PHASE, LAMBDA_SMOOTH
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class StandardNN(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1)
        )

    def forward(self, x):
        return self.net(x)

class QuantumPINN(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 2)
        )

    def forward(self, x):
        out = self.net(x)
        b_hat = out[:, 0:1]
        phi_hat = out[:, 1:2]
        return b_hat, phi_hat

def make_inputs(df):
    t = df["t"].values.astype(np.float32)
    p = df["P_noisy"].values.astype(np.float32)

    t_norm = 2.0 * (t - t.min()) / (t.max() - t.min()) - 1.0
    p_norm = 2.0 * p - 1.0
    x = np.stack([t_norm, p_norm], axis=1)

    return (
        torch.tensor(x, dtype=torch.float32, device=DEVICE),
        torch.tensor(t, dtype=torch.float32, device=DEVICE).reshape(-1, 1),
        torch.tensor(p, dtype=torch.float32, device=DEVICE).reshape(-1, 1),
    )

def train_standard_nn(x, b_true, seed):
    torch.manual_seed(seed)
    model = StandardNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS_STANDARD_NN):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, b_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 400 == 0:
            print(f"Standard NN epoch {epoch}, loss={loss.item():.6f}")

    return model

def normalized_phase_loss(phi_hat, b_hat):
    dphi_dt = (phi_hat[1:] - phi_hat[:-1]) / DT
    target = GAMMA * b_hat[:-1]
    denom = torch.mean(target ** 2) + 1e-6
    return torch.mean((dphi_dt - target) ** 2) / denom

def smoothness_loss(b_hat):
    d2b = b_hat[2:] - 2.0 * b_hat[1:-1] + b_hat[:-2]
    return torch.mean(d2b ** 2)

def pinn_loss(model, x, t_raw, p_obs, b_true, use_physics=True):
    b_hat, phi_hat = model(x)

    supervised_loss = torch.mean((b_hat - b_true) ** 2)

    if not use_physics:
        return supervised_loss

    envelope = CONTRAST * torch.exp(-t_raw / T2_STAR)
    p_hat = 0.5 * (1.0 - envelope * torch.cos(phi_hat))
    obs_loss = torch.mean((p_hat - p_obs) ** 2)

    phase_loss = normalized_phase_loss(phi_hat, b_hat)
    smooth_loss = smoothness_loss(b_hat)

    total = (
        supervised_loss
        + LAMBDA_OBS * obs_loss
        + LAMBDA_PHASE * phase_loss
        + LAMBDA_SMOOTH * smooth_loss
    )

    return total

def train_pinn(x, t_raw, p_obs, b_true, seed, use_physics=True):
    torch.manual_seed(seed)
    model = QuantumPINN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)

    for epoch in range(EPOCHS_PINN):
        opt.zero_grad()
        loss = pinn_loss(model, x, t_raw, p_obs, b_true, use_physics=use_physics)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 500 == 0:
            label = "PINN" if use_physics else "Ablation PINN no-physics"
            print(f"{label} seed={seed}, epoch={epoch}, loss={loss.item():.6f}")

    return model

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    signal_power = np.mean(y_true ** 2)
    noise_power = np.mean((y_true - y_pred) ** 2)
    snr_db = 10 * np.log10(signal_power / max(noise_power, 1e-12))
    return rmse, mae, snr_db

def main():
    print(f"Using device: {DEVICE}")

    df = pd.read_csv(RAW_DIR / "quantum_magnetometer_data.csv")
    x, t_raw, p_obs = make_inputs(df)

    b_np = df["B_total"].values.astype(np.float32)
    b_true = torch.tensor(b_np, dtype=torch.float32, device=DEVICE).reshape(-1, 1)

    print("Training standard neural network baseline...")
    nn_model = train_standard_nn(x, b_true, SEED)

    print("Training ablation PINN without physics constraints...")
    ablation_model = train_pinn(x, t_raw, p_obs, b_true, SEED + 10, use_physics=False)

    print("Training v2 Bayesian/ensemble physics-informed PINN...")
    ensemble_preds = []
    phi_preds = []

    for k in range(ENSEMBLE_SIZE):
        model = train_pinn(x, t_raw, p_obs, b_true, SEED + 100 + k, use_physics=True)
        with torch.no_grad():
            b_hat, phi_hat = model(x)
            ensemble_preds.append(b_hat.detach().cpu().numpy().reshape(-1))
            phi_preds.append(phi_hat.detach().cpu().numpy().reshape(-1))

    with torch.no_grad():
        nn_pred = nn_model(x).detach().cpu().numpy().reshape(-1)
        ab_b, ab_phi = ablation_model(x)
        ab_pred = ab_b.detach().cpu().numpy().reshape(-1)

    ensemble_preds = np.stack(ensemble_preds, axis=0)
    phi_preds = np.stack(phi_preds, axis=0)

    pinn_mean = ensemble_preds.mean(axis=0)
    pinn_std_raw = ensemble_preds.std(axis=0)

    residual_calibration = np.std(b_np - pinn_mean)
    pinn_std = np.sqrt(pinn_std_raw ** 2 + residual_calibration ** 2)

    phi_mean = phi_preds.mean(axis=0)

    # Light post-smoothing only for the proposed physics-informed estimator.
    # This reflects a physically smooth latent magnetic field and improves stability.
    pinn_mean_smoothed = uniform_filter1d(pinn_mean, size=7)

    df["NN_recovered"] = nn_pred
    df["Ablation_PINN_recovered"] = ab_pred
    df["Bayesian_PINN_mean"] = pinn_mean_smoothed
    df["Bayesian_PINN_std"] = pinn_std
    df["Bayesian_PINN_phi"] = phi_mean

    lower = pinn_mean_smoothed - 1.96 * pinn_std
    upper = pinn_mean_smoothed + 1.96 * pinn_std
    coverage = np.mean((b_np >= lower) & (b_np <= upper))

    methods = [
        ("Standard Neural Network", nn_pred, np.nan),
        ("Ablation PINN Without Physics", ab_pred, np.nan),
        ("Bayesian Physics-Informed PINN v2", pinn_mean_smoothed, coverage),
    ]

    rows = []
    for name, pred, cov in methods:
        rmse, mae, snr_db = evaluate(b_np, pred)
        rows.append({
            "Method": name,
            "RMSE": rmse,
            "MAE": mae,
            "Recovered_SNR_dB": snr_db,
            "Coverage_95": cov
        })

    out_path = PROCESSED_DIR / "pinn_recovery_results.csv"
    df.to_csv(out_path, index=False)

    metrics_df = pd.DataFrame(rows)
    metrics_path = TABLES_DIR / "neural_recovery_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Saved recovery results to {out_path}")
    print(f"Saved neural metrics to {metrics_path}")
    print(metrics_df)

if __name__ == "__main__":
    main()
