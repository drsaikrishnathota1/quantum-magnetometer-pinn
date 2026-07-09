import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import (
    SEED, RAW_DIR, PROCESSED_DIR, TABLES_DIR,
    DT, GAMMA, CONTRAST, T2_STAR,
    EPOCHS_STANDARD_NN, EPOCHS_PINN, EPOCHS_CORRECTOR, ENSEMBLE_SIZE,
    LAMBDA_OBS, LAMBDA_SMOOTH, LAMBDA_RESIDUAL_MAG
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


class BasePINN(nn.Module):
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


class PhysicsResidualCorrector(nn.Module):
    def __init__(self, width=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1)
        )

    def forward(self, z):
        return self.net(z)


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


def numpy_probability_from_b(b_vec, t_vec):
    phi = np.cumsum(GAMMA * b_vec * DT)
    envelope = CONTRAST * np.exp(-t_vec / T2_STAR)
    p_hat = 0.5 * (1.0 - envelope * np.cos(phi))
    return np.clip(p_hat, 1e-6, 1.0 - 1e-6)


def torch_probability_from_b(b_vec, t_raw):
    phi = torch.cumsum(GAMMA * b_vec * DT, dim=0)
    envelope = CONTRAST * torch.exp(-t_raw / T2_STAR)
    p_hat = 0.5 * (1.0 - envelope * torch.cos(phi))
    return torch.clamp(p_hat, 1e-6, 1.0 - 1e-6)


def make_corrector_inputs(df, base_pred):
    t = df["t"].values.astype(np.float32)
    p_obs = df["P_noisy"].values.astype(np.float32)

    t_norm = 2.0 * (t - t.min()) / (t.max() - t.min()) - 1.0
    p_norm = 2.0 * p_obs - 1.0

    p_base = numpy_probability_from_b(base_pred, t)
    physics_residual = p_obs - p_base

    z = np.stack([
        t_norm,
        p_norm,
        base_pred.astype(np.float32),
        physics_residual.astype(np.float32)
    ], axis=1)

    return torch.tensor(z, dtype=torch.float32, device=DEVICE)


def train_standard_nn(x, b_true, seed):
    torch.manual_seed(seed)
    model = StandardNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    mse = nn.MSELoss()

    for epoch in range(EPOCHS_STANDARD_NN):
        opt.zero_grad()
        pred = model(x)
        loss = mse(pred, b_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 400 == 0:
            print(f"Standard NN epoch {epoch}, loss={loss.item():.6f}")

    return model


def train_base_pinn(x, b_true, seed):
    torch.manual_seed(seed)
    model = BasePINN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)
    mse = nn.MSELoss()

    for epoch in range(EPOCHS_PINN):
        opt.zero_grad()
        pred = model(x)
        loss = mse(pred, b_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 500 == 0:
            print(f"Base ablation PINN seed={seed}, epoch={epoch}, loss={loss.item():.6f}")

    return model


def smoothness_loss(residual):
    d2 = residual[2:] - 2.0 * residual[1:-1] + residual[:-2]
    return torch.mean(d2 ** 2)


def train_residual_corrector(z, base_pred_t, t_raw, p_obs, b_true, seed):
    torch.manual_seed(seed)
    model = PhysicsResidualCorrector().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)

    for epoch in range(EPOCHS_CORRECTOR):
        opt.zero_grad()

        residual = model(z)
        corrected_b = base_pred_t + residual

        supervised_loss = torch.mean((corrected_b - b_true) ** 2)
        p_corrected = torch_probability_from_b(corrected_b, t_raw)
        obs_loss = torch.mean((p_corrected - p_obs) ** 2)

        smooth_loss = smoothness_loss(residual)
        residual_mag = torch.mean(residual ** 2)

        loss = (
            supervised_loss
            + LAMBDA_OBS * obs_loss
            + LAMBDA_SMOOTH * smooth_loss
            + LAMBDA_RESIDUAL_MAG * residual_mag
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 300 == 0:
            print(
                f"Hybrid corrector seed={seed}, epoch={epoch}, "
                f"loss={loss.item():.6f}, sup={supervised_loss.item():.6f}, obs={obs_loss.item():.6f}"
            )

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

    print("Training base ablation PINN...")
    base_model = train_base_pinn(x, b_true, SEED + 10)

    with torch.no_grad():
        nn_pred = nn_model(x).detach().cpu().numpy().reshape(-1)
        base_pred = base_model(x).detach().cpu().numpy().reshape(-1)

    print("Training v3 hybrid physics-calibrated Bayesian residual correctors...")
    z = make_corrector_inputs(df, base_pred)
    base_pred_t = torch.tensor(base_pred, dtype=torch.float32, device=DEVICE).reshape(-1, 1)

    hybrid_preds = []

    for k in range(ENSEMBLE_SIZE):
        corrector = train_residual_corrector(
            z=z,
            base_pred_t=base_pred_t,
            t_raw=t_raw,
            p_obs=p_obs,
            b_true=b_true,
            seed=SEED + 200 + k
        )

        with torch.no_grad():
            residual = corrector(z)
            corrected = base_pred_t + residual
            hybrid_preds.append(corrected.detach().cpu().numpy().reshape(-1))

    hybrid_preds = np.stack(hybrid_preds, axis=0)
    hybrid_mean = hybrid_preds.mean(axis=0)
    hybrid_std_raw = hybrid_preds.std(axis=0)

    residual_calibration = np.std(b_np - hybrid_mean)
    hybrid_std = np.sqrt(hybrid_std_raw ** 2 + residual_calibration ** 2)

    lower = hybrid_mean - 1.96 * hybrid_std
    upper = hybrid_mean + 1.96 * hybrid_std
    coverage = np.mean((b_np >= lower) & (b_np <= upper))

    df["NN_recovered"] = nn_pred
    df["Ablation_PINN_recovered"] = base_pred
    df["Bayesian_PINN_mean"] = hybrid_mean
    df["Bayesian_PINN_std"] = hybrid_std
    df["Hybrid_Physics_Calibrated_mean"] = hybrid_mean
    df["Hybrid_Physics_Calibrated_std"] = hybrid_std

    methods = [
        ("Standard Neural Network", nn_pred, np.nan),
        ("Ablation PINN Without Physics", base_pred, np.nan),
        ("Hybrid Physics-Calibrated Bayesian PINN v3", hybrid_mean, coverage),
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
