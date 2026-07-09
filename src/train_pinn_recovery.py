import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import SEED, RAW_DIR, PROCESSED_DIR, TABLES_DIR, DT, GAMMA

torch.manual_seed(SEED)
np.random.seed(SEED)

class StandardNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        out = self.net(x)
        b_hat = out[:, 0:1]
        phi_hat = out[:, 1:2]
        return b_hat, phi_hat

def train_standard_nn(x, y, epochs=1200):
    model = StandardNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

        if epoch % 300 == 0:
            print(f"Standard NN epoch {epoch}, loss={loss.item():.6f}")

    return model

def train_pinn(x, y_b, y_p, epochs=1800, lambda_phys=0.35):
    model = PINN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    for epoch in range(epochs):
        opt.zero_grad()

        b_hat, phi_hat = model(x)

        p_hat = 0.5 * (1 - torch.cos(phi_hat))

        data_loss = mse(b_hat, y_b) + mse(p_hat, y_p)

        dphi = phi_hat[1:] - phi_hat[:-1]
        b_mid = b_hat[:-1]
        physics_residual = dphi / DT - GAMMA * b_mid
        physics_loss = torch.mean(physics_residual ** 2)

        loss = data_loss + lambda_phys * physics_loss

        loss.backward()
        opt.step()

        if epoch % 300 == 0:
            print(
                f"PINN epoch {epoch}, total={loss.item():.6f}, "
                f"data={data_loss.item():.6f}, physics={physics_loss.item():.6f}"
            )

    return model

def main():
    df = pd.read_csv(RAW_DIR / "quantum_magnetometer_data.csv")

    features = df[["t", "P_noisy"]].values
    target_b = df[["B_disturbed"]].values
    target_p = df[["P_noisy"]].values

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_scaled = scaler_x.fit_transform(features)
    y_scaled = scaler_y.fit_transform(target_b)

    x = torch.tensor(x_scaled, dtype=torch.float32)
    y_b = torch.tensor(y_scaled, dtype=torch.float32)
    y_p = torch.tensor(target_p, dtype=torch.float32)

    print("Training standard neural network baseline...")
    standard_model = train_standard_nn(x, y_b)

    print("Training physics-informed neural network...")
    pinn_model = train_pinn(x, y_b, y_p)

    with torch.no_grad():
        nn_scaled = standard_model(x).numpy()
        pinn_scaled, phi_hat = pinn_model(x)
        pinn_scaled = pinn_scaled.numpy()
        phi_hat = phi_hat.numpy().reshape(-1)

    nn_pred = scaler_y.inverse_transform(nn_scaled).reshape(-1)
    pinn_pred = scaler_y.inverse_transform(pinn_scaled).reshape(-1)

    df["NN_recovered"] = nn_pred
    df["PINN_recovered"] = pinn_pred
    df["PINN_phi_hat"] = phi_hat

    out_path = PROCESSED_DIR / "pinn_recovery_results.csv"
    df.to_csv(out_path, index=False)

    metrics = []
    for name, pred in [
        ("Standard Neural Network", nn_pred),
        ("Physics-Informed Neural Network", pinn_pred)
    ]:
        rmse = np.sqrt(mean_squared_error(df["B_disturbed"], pred))
        mae = mean_absolute_error(df["B_disturbed"], pred)
        metrics.append({
            "Method": name,
            "RMSE": rmse,
            "MAE": mae
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_path = TABLES_DIR / "neural_recovery_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Saved recovery results to {out_path}")
    print(f"Saved neural metrics to {metrics_path}")
    print(metrics_df)

if __name__ == "__main__":
    main()
