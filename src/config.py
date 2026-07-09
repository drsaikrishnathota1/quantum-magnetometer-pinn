from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

for path in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Normalized computational physics simulation
N_SAMPLES = 4096
DT = 0.01
GAMMA = 2.0

# Ramsey quantum magnetometer parameters
SHOT_COUNT = 256
CONTRAST = 0.86
T2_STAR = 60.0

# Stochastic magnetic disturbance: Ornstein-Uhlenbeck process
OU_THETA = 1.35
OU_SIGMA = 0.18

# Training settings
EPOCHS_STANDARD_NN = 1400
EPOCHS_PINN = 1800
EPOCHS_CORRECTOR = 1200
ENSEMBLE_SIZE = 5

# v3 hybrid physics-calibrated loss weights
LAMBDA_OBS = 0.05
LAMBDA_PHASE = 0.000
LAMBDA_SMOOTH = 0.001
LAMBDA_RESIDUAL_MAG = 0.0001
