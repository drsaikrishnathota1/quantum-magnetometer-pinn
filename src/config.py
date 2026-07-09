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

# Normalized quantum sensor simulation parameters
N_SAMPLES = 2500
DT = 0.01
GAMMA = 2.0  # normalized gyromagnetic scaling factor
SHOT_COUNT = 200
