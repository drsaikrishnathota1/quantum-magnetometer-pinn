#!/bin/bash

source .venv/bin/activate

echo "Step 1: Generate quantum sensor data"
python src/generate_quantum_sensor_data.py

echo "Step 2: Train standard NN and PINN"
python src/train_pinn_recovery.py

echo "Step 3: Compare with classical baselines"
python src/baseline_compare.py

echo "Step 4: Generate plots"
python src/plot_results.py

echo "Done. Check results/figures and results/tables."
