# Hybrid Physics-Calibrated Recovery of Decohered Quantum Magnetometer Signals Under Stochastic Magnetic Disturbance

**Sai Krishna Thota**  
Independent Researcher, USA  
Email: drsaikrishnathota@ieee.org

## Abstract

Recovering weak magnetic-field dynamics from decohered quantum sensor measurements is an important inverse problem in computational physics. This microarticle presents a hybrid physics-calibrated ensemble recovery method for reconstructing a hidden magnetic-field trajectory from noisy Ramsey-type quantum magnetometer measurements under stochastic magnetic disturbance. The simulated measurement model includes Ramsey interference, finite contrast, exponential dephasing, shot-noise-limited readout, and Ornstein–Uhlenbeck magnetic perturbations. A supervised neural estimator is first used to obtain a baseline field recovery, after which a physics-calibrated residual corrector uses the Ramsey probability residual to refine the inferred magnetic field and estimate uncertainty. In the final computational experiment, the proposed hybrid method achieved RMSE = 0.2297, MAE = 0.1795, recovered SNR = 5.5795 dB, and 95% uncertainty coverage = 0.9534, outperforming the ablation neural estimator, standard neural network, moving-average recovery, Kalman filtering, and direct probability inversion baselines. The result demonstrates that a compact physics-calibrated correction layer can improve inverse recovery of decohered quantum magnetometer signals while preserving calibrated uncertainty.

**Keywords:** quantum magnetometry; Ramsey measurement; decoherence; stochastic disturbance; inverse recovery; computational physics

## 1. Introduction

Quantum magnetometers infer weak magnetic-field variations from phase-sensitive quantum measurements. In practical sensing conditions, the recovered signal may be degraded by decoherence, finite readout contrast, shot noise, and stochastic magnetic disturbance. These effects make magnetic-field reconstruction an inverse problem rather than a direct measurement task. Recent studies have examined noise-resilient quantum sensor networks, machine-learning-assisted quantum magnetometry, dephasing-tolerant quantum sensing, and Bayesian quantum estimation strategies [4–8]. Related computational physics work has also used machine learning for phase extraction, transport prediction, and optical sensing problems [1–3].

However, a compact computational comparison for recovering a decohered Ramsey-type magnetometer signal using a physics-residual correction layer under stochastic magnetic disturbance remains less explored. This microarticle addresses that gap by comparing a hybrid physics-calibrated ensemble recovery method against direct probability inversion, moving-average filtering, Kalman filtering, a standard neural network, and an ablation neural estimator without physics calibration.

## 2. Physical model

The normalized magnetic field is modeled as a slowly varying signal disturbed by stochastic Ornstein–Uhlenbeck perturbations. The quantum phase accumulated by the Ramsey sensor is

\[
\phi(t) = \int_0^t \gamma B(\tau) d\tau,
\]

where \(B(t)\) is the hidden magnetic field and \(\gamma\) is the normalized gyromagnetic coefficient. The decohered Ramsey measurement probability is

\[
P(t) = \frac{1}{2}\left[1 - C e^{-t/T_2^*}\cos(\phi(t))\right],
\]

where \(C\) is the measurement contrast and \(T_2^*\) is the dephasing time. Noisy measurements are generated using finite-shot binomial sampling and additive readout noise. The computational experiment used 4096 normalized samples, \(C = 0.86\), \(T_2^* = 60.0\), shot count \(N_s = 256\), normalized gyromagnetic coefficient \(\gamma = 2.0\), time step \(\Delta t = 0.01\), and an Ornstein–Uhlenbeck disturbance with \(	heta = 1.35\) and \(\sigma = 0.18\). These values define a reproducible normalized quantum-sensing simulation rather than a device-specific calibration.

## 3. Hybrid physics-calibrated recovery method

The proposed method uses a two-stage recovery process. First, a supervised baseline neural estimator recovers the magnetic field from normalized time and noisy Ramsey probability. Second, a residual corrector receives the baseline magnetic-field estimate and the Ramsey probability residual. This corrector learns a small physics-calibrated adjustment to the baseline field estimate. An ensemble of residual correctors provides the mean estimate and an uncertainty band.

The method is physics-calibrated rather than purely physics-constrained. This distinction is important because directly forcing a physics loss during training degraded point-recovery accuracy in preliminary experiments. The final method instead uses the Ramsey measurement residual as a correction signal, allowing the learned estimator to retain low error while improving physical consistency and uncertainty calibration. The uncertainty band should therefore be interpreted as ensemble-calibrated uncertainty, not as a full closed-form Bayesian posterior.

## 4. Results and discussion

The final experiment used 4096 simulated quantum magnetometer samples. The proposed hybrid physics-calibrated Bayesian PINN v3 achieved the best point-recovery performance among all tested methods.

| Method | RMSE | MAE | Recovered SNR (dB) | 95% Coverage |
|---|---:|---:|---:|---:|
| Hybrid Physics-Calibrated Ensemble PINN v3 | 0.2297 | 0.1795 | 5.5795 | 0.9534 |
| Ablation Neural Estimator Without Physics Calibration | 0.2860 | 0.2242 | 3.6776 | — |
| Standard Neural Network | 0.2882 | 0.2246 | 3.6104 | — |
| Moving Average | 0.4945 | 0.3635 | -1.0793 | — |
| Kalman Filter | 0.5394 | 0.4137 | -1.8333 | — |
| Probability Inversion | 2.4405 | 1.9410 | -14.9454 | — |

Figure 1 illustrates the noisy Ramsey measurement probability used as the input observation. Figure 2 compares recovered magnetic-field trajectories, Figure 3 shows the uncertainty band for the proposed ensemble method, and Figure 4 summarizes the RMSE comparison across methods. The hybrid method reduced RMSE relative to the ablation neural estimator and standard neural network while maintaining approximately calibrated 95% uncertainty coverage. Direct probability inversion performed poorly because the inverse cosine operation amplifies noise and phase ambiguity under decoherence. Classical smoothing and Kalman filtering improved stability but could not match the learned recovery methods.

The main limitation is that the study is a normalized computational simulation rather than an experimental quantum magnetometer demonstration. The result should therefore be interpreted as a reproducible computational physics proof-of-concept for Ramsey-signal inverse recovery under controlled stochastic disturbance. Experimental validation with calibrated laboratory quantum-sensor data is a natural next step.

## 5. Conclusion

This microarticle presented a compact computational physics result for recovering a hidden magnetic-field trajectory from decohered Ramsey-type quantum magnetometer measurements. A hybrid physics-calibrated ensemble residual correction method achieved lower RMSE than neural and classical baselines while preserving approximately calibrated uncertainty. The result suggests that physics-calibrated residual learning is a useful strategy for inverse recovery in noisy quantum sensing simulations.

## Data and code availability

The simulation code, trained-recovery outputs, result tables, and generated figures are available in the public GitHub repository associated with this work: https://github.com/drsaikrishnathota1/quantum-magnetometer-pinn.

## References

[1] J. Jeon, Y. Kim, Y. Ito, N. Sugita, and M. Mitsuishi, “One-frame interferometric surface contouring via stepwise phase extraction and deep learning,” Results in Physics, vol. 80, Article 108560, 2026. doi:10.1016/j.rinp.2025.108560.

[2] G. Ramasekhar et al., “Machine learning approach for predicting heat and mass transfer in Maxwell–Sutterby fluid flow over a Riga plate geometry via Levenberg–Marquardt backpropagation neural network algorithm: Influence of Joule heating and activation energy,” Results in Physics, vol. 80, Article 108528, 2026. doi:10.1016/j.rinp.2025.108528.

[3] B. Roumi and V. Fallahi, “Tunable dual-phase TMOKE sensor in a Weyl semimetal–plasmonic waveguide,” Results in Physics, vol. 80, Article 108571, 2026. doi:10.1016/j.rinp.2025.108571.

[4] A. Hamann, P. Aigner, W. Dür, P. Sekatski, “Selective and noise-resilient wave estimation with quantum sensor networks,” Quantum Science and Technology, vol. 10, no. 3, Article 035028, 2025. doi:10.1088/2058-9565/add61b.

[5] I. Jauch, T. Strohm, T. Fuchs, F. Jelezko, “Quantum magnetometry enhanced by machine learning,” Quantum Science and Technology, vol. 11, no. 1, Article 015055, 2026. doi:10.1088/2058-9565/ae3acf.

[6] M. Mezzadri, L. Lepori, A. Chiesa, S. Carretta, “Dephasing-tolerant quantum sensing for transverse magnetic fields with spin qudits,” Quantum Science and Technology, vol. 10, no. 1, Article 015045, 2025. doi:10.1088/2058-9565/ad985e.

[7] V. Cimini, E. Polino, M. Valeri, N. Spagnolo et al., “Benchmarking Bayesian quantum estimation,” Quantum Science and Technology, vol. 9, no. 3, Article 035035, 2024. doi:10.1088/2058-9565/ad48b3.

[8] E. L. André, J. Bavaresco, M. Mehboudi, “Strategy optimization for Bayesian quantum parameter estimation with finite copies: Adaptive greedy, parallel, sequential, and general strategies,” Quantum Science and Technology, 2026. doi:10.1088/2058-9565/ae846c.
