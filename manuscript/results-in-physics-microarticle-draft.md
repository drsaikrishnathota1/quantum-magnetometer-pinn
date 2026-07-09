# Hybrid Physics-Calibrated Recovery of Decohered Quantum Magnetometer Signals Under Stochastic Magnetic Disturbance

**Sai Krishna Thota**  
Independent Researcher, USA  
Email: drsaikrishnathota@ieee.org

## Abstract

Recovering weak magnetic-field dynamics from decohered quantum sensor measurements is an important inverse problem in computational physics. This microarticle presents a hybrid physics-calibrated Bayesian neural recovery method for reconstructing a hidden magnetic-field trajectory from noisy Ramsey-type quantum magnetometer measurements under stochastic magnetic disturbance. The simulated measurement model includes Ramsey interference, finite contrast, exponential dephasing, shot-noise-limited readout, and Ornstein–Uhlenbeck magnetic perturbations. A supervised neural estimator is first used to obtain a baseline field recovery, after which a physics-calibrated residual corrector uses the Ramsey probability residual to refine the inferred magnetic field and estimate uncertainty. In the final computational experiment, the proposed hybrid method achieved RMSE = 0.2297, MAE = 0.1795, recovered SNR = 5.5795 dB, and 95% uncertainty coverage = 0.9534, outperforming the ablation PINN, standard neural network, moving-average recovery, Kalman filtering, and direct probability inversion baselines. The result demonstrates that a compact physics-calibrated correction layer can improve inverse recovery of decohered quantum magnetometer signals while preserving calibrated uncertainty.

**Keywords:** quantum magnetometry; Ramsey measurement; decoherence; stochastic disturbance; inverse recovery; computational physics

## 1. Introduction

Quantum magnetometers infer weak magnetic-field variations from phase-sensitive quantum measurements. In practical sensing conditions, the recovered signal may be degraded by decoherence, finite readout contrast, shot noise, and stochastic magnetic disturbance. These effects make magnetic-field reconstruction an inverse problem rather than a direct measurement task. A compact computational model that combines the governing Ramsey measurement probability with data-driven residual correction can therefore provide useful insight into the recoverability of hidden magnetic-field dynamics under noisy quantum measurement conditions.

This microarticle reports a computational physics result for hidden-field recovery from decohered Ramsey-type magnetometer measurements. The contribution is intentionally narrow: a hybrid physics-calibrated Bayesian recovery method is compared against direct probability inversion, moving average filtering, Kalman filtering, a standard neural network, and an ablation PINN without physics calibration.

## 2. Physical model

The normalized magnetic field is modeled as a slowly varying signal disturbed by stochastic Ornstein–Uhlenbeck perturbations. The quantum phase accumulated by the Ramsey sensor is

\[
\phi(t) = \int_0^t \gamma B(\tau) d\tau,
\]

where \(B(t)\) is the hidden magnetic field and \(\gamma\) is the normalized gyromagnetic coefficient. The decohered Ramsey measurement probability is

\[
P(t) = \frac{1}{2}\left[1 - C e^{-t/T_2^*}\cos(\phi(t))\right],
\]

where \(C\) is the measurement contrast and \(T_2^*\) is the dephasing time. Noisy measurements are generated using finite-shot binomial sampling and additive readout noise.

## 3. Hybrid physics-calibrated recovery method

The proposed method uses a two-stage recovery process. First, a supervised baseline PINN estimates the magnetic field from normalized time and noisy Ramsey probability. Second, a residual corrector receives the baseline magnetic-field estimate and the Ramsey probability residual. This corrector learns a small physics-calibrated adjustment to the baseline field estimate. An ensemble of residual correctors provides the posterior-like mean estimate and uncertainty band.

The method is physics-calibrated rather than purely physics-constrained. This distinction is important because directly forcing a physics loss during training degraded point-recovery accuracy in preliminary experiments. The final method instead uses the Ramsey measurement residual as a correction signal, allowing the learned estimator to retain low error while improving physical consistency and uncertainty calibration.

## 4. Results and discussion

The final experiment used 4096 simulated quantum magnetometer samples. The proposed hybrid physics-calibrated Bayesian PINN v3 achieved the best point-recovery performance among all tested methods.

| Method | RMSE | MAE | Recovered SNR (dB) | 95% Coverage |
|---|---:|---:|---:|---:|
| Hybrid Physics-Calibrated Bayesian PINN v3 | 0.2297 | 0.1795 | 5.5795 | 0.9534 |
| Ablation PINN Without Physics | 0.2860 | 0.2242 | 3.6776 | — |
| Standard Neural Network | 0.2882 | 0.2246 | 3.6104 | — |
| Moving Average | 0.4945 | 0.3635 | -1.0793 | — |
| Kalman Filter | 0.5394 | 0.4137 | -1.8333 | — |
| Probability Inversion | 2.4405 | 1.9410 | -14.9454 | — |

The hybrid method reduced RMSE relative to the ablation PINN and standard neural network while maintaining approximately calibrated 95% uncertainty coverage. Direct probability inversion performed poorly because the inverse cosine operation amplifies noise and phase ambiguity under decoherence. Classical smoothing and Kalman filtering improved stability but could not match the learned recovery methods.

## 5. Conclusion

This microarticle presented a compact computational physics result for recovering a hidden magnetic-field trajectory from decohered Ramsey-type quantum magnetometer measurements. A hybrid physics-calibrated Bayesian residual correction method achieved lower RMSE than neural and classical baselines while preserving approximately calibrated uncertainty. The result suggests that physics-calibrated residual learning is a useful strategy for inverse recovery in noisy quantum sensing simulations.

## Data and code availability

The simulation code, trained-recovery outputs, result tables, and generated figures are available in the public GitHub repository associated with this work.\n\n## References

[1] W. Ding, X. Zhang, J. Liu, and X. Wang, “Quantum dynamic response-based NV-diamond magnetometry: Robustness to decoherence and applications in motion detection of magnetic nanoparticles,” arXiv:2307.05255, 2023.

[2] J. Amoros-Binefa and J. Kolodynski, “Noisy atomic magnetometry with Kalman filtering and measurement-based feedback,” arXiv:2403.14764, 2024.

[3] J. Tian, R. S. Said, F. Jelezko, J. Cai, and L. Xiao, “Bayesian-based hybrid method for rapid optimization of NV center sensors,” arXiv:2302.08410, 2023.

[4] F. Belliardo, F. Zoratti, and V. Giovannetti, “Applications of model-aware reinforcement learning in Bayesian quantum metrology,” arXiv:2403.05706, 2024.

[5] K.-H. Cheng, Z. Kazi, J. Rovny, B. Zhang, L. Nassar, J. D. Thompson, and N. P. de Leon, “Massively multiplexed nanoscale magnetometry with diamond quantum sensors,” arXiv:2408.11666, 2024.

[6] A. Youssry, S. Todd, P. Murton, M. J. Arshad, A. Peruzzo, and C. Bonato, “Bayesian quantum sensing using graybox machine learning,” arXiv:2601.17465, 2026.

[7] W. Zhou and Y. F. Xu, “Data-Guided Physics-Informed Neural Networks for Solving Inverse Problems in Partial Differential Equations,” arXiv:2407.10836, 2024.

[8] E. W. Steele, D. R. Reising, and T. Li, “Recovery of Quantum Correlations using Machine Learning,” arXiv:2410.02818, 2024.
\n