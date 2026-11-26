# README facts: every number and claim, with its source

The root `README.md` uses only facts from this table. Each row gives the value, where it comes from (a file on `main`, the code, or the exact command that produced it) and its status:

- **file**: read from a result file committed on `main` (`.npy`, `.mat`, `.npz`, `.pt`).
- **code**: read from the code on `main`.
- **run 23 Nov**: produced by the command shown on 23 Nov 2026 (Windows 11, PowerShell, Python 3.12.15, fresh venv with numpy 2.5.5, scipy 1.18.1, torch 2.15.0+cpu, matplotlib 3.11.2; a second env with torch 2.11.0+cu128 for the GPU rows). Listed in [pull request #1](https://github.com/ymhamdi/VQVAE-Antenna-Inverse-Design/pull/1).
- **run 26 Nov**: produced on 26 Nov 2026 on the same laptop (Intel Core i7-13650HX, NVIDIA GeForce RTX 5050 Laptop GPU), venv `C:\venvs\vqvae-rl` with numpy 2.5.5, scipy 1.18.1, matplotlib 3.11.2, torch 2.14.1+cu130. Listed in [pull request #2](https://github.com/ymhamdi/VQVAE-Antenna-Inverse-Design/pull/2).
- **outside scripts**: measured on 26 Nov with throwaway code that is not in the repo. Stated as such wherever it appears.
- **docs only**: stated in `docs/`, not checkable from the repo.

To read a `.mat` key: `python -c "import scipy.io as s; print(s.loadmat('rl_dqn/comparaison_methodes.mat')['tableau'])"`.

## The array and its physics

| # | Fact | Value | Source | Status |
|---|---|---|---|---|
| P1 | Context | AI lab, INSAT, IIA3 track, teacher N. ISMAIL; two labs (VQ-VAE, then DQN) | `README.md` header (PR #1); `rl_dqn/README.md` « La consigne » | file |
| P2 | Array | 5 concentric rings, f = 2.45 GHz, λ = c / f ≈ 12.24 cm | `dataset.py`: `max_rings`, `carrierFreq`, `lambda_` | code |
| P3 | Radii | 0.2, 0.7, 1.2, 1.7, 2.2 λ (r0 = 0.2 λ, Δr = 0.5 λ) | `dataset.py`: `r0`, `delta_r` | code |
| P4 | Elements | n_r ∈ {0…10}, at least one ring non-empty; φ_n = 2πn / n_r | `dataset.py` lines 159-162 | code |
| P5 | Possible arrays | 11⁵ − 1 = 161,050 | from P4; `rl_dqn/physique_antenne.py` `NB_ARCHITECTURES` | code |
| P6 | Specs | G, SSL, HPBW, θ0; θ0 uniform in [0°, 180°] | `dataset.py` lines 166-176 | code |
| P7 | Cut | φ = 0, 1,000 angles θ ∈ [0, 2π] | `dataset.py`: `theta`, `phi` | code |
| P8 | Ranges in the dataset | G 12.04 → 33.44 dB; SSL −38.69 → −1.35 dB; HPBW 11.17 → 113.51°; θ0 0.06 → 179.68° (README: 12.0 to 33.4 dB, −38.7 to −1.3 dB, 11.2 to 113.5°, 0.1 to 179.7°) | `Minput.npy`, min / max per row | file |
| P9 | G depends only on the total | \|G − 20·log10(Σ n_r)\| ≤ 0.0007 dB on the 1,000 samples | `Minput.npy` row 0 vs `Moutput.npy` | file |
| P10 | θ0 acts only through sin θ0 | phase = k·a_r·cos φ_n·(sin θ − sin θ0) since φ = φ0 = 0; θ0 and 180° − θ0 give the same G, SSL, HPBW (max gap 1.8e-14 dB on 500 arrays) | `calcul_AF_performance_metrics`; `rl_dqn/verification_physique.mat` `ecart_max_symetrie_*` | code, file |
| P11 | Time per array, teacher's function | 0.73 to 0.82 ms (`python dataset.py`, 23 Nov); 0.86 ms (`python rl_dqn/physique_antenne.py`, 26 Nov) | console; `verification_physique.mat` `temps_enseignant_ms` | run 23 Nov, file |
| P12 | Hook spec (test spec 0) | θ0 = 109.9°; requested 29.83 dB / −9.28 dB / 73.2°; reference [8 4 10 0 9] (31 elements); VQ-VAE [9 1 9 5 7] error 0.27%; FFNN [6 7 7 6 5] error 1.76% | `Minput_test.npy`, `Moutput_test.npy`, `Moutput_pred_*.npy`, `erreurs_*.npy`, column / row 0 | file |

## First lab: the VQ-VAE

| # | Fact | Value | Source | Status |
|---|---|---|---|---|
| D1 | Data and split | 1,000 samples, seed 46; 900 training / 100 test | `dataset.py` `SEED`; `train_indices.npy`, `test_indices.npy` | file |
| D2 | Unseen test | 0 of the 100 test arrays is in the training set | `Moutput_test.npy` vs `Moutput_training.npy` | file |
| D3 | Same split as the FFNN | teacher's `normalization_params.npz` min / max = our `Minput_training` min / max (gap 0.0) | `normalization_params.npz` | file |
| D4 | "Same dataset, checked file by file"; "function copied unchanged" | the teacher's files are not in the repo; D3 is partial proof | `docs/01_dataset.md`, `docs/02_repartition.md` | docs only |
| D5 | Recomputed test specs = `Minput_test` | max gap 0.00e+00 | console of `test_model_VQVAE.py` and of `rl_dqn/test_model_DQN.py` | run 23 Nov, run 26 Nov |
| D6 | Error formula | 0.33·e_G + 0.33·e_SSL + 0.34·e_HPBW, relative to max G, max \|SSL\|, 180° | `test_model_VQVAE.py` `erreur_prediction_enseignant` | code |
| M1 | Codebook | K = 16 codes of D = 8, one per ring; 11 classes per ring | `VQVAE_training.py`: `K`, `D`, `R`, `C` | code |
| M2 | Parameters | VQ-VAE 57,823; prior 43,984 | sum of `vqvae_model.pt`, `prior_model.pt` tensors | file |
| M3 | σ of criterion (25) | 2.8861 dB, 3.0458 dB, 24.9647° (README: 2.89, 3.05, 24.96) | `normalization_params_VQVAE.npz` `sigma_specs` | file |
| M4 | Candidates | 300 samples + the most likely; 294.24 unique on average (269 to 301); 29,424 evaluations | `test_model_VQVAE.py` `N_CAND`; `predictions_VQVAE.mat` `nb_candidats_uniques` | code, file |
| E1 | Reconstruction (training, epoch 100) | 76.4% per ring, 28.3% per array | `historique_VQVAE.npy` last row, columns 7-8 | file |
| E2 | Codebook use | 16 codes from epoch 2 (15 at epoch 1); perplexity 15.76 | `historique_VQVAE.npy` columns 4-5 | file |
| E3 | Prior (training) | 17.7% of rings get the right code; chance 6.25% | `historique_prior.npy` last row, column 1 | file |
| E4 | Curves | reconstruction loss 12.00 → 3.37 (README alt: about 3.4); codebook = engagement every epoch, 6.49 at epoch 100 (README alt: about 6.5) | `historique_VQVAE.npy` columns 1-3 | file |
| E5 | Committed durations | stage A 13.2 s, stage B 12.0 s | `historique_*.mat` `duree_s` | file |
| T1 | Mean global error | VQ-VAE 0.6136%, FFNN 3.3715% (README: 0.61%, 3.37%; console: 0.614, 3.372) | `comparaison_VQVAE_FFNN.npy` row 0 | file |
| T2 | Median · 90th percentile · worst | VQ-VAE 0.544 · 1.100 · 2.126%; FFNN 2.990 · 6.671 · 8.232% (README: 0.54 · 1.10 · 2.13 and 2.99 · 6.67 · 8.23) | `erreurs_*.npy` column 6 | file |
| T3 | Mean absolute errors | G 0.133 / 0.326 dB; SSL 0.2349 / 2.978 dB; HPBW 1.492 / 2.699° (README: 0.13 / 0.33, 0.23 / 2.98, 1.49 / 2.70) | `comparaison_VQVAE_FFNN.npy` rows 5-7 | file |
| T4 | Mean relative SSL error | 0.607% / 7.696% (README: 0.61% / 7.70%) | row 3 | file |
| T5 | Wins | VQ-VAE lower on 96 / 100 specs | `erreurs_VQVAE.npy[:, 6] < erreurs_FFNN.npy[:, 6]` | file |
| T6 | Exact reference found | 0 / 100 for both | row 8 | file |
| T7 | Time per spec | VQ-VAE 1.0449 s, FFNN 0.0035 s (machine not recorded) | row 9 | file |
| T8 | FFNN "middle" arrays | 74.4% of values in [4, 7]; per-ring std 1.58 (FFNN), 3.13 (reference), 3.19 (VQ-VAE) | `Moutput_pred_FFNN.npy`, `Moutput_test.npy`, `Moutput_pred_VQVAE.npy` | file |
| T9 | Test reconstruction | 49.2% per ring, 3.0% per array | `metriques_VQVAE.mat` | file |
| T10 | Spec 77 (plots) | θ0 = 2.7°; reference [1 6 6 7 8]; VQ-VAE [4 3 5 9 8] 0.34% (SSL −10.42 dB for −10.37 requested); FFNN [5 6 6 6 6] 5.13% (SSL −15.71 dB) | column 77 of the P12 files | file |
| T11 | Specs 10 and 42 | 10: θ0 30.4°, VQ-VAE [9 1 2 1 9] 0.43%, FFNN [4 4 4 5 5] 0.57%; 42: θ0 42.9°, VQ-VAE [3 3 10 4 9] 0.32%, FFNN [5 5 6 6 7] 0.39% | columns 10, 42 | file |
| S1 | Script 4 on CPU, committed models | identical to the committed run (0.614%) | `$env:CUDA_VISIBLE_DEVICES = "-1"; python test_model_VQVAE.py` | run 23 Nov |
| S2 | Script 4 on GPU, committed models | 0.713% | `python test_model_VQVAE.py` with CUDA | run 23 Nov |
| S3 | Seed spread, committed models, CPU | seeds 1-5: 0.665 / 0.720 / 0.678 / 0.658 / 0.713% | `SEED` at the top of `test_model_VQVAE.py` | run 23 Nov |
| S4 | Full retrain | CPU 0.654%, GPU 0.799% (README: 0.65, 0.80) | the four scripts | run 23 Nov |
| S5 | Durations (CPU, fresh venv) | `pip install` 41 s; scripts 1.8 s, 0.5 s, 11.1 s, 32.9 s; script 4 alone 34.6 s; time per spec 0.21 s (GPU) to 0.30 s (CPU) | PowerShell timing | run 23 Nov |
| S7 | Long paths | `pip install torch` fails with `OSError` when the venv sits in a deep folder (Windows 260-character path limit); works in a short folder | first fresh-clone attempt, PR #1 | run 23 Nov |
| S6 | Quick chain | `--rapide`: 50 samples, 2 + 2 epochs, ≈ 8 s | the four scripts with `--rapide` | run 23 Nov |
| X1 | Random search, same check | 0.72% (protocol not in the repo; our own re-run gave 0.76 to 0.86%) | `docs/05_resultats.md` Limits §1; PR #1 | docs only |
| X2 | Durations of the committed run | 2-5 s, < 1 s, about 25 s, about 2 min | `docs/01`-`04` | docs only |
| X3 | Equation numbers (15), (24), (25) | refer to the group's theoretical report, not in the repo | `docs/03_entrainement.md` | docs only |

## Second lab: the DQN

| # | Fact | Value | Source | Status |
|---|---|---|---|---|
| Q1 | Problem | min SSL subject to G ≥ G_min − 0.05 dB and HPBW ≤ HPBW_max + 0.5° | `rl_dqn/physique_antenne.py` `TOL_G_DB`, `TOL_HPBW_DEG`, `est_valide` | code |
| Q2 | G tolerance is exact | consecutive totals differ by ≥ 20·log10(50/49) = 0.175 dB | from P9 | code |
| Q3 | Unconstrained optimum is small and wide | θ0 = 30°: [7 6 0 0 0], 13 elements, SSL −58.89 dB, HPBW 54.05° (README: −58.9 dB, 54°); θ0 = 90°: [8 0 0 0 0], 8 elements, −25.24 dB, 168.29° (also 5°, 60°, 85°) | `rl_dqn/verification_physique.mat` `optimum_sans_contrainte` | file |
| Q4 | Fast physics = teacher's function | 2,000 random (array, θ0) pairs: max gap 1.07e-14 dB (G), 4.97e-14 dB (SSL), 0 (HPBW); 0 pairs above 1e-9; batch version 5.68e-14 dB, 8 exact fallbacks; copy identical to `dataset.py` | `rl_dqn/verification_physique.npy` / `.mat` | file |
| Q5 | Speeds | fast version 0.19 ms per array (+ 5.3 ms per θ0 for the ring patterns); exhaustive search 0.40 s per θ0 on GPU (≈ 138 s estimated with the teacher's function) | `verification_physique.mat` `temps_*` | file |
| Q6 | Environment | 11 actions (+1 / −1 per ring, stop); 30 steps; 14-value state with sin θ0 only; requests drawn on the fly (θ0 uniform, G_min and HPBW_max uniform in the ranges of 64 random arrays at that θ0); no data file read | `rl_dqn/env_antenne.py` constants, `tirer_requete` | code |
| Q7 | Score and reward | J = −SSL − 10·(G shortfall, dB) − 1·(HPBW excess, °) − 5 dB if invalid; r = ΔJ/10 − 0.005; illegal move −0.1. One missing antenna costs ≥ 10·20·log10(50/49) = 1.75 dB of score (N ≤ 50) | `env_antenne.py` `ALPHA`, `BETA`, `PENALITE_INVALIDE`, `ECHELLE_J`, `COUT_PAS`, `PENALITE_ILLEGAL` | code |
| Q8 | Network | MLP 14 → 256 → 256 → 11, 72,459 parameters; Double DQN, Huber, replay 100,000, batch 512 every 4 steps, target copy every 500 updates, ε 1 → 0.05 | `rl_dqn/DQN_training.py`; `dqn_model.pt` | code, file |
| Q9 | Training run | 3 seeds (46, 47, 48) × 300,000 steps on GPU; 51,323 episodes; 388 / 410 / 436 s (1,238 s in all); last 500 episodes valid in 84.4 / 83.6 / 84.6% | `rl_dqn/historique_DQN.mat` `duree_s`; console of `DQN_training.py` | file, run 26 Nov |
| Q10 | Seed 47 degrades | return 1.88 → 1.21 and loss 0.0036 → 0.0114 (200-250k vs > 275k steps; README: degrades after 275,000 steps) | `rl_dqn/historique_DQN.npy` columns 3, 11 | file |
| Q11 | Test results (mean ± std, 3 seeds) | DQN: valid 86.0 ± 4.3%, mean gap to optimum 6.90 ± 1.28 dB (valid answers), mean SSL −15.22 ± 1.42 dB, 20.5 evaluations, 0.012 s per spec. Random search, same budget: 83.0 ± 2.9%, 10.24 ± 0.20 dB, −12.15 dB. Local search, same budget: 54.7 ± 2.5%, 10.99 ± 0.48 dB, −12.40 dB. Optimum: −21.66 dB mean (−18.76 median), 161,050 evaluations, 0.376 s per spec (README: 0.38 s; heading: 0.4 s) | `rl_dqn/comparaison_methodes.mat` `tableau`, `ecart_type_graines` | file |
| Q12 | Same table, first-lab rows | reference −8.30 dB (100% valid, gap 13.37 dB); VQ-VAE −8.32 dB (65% valid); FFNN −8.71 dB (31% valid) | same | file |
| Q13 | Per seed | DQN valid 82 / 84 / 92%, gap 5.97 / 8.72 / 6.02 dB; random search valid 84 / 86 / 79%, gap 10.27 / 9.98 / 10.46 dB; DQN lower SSL than random on 64 / 73, 46 / 73, 70 / 77 specs where both are valid | `rl_dqn/resultats_DQN.mat` `resultats` | file |
| Q14 | "Number" of antennas | the optimum uses more elements than G_min requires on 96% of specs (+10.43 on average); 13.37 dB lower than the reference on average (README: 13.4 dB); median 32,590 valid arrays per spec | `resultats_DQN.mat` `resultats`, `N_ref`, `nb_valides_exhaustive` | file |
| Q15 | Spec 0 with the DQN | DQN seed 46 [6 2 10 10 3]: SSL −16.11 dB, G 29.83 dB, HPBW 72.43°, valid; VQ-VAE [9 1 9 5 7]: −9.36 dB, HPBW 72.07°; optimum [5 6 8 10 3] −18.54 dB. Re-drawn by `assets/readme/make_readme_images.py` with the teacher's function (same values) | `resultats_DQN.mat`, row 0; console of the generator | file, run 26 Nov |
| Q16 | Spec 10 with the DQN | seed 46 fails: no valid array in 25 evaluations, ends at HPBW 18.02° (limit 17.8°), 6 repeated illegal moves | `resultats_DQN.mat` `trajectoire_DQN` | file |
| Q17 | The 5 dB penalty saw the test set | pilots on the 100 test specs (1 seed, 100k steps): P = 0 → 59% valid, 7.2 dB; P = 20 dB → 94%, 10.3 dB; P = 5 dB → 85%, 8.1 dB | throwaway runs, logs not in the repo | outside scripts |
| Q18 | Update cost | ≈ 1.4 ms per update, same for batches of 128 and 512 on the GPU | throwaway benchmark | outside scripts |
| Q19 | Durations | `physique_antenne.py` 30 s; R2 50 s; `--rapide` chain (3 scripts) 22.3 s | timed in Git Bash | run 26 Nov |
| F1 | Fresh clone, 26 Nov (PowerShell, `C:\venvs\vqvae-rl` first on PATH) | `git clone` 5.2 s; `test_model_VQVAE.py` CPU (`$env:CUDA_VISIBLE_DEVICES = "-1"`) 0.614%, 32.9 s; GPU 0.713%, 26.3 s; `rl_dqn/test_model_DQN.py` 51.9 s, DQN / random / local / optimum rows identical to the committed table; `--rapide` chains 10.9 s (scripts 1-4) and 23.1 s (R0-R2); `git restore .` leaves 0 changed files. R2 run after script 4 shows a different "VQ-VAE" row (48% valid): it reads the arrays script 4 wrote last | console and `Measure-Command` in a fresh clone of `main` | run 26 Nov |
| Q20 | Training curves (mean of 3 seeds, moving average over 500 episodes, first vs last window) | return −0.14 → 1.56 (about 2 near 200,000 steps); best valid SSL −10.54 → −15.85 dB (README alt: −10.5 → −15.9); valid episodes 50.0 → 84.2%; evaluations per episode 8.25 → 19.92 | `rl_dqn/historique_DQN.npy` columns 3, 5, 6, 7; `courbes_DQN.png` | file |

## History and credits

| # | Fact | Value | Source | Status |
|---|---|---|---|---|
| H1 | Commits | 20 on `main` (18 non-merge, 2 merges of PR #1 and PR #2); one author, as "Youssef El Mhamdi" (12) and "Youssef Mhamdi" (8) | `git rev-list --count origin/main`; `git log origin/main --format=%an` | run 26 Nov |
| H2 | AI use | on 26 Nov, 10 of the 20 commits on `main` carry a `Co-authored-by: Claude` trailer; the 6 DQN commits of 26 Nov (`038114d` … `f823a30`) do not, though they were written with Claude Code | `git log origin/main -i --grep="Co-authored-by: Claude" --oneline` | run 26 Nov |
| H8 | Team | Youssef El Mhamdi, INSAT, IIA3 | given by the author (Checkpoint 1, 26 Nov) | author |
| H3 | Libraries | numpy, scipy (`scipy.io`, `scipy.signal.find_peaks`), torch, matplotlib; no requirements file | `import` lines of the 7 scripts | code |
| H4 | VQ-VAE paper | A. van den Oord, O. Vinyals, K. Kavukcuoglu, *Neural Discrete Representation Learning*, NeurIPS 2017, arXiv:1711.00937 | the paper | external |
| H5 | DQN papers | V. Mnih et al., *Human-level control through deep reinforcement learning*, Nature 518, 2015; H. van Hasselt, A. Guez, D. Silver, *Deep Reinforcement Learning with Double Q-learning*, AAAI 2016, arXiv:1509.06461 | the papers | external |
| H6 | Tests | no automated test suite (`pytest` / `unittest`); two built-in checks: recomputed test specs = `Minput_test` (scripts 4 and R2), fast physics vs teacher's function (`physique_antenne.py`) | search of the repo | code |
| H7 | Visibility | public repository | `gh repo view --json visibility` | run 26 Nov |
