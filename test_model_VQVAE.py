# -*- coding: utf-8 -*-
"""
Script 4/4 — Test du VQ-VAE (conception inverse) et comparaison avec le FFNN de l'enseignant

Pour chaque spécification s* = (G*, SSL*, HPBW*, θ0*) du jeu de test (rapport §4.5, point 3) :
  1. tirer N_CAND = 300 combinaisons de codes selon le prior p(k | s*), plus la combinaison la plus probable ;
  2. décoder chaque combinaison en architecture (argmax par anneau) et supprimer les doublons ;
  3. calculer G, SSL et HPBW de chaque candidat avec le code de l'enseignant (à l'angle θ0*) ;
  4. retenir n̂ = argmin sum_i w_i |spec_i - spec_i*| / σ_i, éq. (25).

Évaluation :
  - erreur de prédiction calculée avec le code de l'enseignant (test_model_FFNN.py) + MAE par spécification ;
  - précision de reconstruction et perplexité du VQ-VAE sur le test ;
  - diagrammes demandé / obtenu pour quelques exemples ;
  - si ffnn_model.pth et normalization_params.npz (fichiers de l'enseignant) sont présents :
    même évaluation pour le FFNN et tableau comparatif VQ-VAE vs FFNN.

Fichiers produits (dans le dossier du script) :
  Moutput_pred_VQVAE.npy, Minput_obtenu_VQVAE.npy, erreurs_VQVAE.npy, predictions_VQVAE.mat
  metriques_VQVAE.npy, metriques_VQVAE.mat
  Moutput_pred_FFNN.npy, Minput_obtenu_FFNN.npy, erreurs_FFNN.npy,
  comparaison_VQVAE_FFNN.npy, comparaison_VQVAE_FFNN.mat            (si le FFNN est disponible)
  diagramme_exemple_XX.png, distribution_erreurs.png

Exécution : python test_model_VQVAE.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
from scipy.signal import find_peaks
import os
import time
import random
import matplotlib


### Affichage visuel de démarcation (convention de l'enseignant)
def clear_console():
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")

clear_console()


# =============================================================================
# === Hyperparamètres ===
# =============================================================================
AFFICHER_FIGURES = False                 # True : ouvre les fenêtres matplotlib (bloquant)
SEED = 46
N_CAND = 300                             # combinaisons tirées selon le prior (+ la plus probable)
POIDS_CRITERE = np.array([1.0, 1.0, 1.0])  # w_G, w_SSL, w_HPBW de l'éq. (25)
INDICES_FIGURES = [0, 10, 42, 77]        # exemples de test tracés (10 = exemple utilisé par l'enseignant)

# Architecture des modèles : IDENTIQUE à VQVAE_training.py
R, C, K, D, HIDDEN, BETA = 5, 11, 16, 8, 128, 0.25

if not AFFICHER_FIGURES:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device : {device} | N_CAND = {N_CAND} | poids du critère (25) = {POIDS_CRITERE}")


# === Paramètres physiques : copiés de test_model_FFNN.py (l. 15-20 et 123-129) ===
carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
r0 = 0.2 * lambda_
delta_r = 0.5 * lambda_
max_rings = 5
k = 2 * np.pi / lambda_
phi0 = 0
phi = 0
theta = np.linspace(0, 2 * np.pi, 1000)
theta_deg = np.rad2deg(theta)
theta0 = 0.0   # variable globale lue par calcul_AF_performance_metrics, fixée par specs_architecture()


# =============================================================================
# === Définition des modèles (IDENTIQUE dans VQVAE_training.py et test_model_VQVAE.py) ===
# =============================================================================
class VectorQuantizer(nn.Module):
    """Quantificateur vectoriel (rapport §2.3, §3.1 et code de référence §4.3).
    Remplace chaque vecteur z_e (un par anneau) par le code le plus proche du codebook."""
    def __init__(self, K, D, beta):
        super(VectorQuantizer, self).__init__()
        self.K, self.D, self.beta = K, D, beta
        self.codebook = nn.Embedding(K, D)                   # K codes de dimension D
        self.codebook.weight.data.uniform_(-1 / K, 1 / K)

    def forward(self, z_e):                                  # z_e : (B, R, D)
        flat = z_e.reshape(-1, self.D)                       # (B*R, D)
        # Distances au carré ||z||² - 2 z^T e + ||e||², éq. (6) : (B*R, K)
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.codebook.weight.t()
             + self.codebook.weight.pow(2).sum(1))
        idx = d.argmin(dim=1)                                # plus proche voisin, éq. (9)
        z_q = self.codebook(idx).view_as(z_e)
        # Pertes de codebook et d'engagement, éq. (16) : normes au carré sommées sur les R positions,
        # puis moyennées sur le lot (même convention que la reconstruction, sommée sur les anneaux).
        # .detach() joue le rôle de sg[.]
        loss_codebook = (z_q - z_e.detach()).pow(2).flatten(1).sum(1).mean()   # ne met à jour que le codebook
        loss_commit = (z_e - z_q.detach()).pow(2).flatten(1).sum(1).mean()     # ne met à jour que l'encodeur
        # Estimateur straight-through, éq. (19)
        z_q = z_e + (z_q - z_e).detach()
        return z_q, loss_codebook, loss_commit, idx.view(z_e.shape[:-1])   # idx : (B, R)


class VQVAE(nn.Module):
    """Autoencodeur à latent discret sur les architectures (rapport §4.5, point 1)."""
    def __init__(self, R, C, K, D, hidden, beta):
        super(VQVAE, self).__init__()
        self.R, self.C, self.D = R, C, D
        # Encodeur MLP : R*C -> hidden -> hidden -> R*D (rapport §2.2)
        self.encoder = nn.Sequential(nn.Linear(R * C, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, R * D))
        self.vq = VectorQuantizer(K, D, beta)
        # Décodeur MLP symétrique : R*D -> hidden -> hidden -> R*C (rapport §2.4)
        self.decoder = nn.Sequential(nn.Linear(R * D, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, R * C))

    def encode(self, n):                                     # n : (B, R) entiers dans {0..C-1}
        x = F.one_hot(n, self.C).float().flatten(1)          # (B, R*C) = (B, 55)
        return self.encoder(x).view(-1, self.R, self.D)      # z_e : (B, R, D)

    def decode(self, z_q):                                   # z_q : (B, R, D)
        # logits (B, R, C) ; le softmax par anneau est appliqué dans F.cross_entropy / argmax
        return self.decoder(z_q.flatten(1)).view(-1, self.R, self.C)

    def decode_indices(self, idx):                           # idx : (B, R) indices de codes
        return self.decode(self.vq.codebook(idx))

    def forward(self, n):
        z_e = self.encode(n)
        z_q, loss_codebook, loss_commit, idx = self.vq(z_e)
        logits = self.decode(z_q)
        return logits, loss_codebook, loss_commit, idx, z_e


class PriorConditionnel(nn.Module):
    """Prior conditionnel p(k | s) = prod_r p(k_r | s), éq. (24) (rapport §4.5, point 2).
    Entrée : spécifications normalisées (G, SSL, HPBW, θ0) ; sortie : logits (B, R, K)."""
    def __init__(self, n_specs, hidden, R, K):
        super(PriorConditionnel, self).__init__()
        self.R, self.K = R, K
        self.net = nn.Sequential(nn.Linear(n_specs, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, R * K))

    def forward(self, s_norm):                               # s_norm : (B, 4)
        return self.net(s_norm).view(-1, self.R, self.K)


def perplexite(compte_codes):
    """Perplexité du codebook, éq. (23) : exp(-sum p_k log p_k), entre 1 et K."""
    p = compte_codes.float() / compte_codes.sum()
    return torch.exp(-(p * torch.log(p + 1e-10)).sum()).item()
# ======================== fin des définitions communes ========================


# =============================================================================
# Fonction de l'enseignant — COPIÉE TELLE QUELLE depuis test_model_FFNN.py (l. 133-207), non modifiée.
# (Le fichier de l'enseignant exécute du code à l'import : il ne peut pas être importé.)
# ATTENTION : elle lit les variables GLOBALES theta0 (en rad), theta, k, r0, delta_r, max_rings, phi, phi0.
# Sorties : AF_dB_az (diagramme normalisé en dB, 1000 points), HPBW (°), Gain non normalisé (dB), SSL (dB)
# =============================================================================
def calcul_AF_performance_metrics(elements_per_ring):
    rings = len(elements_per_ring)
    radii = r0 + delta_r * np.arange(max_rings)

    AF_az = np.zeros_like(theta, dtype=complex)

    for ring in range(rings):
        a = radii[ring]
        N = elements_per_ring[ring]
        if N == 0:
            continue
        phi_n = 2 * np.pi * np.arange(N) / N
        for n in range(N):
            phase = k * a * (np.sin(theta) * np.cos(phi - phi_n[n]) -
                             np.sin(theta0) * np.cos(phi0 - phi_n[n]))
            AF_az += np.exp(1j * phase)


    AF_norm_az = np.abs(AF_az) / (np.max(np.abs(AF_az)) + np.finfo(float).eps)
    AF_dB_az = 20 * np.log10(AF_norm_az + np.finfo(float).eps)
    AF_dB_az[AF_dB_az < -40] = -40
    theta_deg = np.rad2deg(theta)

    #calcul du gain max non normalisé du lobe principale
    AF_abs_az = np.abs(AF_az)
    maxVal = np.max(AF_abs_az)
    maxVal_non_norm = 20 * np.log10(maxVal + np.finfo(float).eps)

    # --- HPBW ---
    maxVal_dB = np.max(AF_dB_az)
    maxIdx = np.argmax(AF_dB_az)
    halfPower = maxVal_dB - 3#
    AF_dB_ext = np.concatenate((AF_dB_az, AF_dB_az, AF_dB_az))#
    theta_deg_ext = np.concatenate((theta_deg - 360, theta_deg, theta_deg + 360))#
    maxIdx_ext = maxIdx + len(theta_deg)#


    leftIdx_ext = np.where(AF_dB_ext[:maxIdx_ext] <= halfPower)[0][-1] if np.any(
        AF_dB_ext[:maxIdx_ext] <= halfPower) else None #

    rightIdx_ext = np.where(AF_dB_ext[maxIdx_ext:] <= halfPower)[0]


    rightIdx_ext = rightIdx_ext[0] + maxIdx_ext if len(rightIdx_ext) > 0 else None

    if leftIdx_ext is None or rightIdx_ext is None:
        HPBW = 180
    else:
        HPBW = theta_deg_ext[rightIdx_ext] - theta_deg_ext[leftIdx_ext]

    # --- Gain lobe principal & SSL ---
    responseLin = AF_norm_az#
    peaks, _ = find_peaks(responseLin, distance=5)

    pk = responseLin[peaks]#
    if len(pk) == 0:#
        main_lobe_gain = maxVal_non_norm
        true_SSL_gain = 0#
    else:
        sorted_idx = np.argsort(pk)[::-1]#
        sorted_pk = pk[sorted_idx]#
        sorted_angles = theta_deg[peaks][sorted_idx]#

        threshold_dB = 1#
        main_lobes_idx = np.where(
            20 * np.log10(sorted_pk + np.finfo(float).eps) >=
            20 * np.log10(sorted_pk[0] + np.finfo(float).eps) - threshold_dB)[0]#
        side_lobe_idx = np.setdiff1d(np.arange(len(sorted_pk)), main_lobes_idx)#

        main_lobe_gain = maxVal_non_norm

        true_SSL_gain = 20 * np.log10(sorted_pk[side_lobe_idx[0]] + np.finfo(float).eps) if len(
            side_lobe_idx) > 0 else 0

    return AF_dB_az, HPBW, maxVal_non_norm, true_SSL_gain
# ===================== fin de la copie du code de l'enseignant =====================


def erreur_prediction_enseignant(gain_main_pred, SSL_pred, HPBW_pred,
                                 gain_main_ref, SSL_ref, HPBW_ref, Minput_complete):
    """Erreur de prédiction de l'enseignant. Les lignes ci-dessous sont COPIÉES TELLES QUELLES depuis
    test_model_FFNN.py (l. 263-265 et 274-298) ; elles sont seulement placées dans une fonction pour
    être appliquées à chaque exemple de test. Minput_complete : matrice Minput complète (4 × M)."""
    # --- copié de test_model_FFNN.py, l. 263-265 ---
    mae_HPBW = abs(HPBW_pred - HPBW_ref)
    mae_main_lobe_gain = abs(gain_main_pred - gain_main_ref)
    mae_true_SSL_gain = abs(SSL_pred - SSL_ref)

    # --- copié de test_model_FFNN.py, l. 274-298 ---
    # --- Extraction des valeurs maximales ---
    max_main_lobe_gain = np.max(Minput_complete[0, :])
    max_true_SSL_gain  = np.max(np.abs(Minput_complete[1, :]))
    max_HPBW          = np.max(Minput_complete[2, :])

    # === Coefficients de pondération ===
    w1 = 0.33
    w2 = 0.33
    w3 = 0.34

    # === Calcul des erreurs relatives (en %) ===
    err_rel_main_lobe_gain = 100*(abs(gain_main_pred - gain_main_ref) / (abs(max_main_lobe_gain) + 1e-8))
    err_rel_true_SSL_gain  = 100*(abs(SSL_pred - SSL_ref) / (abs(max_true_SSL_gain) + 1e-8))
    err_rel_HPBW           = 100*(abs(HPBW_pred - HPBW_ref) / (180 + 1e-8))

    # === Calcul de l'erreur de prediction globale pondérée ===
    error_pred_global = (    w1 * err_rel_main_lobe_gain +    w2 * err_rel_true_SSL_gain  +    w3 * err_rel_HPBW  )
    # --- fin de la copie ---

    return [mae_main_lobe_gain, mae_true_SSL_gain, mae_HPBW,
            err_rel_main_lobe_gain, err_rel_true_SSL_gain, err_rel_HPBW, error_pred_global]

COLONNES_ERREURS = ['MAE_G_dB', 'MAE_SSL_dB', 'MAE_HPBW_deg',
                    'err_rel_G_pct', 'err_rel_SSL_pct', 'err_rel_HPBW_pct', 'erreur_globale_pct']


def specs_architecture(architecture, theta0deg_cible):
    """Fixe θ0 (variable globale lue par la fonction de l'enseignant), puis l'appelle.
    Retourne (AF_dB, G, SSL, HPBW)."""
    global theta0
    theta0 = np.deg2rad(theta0deg_cible)
    AF_dB, HPBW, gain, SSL = calcul_AF_performance_metrics(np.asarray(architecture, dtype=int))
    return AF_dB, gain, SSL, HPBW


# =============================================================================
# === Chargement des données et des modèles (dossier du script) ===
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
Minput_test = np.load(os.path.join(script_dir, 'Minput_test.npy')).T      # (n_test, 4) : G, SSL, HPBW, θ0
Moutput_test = np.load(os.path.join(script_dir, 'Moutput_test.npy')).T    # (n_test, 5)
Minput_complete = np.load(os.path.join(script_dir, 'Minput.npy'))         # (4, M), pour l'erreur de l'enseignant
n_test = Minput_test.shape[0]
print(f"Jeu de test : {n_test} exemples (dataset complet : {Minput_complete.shape[1]} échantillons)")

norm = np.load(os.path.join(script_dir, 'normalization_params_VQVAE.npz'))
Minput_min, Minput_max, sigma_specs = norm['Minput_min'], norm['Minput_max'], norm['sigma_specs']

vqvae = VQVAE(R, C, K, D, HIDDEN, BETA).to(device)
vqvae.load_state_dict(torch.load(os.path.join(script_dir, 'vqvae_model.pt'), map_location=device))
vqvae.eval()
prior = PriorConditionnel(Minput_test.shape[1], HIDDEN, R, K).to(device)
prior.load_state_dict(torch.load(os.path.join(script_dir, 'prior_model.pt'), map_location=device))
prior.eval()


# =============================================================================
# === 1) Diagnostics du VQ-VAE et du prior sur le test ===
# =============================================================================
n_test_t = torch.tensor(Moutput_test.astype(np.int64), dtype=torch.long, device=device)
s_test_norm = torch.tensor((Minput_test - Minput_min) / (Minput_max - Minput_min + 1e-8),
                           dtype=torch.float32, device=device)
with torch.no_grad():
    logits, _, _, k_test, _ = vqvae(n_test_t)
    n_hat = logits.argmax(dim=-1)
    precision_recon_anneau = (n_hat == n_test_t).float().mean().item()
    precision_recon_archi = (n_hat == n_test_t).all(dim=1).float().mean().item()
    compte = torch.bincount(k_test.flatten(), minlength=K)
    perplexite_test = perplexite(compte)
    codes_utilises_test = int((compte > 0).sum().item())
    logits_prior = prior(s_test_norm)
    loss_prior_test = (F.cross_entropy(logits_prior.reshape(-1, K), k_test.reshape(-1),
                                       reduction='sum') / n_test).item()
    precision_prior_test = (logits_prior.argmax(dim=-1) == k_test).float().mean().item()

print("\n--- VQ-VAE sur le jeu de test ---")
print(f"Précision de reconstruction par anneau      : {100*precision_recon_anneau:.2f} %")
print(f"Précision de reconstruction par architecture : {100*precision_recon_archi:.2f} %")
print(f"Perplexité du codebook (test)               : {perplexite_test:.2f} / {K} ({codes_utilises_test} codes utilisés)")
print(f"Prior : loss éq. (24) = {loss_prior_test:.4f}, précision code/anneau = {100*precision_prior_test:.2f} %")


# =============================================================================
# === 2) Conception inverse : candidats du prior + vérification physique, éq. (25) ===
# =============================================================================
print(f"\n--- Conception inverse sur {n_test} spécifications (jusqu'à {N_CAND + 1} candidats chacune) ---")
Moutput_pred_VQVAE = np.zeros((n_test, R), dtype=int)
Minput_obtenu_VQVAE = np.zeros((n_test, 4))
nb_candidats_uniques = np.zeros(n_test, dtype=int)
score_critere = np.zeros(n_test)
reference_parmi_candidats = np.zeros(n_test, dtype=bool)

t_debut = time.time()
for i in range(n_test):
    s_cible = Minput_test[i]                     # spécification demandée s* = (G*, SSL*, HPBW*, θ0*)
    with torch.no_grad():
        logits_i = prior(s_test_norm[i:i + 1])[0]                           # (R, K)
        k_tires = torch.distributions.Categorical(logits=logits_i).sample((N_CAND,))   # (N_CAND, R)
        k_mode = logits_i.argmax(dim=-1, keepdim=True).T                    # (1, R) la plus probable
        k_cand = torch.cat([k_mode, k_tires], dim=0)                        # (N_CAND + 1, R)
        archis = vqvae.decode_indices(k_cand).argmax(dim=-1).cpu().numpy()  # argmax par anneau
    archis = np.unique(archis, axis=0)           # suppression des doublons
    archis = archis[archis.sum(axis=1) > 0]      # on écarte [0,0,0,0,0] (aucun élément)
    if len(archis) == 0:                         # cas pathologique : un seul élément sur l'anneau 1
        archis = np.array([[1, 0, 0, 0, 0]])

    # Spécifications de chaque candidat avec le code de l'enseignant, à l'angle θ0* demandé
    specs_cand = np.array([specs_architecture(a, s_cible[3])[1:] for a in archis])   # (Nc, 3) : G, SSL, HPBW
    # Critère de sélection, éq. (25)
    scores = (POIDS_CRITERE * np.abs(specs_cand - s_cible[:3]) / sigma_specs).sum(axis=1)
    meilleur = int(np.argmin(scores))

    Moutput_pred_VQVAE[i] = archis[meilleur]
    Minput_obtenu_VQVAE[i] = [*specs_cand[meilleur], s_cible[3]]
    nb_candidats_uniques[i] = len(archis)
    score_critere[i] = scores[meilleur]
    reference_parmi_candidats[i] = bool((archis == Moutput_test[i].astype(int)).all(axis=1).any())
    if (i + 1) % 10 == 0 or i + 1 == n_test:
        print(f"  {i + 1}/{n_test} | {time.time() - t_debut:6.1f} s écoulées | candidats uniques (dernier) : {len(archis)}")

duree_vqvae = time.time() - t_debut
print(f"Temps total d'inférence VQ-VAE : {duree_vqvae:.1f} s, soit {duree_vqvae / n_test:.2f} s par spécification "
      f"({nb_candidats_uniques.sum()} évaluations du facteur de réseau)")


# =============================================================================
# === 3) Erreur de prédiction (code de l'enseignant) ===
# =============================================================================
# Comme dans test_model_FFNN.py, la référence est recalculée à partir de l'architecture de test.
specs_ref = np.array([specs_architecture(Moutput_test[i].astype(int), Minput_test[i, 3])[1:]
                      for i in range(n_test)])                              # (n_test, 3)
ecart_ref = np.abs(specs_ref - Minput_test[:, :3]).max()
print(f"\nContrôle : écart max entre les specs de référence recalculées et Minput_test = {ecart_ref:.2e}")

erreurs_VQVAE = np.array([erreur_prediction_enseignant(*Minput_obtenu_VQVAE[i, :3], *specs_ref[i], Minput_complete)
                          for i in range(n_test)])                          # (n_test, 7)
archis_exactes_VQVAE = (Moutput_pred_VQVAE == Moutput_test.astype(int)).all(axis=1).mean()


# =============================================================================
# === 4) FFNN de l'enseignant sur le même jeu de test (si ses fichiers sont présents) ===
# =============================================================================
chemin_ffnn = os.path.join(script_dir, 'ffnn_model.pth')
chemin_norm_ffnn = os.path.join(script_dir, 'normalization_params.npz')
ffnn_disponible = os.path.exists(chemin_ffnn) and os.path.exists(chemin_norm_ffnn)

if ffnn_disponible:
    print("\n--- FFNN de l'enseignant (ffnn_model.pth) sur le même jeu de test ---")
    if Minput_complete.shape[1] != 1000:
        print("ATTENTION : dataset réduit -> le split diffère de celui du FFNN, comparaison indicative seulement.")

    # Classe COPIÉE TELLE QUELLE depuis test_model_FFNN.py (l. 53-80), commentaires abrégés
    class FFNN(nn.Module):
        def __init__(self, input_size, hidden1, hidden2, hidden3, output_size, dropout_rate=0.000):
            super(FFNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden1)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(p=dropout_rate)

            self.fc2 = nn.Linear(hidden1, hidden2)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(p=dropout_rate)

            self.fc3 = nn.Linear(hidden2, hidden3)
            self.relu3 = nn.ReLU()
            self.dropout3 = nn.Dropout(p=dropout_rate)

            self.fc4 = nn.Linear(hidden3, output_size)

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.dropout1(x)       # applique dropout

            x = self.relu2(self.fc2(x))
            x = self.dropout2(x)       # applique dropout

            x = self.relu3(self.fc3(x))
            x = self.dropout3(x)       # applique dropout

            return self.fc4(x)

    norm_ffnn = np.load(chemin_norm_ffnn)
    # Lignes de test_model_FFNN.py (l. 45, 86-106, 112), appliquées à tout le jeu de test au lieu d'un exemple
    Minput_norm_test = (Minput_test - norm_ffnn['Minput_min']) / (norm_ffnn['Minput_max'] - norm_ffnn['Minput_min'] + 1e-8)
    model = FFNN(input_size=4, hidden1=14, hidden2=20, hidden3=8, output_size=5, dropout_rate=0.000)
    model.load_state_dict(torch.load(chemin_ffnn, map_location='cpu'))
    model.eval()
    t_debut = time.time()
    with torch.no_grad():
        y_pred_norm = model(torch.tensor(Minput_norm_test, dtype=torch.float32))
    y_pred = y_pred_norm.numpy() * (norm_ffnn['Moutput_max'] - norm_ffnn['Moutput_min'] + 1e-8) + norm_ffnn['Moutput_min']
    Moutput_pred_FFNN = np.round(y_pred).astype(int)

    Minput_obtenu_FFNN = np.zeros((n_test, 4))
    for i in range(n_test):
        _, g, s, h = specs_architecture(Moutput_pred_FFNN[i], Minput_test[i, 3])
        Minput_obtenu_FFNN[i] = [g, s, h, Minput_test[i, 3]]
    duree_ffnn = time.time() - t_debut
    erreurs_FFNN = np.array([erreur_prediction_enseignant(*Minput_obtenu_FFNN[i, :3], *specs_ref[i], Minput_complete)
                             for i in range(n_test)])
    archis_exactes_FFNN = (Moutput_pred_FFNN == Moutput_test.astype(int)).all(axis=1).mean()
    print(f"Temps : {duree_ffnn:.2f} s (dont calcul des spécifications des {n_test} architectures prédites)")
else:
    print("\nFFNN non disponible (ffnn_model.pth / normalization_params.npz absents) : pas de comparaison.")


# =============================================================================
# === 5) Résumé console et tableau comparatif ===
# =============================================================================
def resume(erreurs, archis_exactes, duree):
    """Métriques agrégées sur le jeu de test (ordre = LIGNES_TABLEAU)."""
    return np.array([erreurs[:, 6].mean(), np.median(erreurs[:, 6]),
                     erreurs[:, 3].mean(), erreurs[:, 4].mean(), erreurs[:, 5].mean(),
                     erreurs[:, 0].mean(), erreurs[:, 1].mean(), erreurs[:, 2].mean(),
                     100 * archis_exactes, duree / n_test])

LIGNES_TABLEAU = ['Erreur globale moyenne (%)', 'Erreur globale médiane (%)',
                  'Erreur relative G (%)', 'Erreur relative SSL (%)', 'Erreur relative HPBW (%)',
                  'MAE G (dB)', 'MAE SSL (dB)', 'MAE HPBW (°)',
                  'Architectures identiques à la référence (%)', "Temps d'inférence par exemple (s)"]
NOMS_MAT = ['erreur_globale_moyenne_pct', 'erreur_globale_mediane_pct',
            'err_rel_G_pct', 'err_rel_SSL_pct', 'err_rel_HPBW_pct',
            'MAE_G_dB', 'MAE_SSL_dB', 'MAE_HPBW_deg',
            'archis_identiques_pct', 'temps_inference_par_exemple_s']

resume_VQVAE = resume(erreurs_VQVAE, archis_exactes_VQVAE, duree_vqvae)

print("\n" + "="*80 + "\nRÉSUMÉ DU TEST\n" + "="*80)
print(f"Candidats uniques par spécification : moyenne {nb_candidats_uniques.mean():.1f}, "
      f"min {nb_candidats_uniques.min()}, max {nb_candidats_uniques.max()}")
print(f"L'architecture de référence figure parmi les candidats dans {100*reference_parmi_candidats.mean():.1f} % des cas")
print("(rappel : le problème inverse n'a pas de solution unique, une autre architecture peut convenir aussi bien)\n")

if ffnn_disponible:
    resume_FFNN = resume(erreurs_FFNN, archis_exactes_FFNN, duree_ffnn)
    print(f"{'Métrique':<46}{'VQ-VAE':>12}{'FFNN':>12}")
    print("-" * 70)
    for nom, v, f in zip(LIGNES_TABLEAU, resume_VQVAE, resume_FFNN):
        print(f"{nom:<46}{v:>12.3f}{f:>12.3f}")
else:
    print(f"{'Métrique':<46}{'VQ-VAE':>12}")
    print("-" * 58)
    for nom, v in zip(LIGNES_TABLEAU, resume_VQVAE):
        print(f"{nom:<46}{v:>12.3f}")


# =============================================================================
# === 6) Sauvegardes (.npy et .mat, orientation de l'enseignant : une colonne = un exemple) ===
# =============================================================================
np.save(os.path.join(script_dir, 'Moutput_pred_VQVAE.npy'), Moutput_pred_VQVAE.T)     # (5, n_test)
np.save(os.path.join(script_dir, 'Minput_obtenu_VQVAE.npy'), Minput_obtenu_VQVAE.T)   # (4, n_test)
np.save(os.path.join(script_dir, 'erreurs_VQVAE.npy'), erreurs_VQVAE)                 # (n_test, 7)
scipy.io.savemat(os.path.join(script_dir, 'predictions_VQVAE.mat'), {
    'Moutput_pred_VQVAE': Moutput_pred_VQVAE.T,
    'Minput_obtenu_VQVAE': Minput_obtenu_VQVAE.T,
    'nb_candidats_uniques': nb_candidats_uniques,
    'score_critere': score_critere,
    'reference_parmi_candidats': reference_parmi_candidats.astype(int),
    'erreurs_VQVAE': erreurs_VQVAE,
    'colonnes_erreurs': np.array(COLONNES_ERREURS, dtype=object)})

metriques_VQVAE = np.concatenate([resume_VQVAE,
                                  [100 * precision_recon_anneau, 100 * precision_recon_archi,
                                   perplexite_test, codes_utilises_test, loss_prior_test,
                                   100 * precision_prior_test, nb_candidats_uniques.mean(),
                                   100 * reference_parmi_candidats.mean()]])
noms_metriques = NOMS_MAT + ['precision_recon_anneau_test_pct', 'precision_recon_archi_test_pct',
                             'perplexite_test', 'codes_utilises_test', 'loss_prior_test',
                             'precision_prior_anneau_test_pct', 'nb_candidats_uniques_moyen',
                             'reference_parmi_candidats_pct']
np.save(os.path.join(script_dir, 'metriques_VQVAE.npy'), metriques_VQVAE)
scipy.io.savemat(os.path.join(script_dir, 'metriques_VQVAE.mat'),
                 {**dict(zip(noms_metriques, metriques_VQVAE)),
                  'noms_metriques': np.array(noms_metriques, dtype=object), 'N_CAND': N_CAND,
                  'POIDS_CRITERE': POIDS_CRITERE})

fichiers = ['Moutput_pred_VQVAE.npy', 'Minput_obtenu_VQVAE.npy', 'erreurs_VQVAE.npy', 'predictions_VQVAE.mat',
            'metriques_VQVAE.npy', 'metriques_VQVAE.mat']
if ffnn_disponible:
    tableau = np.column_stack([resume_VQVAE, resume_FFNN])                           # (10, 2)
    np.save(os.path.join(script_dir, 'Moutput_pred_FFNN.npy'), Moutput_pred_FFNN.T)
    np.save(os.path.join(script_dir, 'Minput_obtenu_FFNN.npy'), Minput_obtenu_FFNN.T)
    np.save(os.path.join(script_dir, 'erreurs_FFNN.npy'), erreurs_FFNN)
    np.save(os.path.join(script_dir, 'comparaison_VQVAE_FFNN.npy'), tableau)
    scipy.io.savemat(os.path.join(script_dir, 'comparaison_VQVAE_FFNN.mat'), {
        'tableau': tableau,
        'lignes': np.array(NOMS_MAT, dtype=object),
        'colonnes': np.array(['VQVAE', 'FFNN'], dtype=object),
        'Moutput_pred_FFNN': Moutput_pred_FFNN.T,
        'Minput_obtenu_FFNN': Minput_obtenu_FFNN.T,
        'erreurs_FFNN': erreurs_FFNN,
        'colonnes_erreurs': np.array(COLONNES_ERREURS, dtype=object)})
    fichiers += ['Moutput_pred_FFNN.npy', 'Minput_obtenu_FFNN.npy', 'erreurs_FFNN.npy',
                 'comparaison_VQVAE_FFNN.npy', 'comparaison_VQVAE_FFNN.mat']


# =============================================================================
# === 7) Figures : diagrammes demandé / obtenu et distribution des erreurs ===
# =============================================================================
BLEU, ORANGE, VERT = '#2a78d6', '#eb6834', '#1baf7a'   # Demandé, VQ-VAE, FFNN (couleurs fixes)
indices_figures = [i for i in INDICES_FIGURES if i < n_test] or [0]
for i in indices_figures:
    theta0deg_i = Minput_test[i, 3]
    courbes = [('Demandé ' + str(Moutput_test[i].astype(int)), BLEU, '-',
                specs_architecture(Moutput_test[i].astype(int), theta0deg_i)[0]),
               ('VQ-VAE ' + str(Moutput_pred_VQVAE[i]), ORANGE, '--',
                specs_architecture(Moutput_pred_VQVAE[i], theta0deg_i)[0])]
    if ffnn_disponible:
        courbes.append(('FFNN ' + str(Moutput_pred_FFNN[i]), VERT, ':',
                        specs_architecture(Moutput_pred_FFNN[i], theta0deg_i)[0]))

    fig = plt.figure(figsize=(14, 5.5))
    ax_pol = fig.add_subplot(1, 2, 1, polar=True)
    ax_cart = fig.add_subplot(1, 2, 2)
    for label, couleur, style, AF_dB in courbes:
        ax_pol.plot(theta, AF_dB, color=couleur, linestyle=style, linewidth=2, label=label)
        ax_cart.plot(theta_deg, AF_dB, color=couleur, linestyle=style, linewidth=2, label=label)
    ax_pol.set_rlim([-40, 0])
    ax_pol.set_title("Diagramme de rayonnement (polaire)")
    ax_cart.set_xlabel("Azimut (°)")
    ax_cart.set_ylabel("Gain (dB)")
    ax_cart.set_xlim([0, 360])
    ax_cart.set_ylim([-40, 0])
    ax_cart.grid(True, alpha=0.3)
    ax_cart.legend(frameon=False, loc='lower right')
    ax_cart.set_title(f"G / SSL / HPBW demandés : {Minput_test[i, 0]:.2f} dB / {Minput_test[i, 1]:.2f} dB / {Minput_test[i, 2]:.1f}°\n"
                      f"VQ-VAE : {Minput_obtenu_VQVAE[i, 0]:.2f} dB / {Minput_obtenu_VQVAE[i, 1]:.2f} dB / "
                      f"{Minput_obtenu_VQVAE[i, 2]:.1f}° (erreur {erreurs_VQVAE[i, 6]:.2f} %)", fontsize=10)
    fig.suptitle(f"Exemple de test {i} - θ0 = {theta0deg_i:.1f}°")
    fig.tight_layout()
    nom = f'diagramme_exemple_{i:02d}.png'
    fig.savefig(os.path.join(script_dir, nom), dpi=150)
    fichiers.append(nom)

# Distribution des erreurs (même unité, %, sur chaque axe)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
modeles = [('VQ-VAE', erreurs_VQVAE, ORANGE)] + ([('FFNN', erreurs_FFNN, VERT)] if ffnn_disponible else [])
bp = axes[0].boxplot([e[:, 6] for _, e, _ in modeles], patch_artist=True, widths=0.5)
axes[0].set_xticks(range(1, len(modeles) + 1), [n for n, _, _ in modeles])
for patch, (_, _, couleur) in zip(bp['boxes'], modeles):
    patch.set_facecolor(couleur)
    patch.set_alpha(0.6)
axes[0].set_ylabel("Erreur globale de prédiction (%)")
axes[0].set_title("Erreur globale (formule de l'enseignant)")
largeur = 0.8 / len(modeles)
positions = np.arange(3)
for j, (nom_modele, e, couleur) in enumerate(modeles):
    axes[1].bar(positions + (j - (len(modeles) - 1) / 2) * largeur, e[:, 3:6].mean(axis=0),
                width=largeur * 0.95, color=couleur, label=nom_modele)
axes[1].set_xticks(positions, ['G', 'SSL', 'HPBW'])
axes[1].set_ylabel("Erreur relative moyenne (%)")
axes[1].set_title("Erreur relative moyenne par spécification")
axes[1].legend(frameon=False)
for ax in axes:
    ax.grid(True, axis='y', alpha=0.3)
fig.suptitle(f"Jeu de test ({n_test} exemples)")
fig.tight_layout()
fig.savefig(os.path.join(script_dir, 'distribution_erreurs.png'), dpi=150)
fichiers.append('distribution_erreurs.png')

if AFFICHER_FIGURES:
    plt.show()
plt.close('all')

print(f"\nFichiers enregistrés dans {script_dir} :")
for nom in fichiers:
    print(f"  {nom}")
