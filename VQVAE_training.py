# -*- coding: utf-8 -*-
"""
Script 3/4 — Entraînement du VQ-VAE et du prior conditionnel

Les numéros d'équations et de sections renvoient au rapport de synthèse théorique du groupe
(Rapport_VQ-VAE_synthese_theorique.pdf).

Étage A — VQ-VAE non supervisé sur les architectures (rapport §2, §3, §4.5 point 1)
  n (B,5) entiers dans {0..10} -> one-hot (B,5,11) aplati (B,55)
  -> encodeur MLP 55 -> 128 -> 128 -> 5×8 = Z_e (un vecteur de dimension D=8 par anneau)
  -> quantification sur K=16 codes (plus proche voisin, éq. (6) et (9), straight-through éq. (19))
  -> décodeur MLP 40 -> 128 -> 128 -> 5×11 (softmax par anneau)
  Perte éq. (15)-(16) : entropie croisée par anneau + perte de codebook + β·engagement (β = 0,25)
  Convention : chaque terme est sommé sur les 5 anneaux (positions latentes) puis moyenné sur le lot,
  comme dans l'éq. (16). (Le code de référence §4.3 moyenne tout avec F.mse_loss : essayé, il laisse
  z_e s'éloigner des codes ; reconstruction moins bonne.)

  Remarque : chez l'enseignant, nr ∈ {0,...,10} (0 = anneau vide), soit C = 11 classes par anneau
  (et non 10 comme dans le §4.5 du rapport). Capacité : 5·log2(11) ≈ 17,3 bits <= 5·log2(16) = 20 bits.

Étage B — Prior conditionnel supervisé (rapport §4.5 point 2, éq. (24))
  Autoencodeur figé, X_train encodé en indices k (B,5).
  MLP 4 -> 128 -> 128 -> 128 -> 5×16 : spécifications normalisées -> logits des codes de chaque anneau.
  Perte : somme sur les anneaux des entropies croisées.

Entrées : Minput_training.npy (4 × N), Moutput_training.npy (5 × N)
Fichiers produits (dans le dossier du script) :
  vqvae_model.pt, prior_model.pt                       poids (state_dict)
  normalization_params_VQVAE.npz / .mat                min-max des spécifications (train) et σ (éq. (25))
  historique_VQVAE.npy / .mat, historique_prior.npy / .mat
  courbes_VQVAE.png, courbe_prior.png

Exécution : python VQVAE_training.py            (configuration complète, 100 + 100 époques)
            python VQVAE_training.py --rapide   (configuration réduite, 2 + 2 époques)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io
import os
import sys
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
CONFIG_REDUITE = "--rapide" in sys.argv[1:]
AFFICHER_FIGURES = False   # True : ouvre les fenêtres matplotlib (bloquant), False : enregistre seulement les .png

SEED = 46
R = 5             # nombre d'anneaux (positions latentes N = R, un code par anneau)
C = 11            # classes par anneau : nr ∈ {0,...,10}
K = 16            # taille du codebook
D = 8             # dimension des codes
HIDDEN = 128      # neurones des couches cachées (encodeur, décodeur, prior)
BETA = 0.25       # coefficient d'engagement, éq. (15)

# Étage A (VQ-VAE)
LR_VQVAE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS_VQVAE = 2 if CONFIG_REDUITE else 100

# Étage B (prior conditionnel) : mêmes réglages que l'étage A
LR_PRIOR = 1e-3
NUM_EPOCHS_PRIOR = 2 if CONFIG_REDUITE else 100

if not AFFICHER_FIGURES:
    matplotlib.use("Agg")   # pas de fenêtre : les figures sont seulement enregistrées
import matplotlib.pyplot as plt

# Graines fixées pour des résultats reproductibles
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Configuration : {'RÉDUITE (--rapide)' if CONFIG_REDUITE else 'COMPLÈTE'} | device : {device}")


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
# === Chargement des données (dossier du script) ===
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
# .T : une ligne = un échantillon (convention de l'enseignant pour l'apprentissage)
Minput_training = np.load(os.path.join(script_dir, 'Minput_training.npy')).T    # (N, 4) spécifications
Moutput_training = np.load(os.path.join(script_dir, 'Moutput_training.npy')).T  # (N, 5) architectures

n_train = torch.tensor(Moutput_training.astype(np.int64), dtype=torch.long, device=device)
N_TRAIN = n_train.shape[0]
assert n_train.min() >= 0 and n_train.max() <= C - 1, "les architectures doivent être dans {0..10}"
print(f"Training : {N_TRAIN} architectures | spécifications : {Minput_training.shape[1]} colonnes")


# =============================================================================
# === ÉTAGE A — Entraînement non supervisé du VQ-VAE ===
# =============================================================================
print("\n" + "-"*80 + "\nÉTAGE A - VQ-VAE non supervisé sur les architectures\n" + "-"*80)

vqvae = VQVAE(R, C, K, D, HIDDEN, BETA).to(device)
print(f"Nombre de paramètres du VQ-VAE : {sum(p.numel() for p in vqvae.parameters())}")
optimizer = torch.optim.Adam(vqvae.parameters(), lr=LR_VQVAE)
dataloader = DataLoader(TensorDataset(n_train), batch_size=BATCH_SIZE, shuffle=True)

colonnes_vqvae = ['loss_totale', 'loss_reconstruction', 'loss_codebook', 'loss_engagement',
                  'perplexite', 'codes_utilises', 'codes_reinitialises',
                  'precision_par_anneau', 'precision_par_architecture']
historique_vqvae = []

t_debut = time.time()
for epoch in range(NUM_EPOCHS_VQVAE):
    vqvae.train()
    sommes = np.zeros(4)   # [totale, reconstruction, codebook, engagement]
    nb_batches = 0
    for (batch_n,) in dataloader:
        optimizer.zero_grad()
        logits, loss_codebook, loss_commit, _, _ = vqvae(batch_n)
        # Reconstruction : -sum_r log p(n_r | Z_q), moyennée sur le lot (rapport §4.5, point 1)
        loss_rec = F.cross_entropy(logits.reshape(-1, C), batch_n.reshape(-1),
                                   reduction='sum') / batch_n.shape[0]
        # Perte complète, éq. (15)
        loss = loss_rec + loss_codebook + vqvae.vq.beta * loss_commit
        loss.backward()
        optimizer.step()
        sommes += [loss.item(), loss_rec.item(), loss_codebook.item(), loss_commit.item()]
        nb_batches += 1
    pertes = sommes / nb_batches

    # --- Fin d'époque : diagnostics sur tout le training (rapport §4.4) ---
    vqvae.eval()
    with torch.no_grad():
        logits, _, _, idx, z_e = vqvae(n_train)
        n_hat = logits.argmax(dim=-1)                                   # argmax par anneau
        precision_anneau = (n_hat == n_train).float().mean().item()
        precision_archi = (n_hat == n_train).all(dim=1).float().mean().item()
        compte = torch.bincount(idx.flatten(), minlength=K)
        ppl = perplexite(compte)
        codes_utilises = int((compte > 0).sum().item())

        # --- Codes morts : réinitialisés sur des z_e tirés au hasard (rapport §4.4) ---
        codes_morts = torch.where(compte == 0)[0]
        if len(codes_morts) > 0:
            z_flat = z_e.reshape(-1, D)
            tirage = torch.randint(0, z_flat.shape[0], (len(codes_morts),), device=device)
            vqvae.vq.codebook.weight.data[codes_morts] = z_flat[tirage]

    historique_vqvae.append([*pertes, ppl, codes_utilises, len(codes_morts), precision_anneau, precision_archi])
    print(f"Epoch {epoch+1}/{NUM_EPOCHS_VQVAE} | Loss {pertes[0]:.4f} (rec {pertes[1]:.4f}, "
          f"codebook {pertes[2]:.4f}, engag. {pertes[3]:.4f}) | perplexité {ppl:5.2f} | "
          f"codes utilisés {codes_utilises:2d}/{K} (réinit. {len(codes_morts)}) | "
          f"précision anneau {100*precision_anneau:6.2f} % | archi {100*precision_archi:6.2f} %")

duree_A = time.time() - t_debut
print(f"\n=== Étage A terminé en {duree_A:.2f} secondes ===")
historique_vqvae = np.array(historique_vqvae, dtype=float)


# =============================================================================
# === ÉTAGE B — Prior conditionnel supervisé ===
# =============================================================================
print("\n" + "-"*80 + "\nÉTAGE B - Prior conditionnel p(k | s)\n" + "-"*80)

# --- Autoencodeur figé, X_train encodé en indices de codes k (N, 5) ---
vqvae.eval()
for p in vqvae.parameters():
    p.requires_grad_(False)
with torch.no_grad():
    _, _, _, k_train, _ = vqvae(n_train)

# --- Normalisation Min-Max des spécifications, calculée sur le training uniquement (convention FFNN) ---
Minput_min = Minput_training.min(axis=0)
Minput_max = Minput_training.max(axis=0)
Minput_norm = (Minput_training - Minput_min) / (Minput_max - Minput_min + 1e-8)
# Écarts-types σ_i (G, SSL, HPBW) sur le training, utilisés par le critère de sélection éq. (25)
sigma_specs = Minput_training[:, :3].std(axis=0)

s_train = torch.tensor(Minput_norm, dtype=torch.float32, device=device)
prior = PriorConditionnel(Minput_training.shape[1], HIDDEN, R, K).to(device)
print(f"Nombre de paramètres du prior : {sum(p.numel() for p in prior.parameters())}")
optimizer_prior = torch.optim.Adam(prior.parameters(), lr=LR_PRIOR)
dataloader_prior = DataLoader(TensorDataset(s_train, k_train), batch_size=BATCH_SIZE, shuffle=True)

colonnes_prior = ['loss_prior', 'precision_code_par_anneau', 'precision_combinaison_complete']
historique_prior = []

t_debut = time.time()
for epoch in range(NUM_EPOCHS_PRIOR):
    prior.train()
    somme, nb_batches = 0.0, 0
    for batch_s, batch_k in dataloader_prior:
        optimizer_prior.zero_grad()
        logits = prior(batch_s)                                          # (B, R, K)
        # Éq. (24) : -sum_r log p(k_r | s), moyennée sur le lot
        loss = F.cross_entropy(logits.reshape(-1, K), batch_k.reshape(-1),
                               reduction='sum') / batch_s.shape[0]
        loss.backward()
        optimizer_prior.step()
        somme += loss.item()
        nb_batches += 1

    prior.eval()
    with torch.no_grad():
        k_hat = prior(s_train).argmax(dim=-1)
        prec_code = (k_hat == k_train).float().mean().item()
        prec_comb = (k_hat == k_train).all(dim=1).float().mean().item()
    historique_prior.append([somme / nb_batches, prec_code, prec_comb])
    print(f"Epoch {epoch+1}/{NUM_EPOCHS_PRIOR} | Loss {somme / nb_batches:.4f} | "
          f"précision code/anneau {100*prec_code:6.2f} % | combinaison complète {100*prec_comb:6.2f} %")

duree_B = time.time() - t_debut
print(f"\n=== Étage B terminé en {duree_B:.2f} secondes ===")
historique_prior = np.array(historique_prior, dtype=float)


# =============================================================================
# === Sauvegardes (dossier du script) ===
# =============================================================================
torch.save(vqvae.state_dict(), os.path.join(script_dir, 'vqvae_model.pt'))
torch.save(prior.state_dict(), os.path.join(script_dir, 'prior_model.pt'))

params_norm = {'Minput_min': Minput_min, 'Minput_max': Minput_max, 'sigma_specs': sigma_specs}
np.savez(os.path.join(script_dir, 'normalization_params_VQVAE.npz'), **params_norm)
scipy.io.savemat(os.path.join(script_dir, 'normalization_params_VQVAE.mat'), params_norm)

# Historiques : une ligne par époque, une colonne par grandeur suivie (noms dans le .mat)
np.save(os.path.join(script_dir, 'historique_VQVAE.npy'), historique_vqvae)
scipy.io.savemat(os.path.join(script_dir, 'historique_VQVAE.mat'),
                 {'historique_VQVAE': historique_vqvae,
                  'colonnes': np.array(colonnes_vqvae, dtype=object),
                  'duree_s': duree_A})
np.save(os.path.join(script_dir, 'historique_prior.npy'), historique_prior)
scipy.io.savemat(os.path.join(script_dir, 'historique_prior.mat'),
                 {'historique_prior': historique_prior,
                  'colonnes': np.array(colonnes_prior, dtype=object),
                  'duree_s': duree_B})


# =============================================================================
# === Courbes d'entraînement ===
# =============================================================================
BLEU, ORANGE, VERT = '#2a78d6', '#eb6834', '#1baf7a'   # couleurs fixes par série
epochs_A = np.arange(1, NUM_EPOCHS_VQVAE + 1)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax = axes[0, 0]
ax.plot(epochs_A, historique_vqvae[:, 1], color=BLEU, linewidth=2, label='Reconstruction (entropie croisée)')
ax.plot(epochs_A, historique_vqvae[:, 0], color=ORANGE, linewidth=2, label='Totale, éq. (15)')
ax.set_title("Pertes de l'autoencodeur")
ax.set_ylabel("Perte")

ax = axes[0, 1]
ax.plot(epochs_A, historique_vqvae[:, 2], color=BLEU, linewidth=2, label='Codebook')
ax.plot(epochs_A, historique_vqvae[:, 3], color=ORANGE, linewidth=2, label='Engagement (avant ×β)')
ax.set_title("Pertes de quantification")
ax.set_ylabel("Perte")

ax = axes[1, 0]
ax.plot(epochs_A, historique_vqvae[:, 4], color=BLEU, linewidth=2, label='Perplexité, éq. (23)')
ax.plot(epochs_A, historique_vqvae[:, 5], color=ORANGE, linewidth=2, label='Codes utilisés')
ax.axhline(K, color='0.5', linewidth=1, linestyle=':', label=f'K = {K}')
ax.set_title("Utilisation du codebook")
ax.set_ylabel("Nombre de codes")

ax = axes[1, 1]
ax.plot(epochs_A, 100 * historique_vqvae[:, 7], color=BLEU, linewidth=2, label='Par anneau')
ax.plot(epochs_A, 100 * historique_vqvae[:, 8], color=ORANGE, linewidth=2, label='Architecture complète')
ax.set_title("Précision de reconstruction (training)")
ax.set_ylabel("Précision (%)")
ax.set_ylim(0, 101)

for ax in axes.flat:
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
fig.suptitle("Étage A - entraînement du VQ-VAE")
fig.tight_layout()
fig.savefig(os.path.join(script_dir, 'courbes_VQVAE.png'), dpi=150)

epochs_B = np.arange(1, NUM_EPOCHS_PRIOR + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.plot(epochs_B, historique_prior[:, 0], color=BLEU, linewidth=2)
ax.set_title("Perte du prior, éq. (24)")
ax.set_ylabel("Perte")
ax = axes[1]
ax.plot(epochs_B, 100 * historique_prior[:, 1], color=BLEU, linewidth=2, label='Code par anneau')
ax.plot(epochs_B, 100 * historique_prior[:, 2], color=ORANGE, linewidth=2, label='Combinaison complète')
ax.set_title("Précision du code le plus probable (training)")
ax.set_ylabel("Précision (%)")
ax.set_ylim(0, 101)
ax.legend(frameon=False)
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
fig.suptitle("Étage B - entraînement du prior conditionnel")
fig.tight_layout()
fig.savefig(os.path.join(script_dir, 'courbe_prior.png'), dpi=150)

if AFFICHER_FIGURES:
    plt.show()
plt.close('all')


# =============================================================================
# === Résumé ===
# =============================================================================
print("\n" + "="*80 + "\nRÉSUMÉ DE L'ENTRAÎNEMENT\n" + "="*80)
print(f"Étage A ({NUM_EPOCHS_VQVAE} époques, {duree_A:.1f} s) : "
      f"précision anneau {100*historique_vqvae[-1, 7]:.2f} %, architecture {100*historique_vqvae[-1, 8]:.2f} %, "
      f"perplexité {historique_vqvae[-1, 4]:.2f}, codes utilisés {int(historique_vqvae[-1, 5])}/{K}")
print(f"Étage B ({NUM_EPOCHS_PRIOR} époques, {duree_B:.1f} s) : "
      f"loss {historique_prior[-1, 0]:.4f}, précision code/anneau {100*historique_prior[-1, 1]:.2f} %")
print(f"Ecarts-types sigma (training, éq. (25)) : G = {sigma_specs[0]:.3f} dB, SSL = {sigma_specs[1]:.3f} dB, "
      f"HPBW = {sigma_specs[2]:.3f}°")
print(f"\nFichiers enregistrés dans {script_dir} :")
print("  vqvae_model.pt, prior_model.pt")
print("  normalization_params_VQVAE.npz, normalization_params_VQVAE.mat")
print("  historique_VQVAE.npy, historique_VQVAE.mat, historique_prior.npy, historique_prior.mat")
print("  courbes_VQVAE.png, courbe_prior.png")
