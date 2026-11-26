# -*- coding: utf-8 -*-
"""
Script R1/R2 — Entraînement d'un DQN (Deep Q-Network) pour minimiser le SSL (second TP, apprentissage par renforcement)

Consigne : « optimiser le nombre et l'emplacement des antennes pour minimiser le SSL, avec un modèle d'apprentissage
par renforcement (score → récompense / pénalité), sans dataset et sans répartition training / test ».

  - Aucun dataset : l'agent ne lit aucun fichier (ni Minput, ni Moutput). Les requêtes (θ0, G_min, HPBW_max) et les
    architectures de départ sont tirées à la volée par l'environnement (env_antenne.py), et chaque architecture
    visitée est évaluée par le facteur de réseau (physique_antenne.py, version rapide vérifiée contre la fonction
    de l'enseignant).
  - Un seul agent pour toutes les requêtes. Il est entraîné avec plusieurs graines pour mesurer la dispersion.

DQN (écrit ici avec PyTorch, sans bibliothèque de RL) :
  - réseau Q : MLP 14 -> 256 -> 256 -> 11 (ReLU), une valeur Q(s, a) par action ;
  - mémoire de rejeu (replay buffer) : transitions (s, a, r, s', terminé), lots tirés au hasard ;
  - réseau cible : copie du réseau Q, mise à jour toutes les MAJ_CIBLE mises à jour ;
  - cible Double DQN : y = r + γ·(1 − terminé)·Q_cible(s', argmax_a Q(s', a)) ;
  - perte de Huber entre Q(s, a) et y ; Adam ; écrêtage de la norme du gradient ;
  - exploration ε-greedy, ε décroît linéairement de 1 à 0,05.

Fichiers produits (dans le dossier du script) :
  dqn_model.pt                   poids des réseaux Q (un state_dict par graine) et hyperparamètres
  historique_DQN.npy / .mat      une ligne par épisode (colonnes dans le .mat)
  courbes_DQN.png                courbes d'entraînement (moyenne des graines et enveloppe min-max)

Exécution : python rl_dqn/DQN_training.py            (configuration complète : 3 graines × 300 000 pas)
            python rl_dqn/DQN_training.py --rapide   (configuration réduite : 1 graine × 3 000 pas)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
import os
import sys
import time
import random
import warnings
import matplotlib

import physique_antenne as pa
import env_antenne as ea

sys.stdout.reconfigure(encoding="utf-8")   # console en UTF-8 même si la sortie est redirigée


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
GRAINES = [SEED] if CONFIG_REDUITE else [SEED, SEED + 1, SEED + 2]
NB_PAS = 3_000 if CONFIG_REDUITE else 300_000      # pas d'environnement par graine
HIDDEN = 256
GAMMA = 0.99
LR = 5e-4
TAILLE_BUFFER = 100_000
TAILLE_LOT = 512
PAS_PAR_MAJ = 4            # une mise à jour toutes les 4 transitions (même rejeu qu'un lot de 128 à chaque pas)
DEBUT_APPRENTISSAGE = 500 if CONFIG_REDUITE else 5_000
MAJ_CIBLE = 500            # copie du réseau cible toutes les 500 mises à jour
EPS_DEBUT, EPS_FIN = 1.0, 0.05
FRACTION_EPS = 0.6         # ε décroît sur les 60 premiers % des pas
CLIP_GRAD = 10.0
FENETRE_COURBES = 50 if CONFIG_REDUITE else 500   # moyenne glissante des courbes (épisodes)

if not AFFICHER_FIGURES:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Configuration : {'RÉDUITE (--rapide)' if CONFIG_REDUITE else 'COMPLÈTE'} | device : {device} | "
      f"graines {GRAINES} | {NB_PAS} pas par graine | T = {ea.T_MAX} pas par épisode")


# =============================================================================
# === Réseau Q (IDENTIQUE dans DQN_training.py et test_model_DQN.py) ===
# =============================================================================
class ReseauQ(nn.Module):
    """Q(s, ·) : état (14 valeurs) -> une valeur par action (11). MLP 14 -> 256 -> 256 -> 11, ReLU."""
    def __init__(self, dim_etat, nb_actions, hidden):
        super(ReseauQ, self).__init__()
        self.net = nn.Sequential(nn.Linear(dim_etat, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, nb_actions))

    def forward(self, etat):
        return self.net(etat)
# ======================== fin des définitions communes ========================


def epsilon(pas):
    """ε-greedy : décroissance linéaire de EPS_DEBUT à EPS_FIN sur FRACTION_EPS · NB_PAS pas."""
    return max(EPS_FIN, EPS_DEBUT - (EPS_DEBUT - EPS_FIN) * pas / (FRACTION_EPS * NB_PAS))


COLONNES_HISTORIQUE = ['graine', 'episode', 'pas_cumules', 'retour', 'nb_pas', 'nb_evaluations',
                       'valide_trouvee', 'meilleur_SSL_valide_dB', 'J_initial', 'J_final', 'epsilon', 'perte_moyenne']


def entrainer(graine):
    """Entraîne un DQN avec une graine. Retourne (state_dict, historique (n_episodes, 12), durée en s)."""
    random.seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    env = ea.EnvAntenne(np.random.RandomState(graine))          # requêtes et départs
    rng_actions = np.random.RandomState(graine + 1000)          # exploration ε-greedy

    q = ReseauQ(ea.DIM_ETAT, ea.NB_ACTIONS, HIDDEN).to(device)
    q_cible = ReseauQ(ea.DIM_ETAT, ea.NB_ACTIONS, HIDDEN).to(device)
    q_cible.load_state_dict(q.state_dict())
    optimizer = torch.optim.Adam(q.parameters(), lr=LR)

    # Mémoire de rejeu (tableaux circulaires)
    B_s = np.zeros((TAILLE_BUFFER, ea.DIM_ETAT), dtype=np.float32)
    B_a = np.zeros(TAILLE_BUFFER, dtype=np.int64)
    B_r = np.zeros(TAILLE_BUFFER, dtype=np.float32)
    B_s2 = np.zeros((TAILLE_BUFFER, ea.DIM_ETAT), dtype=np.float32)
    B_t = np.zeros(TAILLE_BUFFER, dtype=np.float32)
    nb_transitions = 0

    historique = []
    etat = env.reset()
    J_initial, retour, somme_perte, nb_maj_ep, nb_maj = env.J, 0.0, torch.zeros((), device=device), 0, 0
    t_debut = time.time()
    for pas in range(NB_PAS):
        # --- action ε-greedy ---
        eps = epsilon(pas)
        if rng_actions.rand() < eps:
            action = int(rng_actions.randint(ea.NB_ACTIONS))
        else:
            with torch.no_grad():
                action = int(q(torch.as_tensor(etat, device=device)[None]).argmax(dim=1).item())
        etat_suivant, recompense, termine, info = env.step(action)

        # --- stockage de la transition ---
        i = nb_transitions % TAILLE_BUFFER
        B_s[i], B_a[i], B_r[i], B_s2[i], B_t[i] = etat, action, recompense, etat_suivant, float(termine)
        nb_transitions += 1
        etat = etat_suivant
        retour += recompense

        # --- mise à jour du réseau Q (Double DQN, perte de Huber) ---
        if pas >= DEBUT_APPRENTISSAGE and pas % PAS_PAR_MAJ == 0:
            idx = np.random.randint(0, min(nb_transitions, TAILLE_BUFFER), size=TAILLE_LOT)
            s = torch.as_tensor(B_s[idx], device=device)
            a = torch.as_tensor(B_a[idx], device=device)
            r = torch.as_tensor(B_r[idx], device=device)
            s2 = torch.as_tensor(B_s2[idx], device=device)
            t = torch.as_tensor(B_t[idx], device=device)
            with torch.no_grad():
                a_etoile = q(s2).argmax(dim=1, keepdim=True)                       # choix par le réseau Q
                y = r + GAMMA * (1 - t) * q_cible(s2).gather(1, a_etoile).squeeze(1)   # valeur par le réseau cible
            q_sa = q(s).gather(1, a[:, None]).squeeze(1)
            perte = F.smooth_l1_loss(q_sa, y)
            optimizer.zero_grad()
            perte.backward()
            nn.utils.clip_grad_norm_(q.parameters(), CLIP_GRAD)
            optimizer.step()
            somme_perte += perte.detach()
            nb_maj_ep += 1
            nb_maj += 1
            if nb_maj % MAJ_CIBLE == 0:
                q_cible.load_state_dict(q.state_dict())

        # --- fin d'épisode ---
        if termine:
            ssl_valide = env.meilleure_valide[0] if env.meilleure_valide is not None else np.nan
            historique.append([graine, len(historique), pas + 1, retour, env.pas, info['nb_evaluations'],
                               float(env.meilleure_valide is not None), ssl_valide, J_initial, env.J, eps,
                               (somme_perte / nb_maj_ep).item() if nb_maj_ep > 0 else np.nan])
            etat = env.reset()
            J_initial, retour, somme_perte, nb_maj_ep = env.J, 0.0, torch.zeros((), device=device), 0

        if (pas + 1) % max(1, NB_PAS // 10) == 0:
            h = np.array(historique[-FENETRE_COURBES:])
            print(f"  graine {graine} | pas {pas + 1:7d}/{NB_PAS} | épisodes {len(historique):6d} | "
                  f"retour moyen {h[:, 3].mean():6.3f} | valide trouvée {100 * h[:, 6].mean():5.1f} % | "
                  f"SSL valide moyen {np.nanmean(h[:, 7]):7.2f} dB | ε {eps:.3f} | {time.time() - t_debut:6.1f} s")
    duree = time.time() - t_debut
    etat_reseau = {cle: v.detach().cpu() for cle, v in q.state_dict().items()}
    return etat_reseau, np.array(historique, dtype=float), duree


# =============================================================================
# === Entraînement (une fois par graine) ===
# =============================================================================
etats, historiques, durees = {}, [], {}
for graine in GRAINES:
    print("\n" + "-"*80 + f"\nGRAINE {graine}\n" + "-"*80)
    etats[graine], h, durees[graine] = entrainer(graine)
    historiques.append(h)
    print(f"=== Graine {graine} : {len(h)} épisodes en {durees[graine]:.1f} s ===")
historique_DQN = np.concatenate(historiques, axis=0)


# =============================================================================
# === Sauvegardes (dossier du script) ===
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
hyperparametres = {'HIDDEN': HIDDEN, 'GAMMA': GAMMA, 'LR': LR, 'TAILLE_BUFFER': TAILLE_BUFFER,
                   'TAILLE_LOT': TAILLE_LOT, 'PAS_PAR_MAJ': PAS_PAR_MAJ, 'DEBUT_APPRENTISSAGE': DEBUT_APPRENTISSAGE,
                   'MAJ_CIBLE': MAJ_CIBLE, 'EPS_DEBUT': EPS_DEBUT, 'EPS_FIN': EPS_FIN, 'FRACTION_EPS': FRACTION_EPS,
                   'CLIP_GRAD': CLIP_GRAD, 'NB_PAS': NB_PAS, 'T_MAX': ea.T_MAX, 'ALPHA': ea.ALPHA, 'BETA': ea.BETA,
                   'ECHELLE_J': ea.ECHELLE_J, 'COUT_PAS': ea.COUT_PAS, 'PENALITE_ILLEGAL': ea.PENALITE_ILLEGAL,
                   'PENALITE_INVALIDE': ea.PENALITE_INVALIDE,
                   'NB_ARCHIS_PLAGE': ea.NB_ARCHIS_PLAGE, 'TOL_G_DB': pa.TOL_G_DB, 'TOL_HPBW_DEG': pa.TOL_HPBW_DEG}
torch.save({'graines': GRAINES, 'etats': etats, 'dim_etat': ea.DIM_ETAT, 'nb_actions': ea.NB_ACTIONS,
            'hyperparametres': hyperparametres, 'config_reduite': CONFIG_REDUITE, 'device': str(device)},
           os.path.join(script_dir, 'dqn_model.pt'))
np.save(os.path.join(script_dir, 'historique_DQN.npy'), historique_DQN)
scipy.io.savemat(os.path.join(script_dir, 'historique_DQN.mat'),
                 {'historique_DQN': historique_DQN, 'colonnes': np.array(COLONNES_HISTORIQUE, dtype=object),
                  'graines': np.array(GRAINES), 'duree_s': np.array([durees[g] for g in GRAINES]),
                  **hyperparametres})


# =============================================================================
# === Courbes d'entraînement : moyenne des graines et enveloppe min-max ===
# =============================================================================
VIOLET = '#6a4ce2'   # couleur fixe du DQN dans toutes les figures du second TP


def moyenne_glissante(x, n):
    """Moyenne glissante sur n épisodes (en ignorant les NaN) ; courte au début."""
    x = np.asarray(x, dtype=float)
    valide = ~np.isnan(x)
    c_val = np.cumsum(np.where(valide, x, 0.0))
    c_nb = np.cumsum(valide)
    debut = np.maximum(np.arange(len(x)) - n + 1, 0)
    somme = c_val - np.concatenate([[0.0], c_val])[debut]
    nb = c_nb - np.concatenate([[0], c_nb])[debut]
    return np.where(nb > 0, somme / np.maximum(nb, 1), np.nan)


grille = np.linspace(0, NB_PAS, 200)
courbes = [(3, "Retour par épisode", "Retour (somme des récompenses)", 1),
           (7, "Meilleur SSL valide de l'épisode", "SSL (dB)", 1),
           (6, "Épisodes avec une architecture valide", "Part des épisodes (%)", 100),
           (5, "Évaluations du facteur de réseau par épisode", "Évaluations", 1),
           (10, "Exploration ε", "ε", 1),
           (11, "Perte de Huber (moyenne de l'épisode)", "Perte", 1)]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (col, titre, ylabel, facteur) in zip(axes.flat, courbes):
    valeurs = []
    for h in historiques:
        y = moyenne_glissante(h[:, col], FENETRE_COURBES) * facteur
        ok = ~np.isnan(y)
        valeurs.append(np.interp(grille, h[ok, 2], y[ok], left=np.nan) if ok.any() else np.full_like(grille, np.nan))
    valeurs = np.array(valeurs)
    with warnings.catch_warnings():          # début de courbe sans valeur (avant la première mise à jour) : NaN
        warnings.simplefilter('ignore', RuntimeWarning)
        bas, haut, moy = np.nanmin(valeurs, axis=0), np.nanmax(valeurs, axis=0), np.nanmean(valeurs, axis=0)
    if len(historiques) > 1:
        ax.fill_between(grille, bas, haut, color=VIOLET, alpha=0.2,
                        linewidth=0, label=f'min-max des {len(historiques)} graines')
    ax.plot(grille, moy, color=VIOLET, linewidth=2,
            label='moyenne des graines' if len(historiques) > 1 else f'graine {GRAINES[0]}')
    ax.set_title(titre)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Pas d'environnement")
    ax.grid(True, alpha=0.3)
    if col == 11:
        ax.set_yscale('log')
axes[0, 0].legend(frameon=False)
fig.suptitle(f"Entraînement du DQN (moyenne glissante sur {FENETRE_COURBES} épisodes)")
fig.tight_layout()
fig.savefig(os.path.join(script_dir, 'courbes_DQN.png'), dpi=150)
if AFFICHER_FIGURES:
    plt.show()
plt.close('all')


# =============================================================================
# === Résumé ===
# =============================================================================
print("\n" + "="*80 + "\nRÉSUMÉ DE L'ENTRAÎNEMENT\n" + "="*80)
print("Aucun fichier de données lu : requêtes et architectures générées à la volée.")
print(f"{'Graine':<8}{'Épisodes':>10}{'Durée (s)':>11}{'Retour':>10}{'Valide (%)':>12}"
      f"{'SSL valide (dB)':>17}{'Évaluations':>13}")
print(f"(moyennes sur les {FENETRE_COURBES} derniers épisodes de chaque graine)")
print("-" * 81)
for g, h in zip(GRAINES, historiques):
    f = h[-FENETRE_COURBES:]
    print(f"{g:<8}{len(h):>10}{durees[g]:>11.1f}{f[:, 3].mean():>10.3f}{100 * f[:, 6].mean():>12.1f}"
          f"{np.nanmean(f[:, 7]):>17.2f}{f[:, 5].mean():>13.1f}")
print(f"\nFichiers enregistrés dans {script_dir} :")
print("  dqn_model.pt, historique_DQN.npy, historique_DQN.mat, courbes_DQN.png")
