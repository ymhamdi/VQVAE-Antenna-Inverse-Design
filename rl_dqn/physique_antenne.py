# -*- coding: utf-8 -*-
"""
Module R0 — Physique du réseau pour le second TP (apprentissage par renforcement, DQN)

Contenu :
  1. La fonction de l'enseignant calcul_AF_performance_metrics, COPIÉE TELLE QUELLE (même copie que dataset.py),
     et son enveloppe metriques_enseignant(archi, θ0), qui fixe la variable globale theta0 avant l'appel.
  2. Une version rapide du même calcul. Le facteur de réseau est une somme de contributions par anneau, qui ne
     dépendent que de (anneau r, N_r, θ0) :
       - diagrammes_anneaux(θ0)    : les 5 × 11 diagrammes d'anneau P[r, N] pour un θ0 (même formule de phase) ;
       - metriques_rapides(n, P)   : une architecture = somme des P[r, n_r], puis les lignes de l'enseignant
                                     (reprises telles quelles) pour G, SSL et HPBW ;
       - metriques_lot(lot, P_t)   : un lot d'architectures (torch, GPU si disponible), mêmes règles vectorisées,
                                     avec repli exact sur les lignes de l'enseignant pour les cas limites
                                     (plateau, ou deux pics à moins de 5 points) ;
       - recherche_exhaustive(...) : les 161 050 architectures pour un θ0, avec ou sans contraintes.
  3. Exécuté directement : vérification de la version rapide contre la fonction de l'enseignant.

Remarque physique (lecture du code de l'enseignant) : φ = φ0 = 0, donc la phase de l'élément n de l'anneau r vaut
  k·a_r·cos(φ_n)·(sin θ − sin θ0). θ0 n'intervient que par sin θ0 : θ0 et 180° − θ0 donnent le même diagramme,
  donc les mêmes G, SSL et HPBW. Le diagramme a toujours deux lobes principaux, en θ0 et en 180° − θ0.

Contraintes (second TP) : une architecture est « valide » pour une requête (θ0, G_min, HPBW_max) si
  G ≥ G_min − TOL_G_DB et HPBW ≤ HPBW_max + TOL_HPBW_DEG.
  TOL_G_DB = 0,05 dB : deux nombres d'éléments voisins diffèrent d'au moins 20·log10(50/49) = 0,175 dB.
  TOL_HPBW_DEG = 0,5° : environ un pas de la grille angulaire (360/999 = 0,36°).

Fichiers produits (dans le dossier du script, seulement quand il est exécuté directement) :
  verification_physique.npy, verification_physique.mat

Exécution : python rl_dqn/physique_antenne.py            (2000 couples (architecture, θ0) + optimum sans contrainte)
            python rl_dqn/physique_antenne.py --rapide   (200 couples)
"""
import numpy as np
import scipy.io
from scipy.signal import find_peaks
import torch
import itertools
import os
import sys
import time


# === Paramètres physiques : copiés de test_model_FFNN.py (l. 15-20 et 123-129), comme dans test_model_VQVAE.py ===
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
theta0 = 0.0   # variable globale lue par calcul_AF_performance_metrics, fixée par metriques_enseignant()

max_elements = 10                                   # n_r ∈ {0, ..., 10}
NB_ARCHITECTURES = (max_elements + 1) ** max_rings - 1   # 161 050 (sans l'architecture vide)
SEED = 46

# Tolérances des contraintes (voir l'en-tête)
TOL_G_DB = 0.05
TOL_HPBW_DEG = 0.5

# Device du calcul par lot (recherche exhaustive, lots d'architectures) : GPU si disponible
DEVICE_AF = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Fonction de l'enseignant — COPIÉE TELLE QUELLE depuis test_model_FFNN.py (l. 133-207), non modifiée.
# Copie identique à celle de dataset.py (comparée ligne à ligne quand ce fichier est exécuté directement).
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


def metriques_enseignant(archi, theta0deg):
    """Fixe θ0 (variable globale lue par la fonction de l'enseignant), puis l'appelle.
    Retourne (AF_dB, G, SSL, HPBW), dans l'ordre de specs_architecture() du script 4."""
    global theta0
    theta0 = np.deg2rad(theta0deg)
    AF_dB, HPBW, gain, SSL = calcul_AF_performance_metrics(np.asarray(archi, dtype=int))
    return AF_dB, gain, SSL, HPBW


def metriques_depuis_AF(AF_az):
    """Lignes de la fonction de l'enseignant reprises TELLES QUELLES, de « AF_norm_az = ... » jusqu'au return.
    Seule différence : le facteur de réseau AF_az est donné au lieu d'être calculé élément par élément.
    Sorties dans l'ordre de l'enseignant : AF_dB_az, HPBW, maxVal_non_norm (= G), true_SSL_gain (= SSL)."""
    # --- début des lignes reprises de calcul_AF_performance_metrics ---
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
    # --- fin des lignes reprises ---


# =============================================================================
# === Version rapide ===
# =============================================================================
EPS = np.finfo(float).eps
RAYONS = r0 + delta_r * np.arange(max_rings)                                # a_r, comme radii chez l'enseignant
THETA_DEG_EXT = np.concatenate((theta_deg - 360, theta_deg, theta_deg + 360))  # comme theta_deg_ext chez l'enseignant
SIN_THETA = np.sin(theta)                                                   # mêmes valeurs que np.sin(theta) de l'enseignant
NB_THETA = len(theta)


def diagrammes_anneaux(theta0deg):
    """P (5, 11, 1000) complexe : P[r, N] = somme des N éléments de l'anneau r pointés vers θ0 (P[r, 0] = 0).
    Même expression de phase que la fonction de l'enseignant (calculée pour les N éléments d'un coup),
    mêmes ajouts successifs élément par élément."""
    theta0_rad = np.deg2rad(theta0deg)
    P = np.zeros((max_rings, max_elements + 1, NB_THETA), dtype=complex)
    for ring in range(max_rings):
        a = RAYONS[ring]
        for N in range(1, max_elements + 1):
            phi_n = 2 * np.pi * np.arange(N) / N
            phase = k * a * (SIN_THETA[None, :] * np.cos(phi - phi_n)[:, None] -
                             np.sin(theta0_rad) * np.cos(phi0 - phi_n)[:, None])     # (N, 1000)
            termes = np.exp(1j * phase)
            for n in range(N):
                P[ring, N] += termes[n]
    return P


def facteur_reseau_rapide(archi, P):
    """AF(θ) d'une architecture = somme, anneau par anneau, des diagrammes précalculés."""
    AF_az = np.zeros(NB_THETA, dtype=complex)
    for ring in range(max_rings):
        if archi[ring] > 0:
            AF_az += P[ring, archi[ring]]
    return AF_az


def metriques_rapides(archi, P):
    """Une architecture, diagrammes d'anneau P déjà calculés pour θ0. Retourne (AF_dB, G, SSL, HPBW)."""
    AF_dB, HPBW, G, SSL = metriques_depuis_AF(facteur_reseau_rapide(archi, P))
    return AF_dB, G, SSL, HPBW


def vers_torch(P, device=DEVICE_AF):
    """Copie exacte des diagrammes d'anneau sur le device du calcul par lot."""
    return torch.as_tensor(P, dtype=torch.complex128, device=device)


def _metriques_AF_lot(AF_t, filtre_SSL=None):
    """Mêmes règles que l'enseignant, vectorisées sur un lot. AF_t : (B, 1000) complexe (torch).
    filtre_SSL(G, HPBW) -> masque des lignes dont on veut le SSL (les autres : NaN). Par défaut : toutes.
    Retourne G, SSL, HPBW (numpy, (B,)) et le nombre de lignes passées par le repli exact."""
    dev = AF_t.device
    AF_abs = AF_t.abs()
    maxVal = AF_abs.max(dim=1).values
    AF_norm = AF_abs / (maxVal + EPS)[:, None]
    AF_dB = 20 * torch.log10(AF_norm + EPS)
    AF_dB = torch.where(AF_dB < -40, torch.full_like(AF_dB, -40.0), AF_dB)     # AF_dB_az[AF_dB_az < -40] = -40
    G = 20 * torch.log10(maxVal + EPS)

    # --- HPBW : premier point à −3 dB de part et d'autre du maximum, sur le diagramme périodisé ---
    maxIdx = AF_dB.argmax(dim=1)                                  # premier maximum, comme np.argmax
    halfPower = AF_dB.max(dim=1).values - 3
    sous = AF_dB <= halfPower[:, None]
    j = torch.arange(NB_THETA, device=dev)
    # Distance (en points, de façon circulaire) du maximum aux points sous −3 dB, à gauche et à droite
    d_gauche = torch.where(sous, torch.remainder(maxIdx[:, None] - j[None, :], NB_THETA), NB_THETA).min(dim=1).values
    d_droite = torch.where(sous, torch.remainder(j[None, :] - maxIdx[:, None], NB_THETA), NB_THETA).min(dim=1).values
    trouve = (d_gauche < NB_THETA).cpu().numpy()
    gauche = (NB_THETA + maxIdx - d_gauche).clamp(0, 3 * NB_THETA - 1).cpu().numpy()   # leftIdx_ext
    droite = (NB_THETA + maxIdx + d_droite).clamp(0, 3 * NB_THETA - 1).cpu().numpy()   # rightIdx_ext
    HPBW = np.where(trouve, THETA_DEG_EXT[droite] - THETA_DEG_EXT[gauche], 180.0)
    G = G.cpu().numpy()

    # --- SSL : plus haut pic à plus de 1 dB sous le plus haut pic (0 s'il n'y en a pas) ---
    SSL = np.full(len(G), np.nan)
    lignes = np.arange(len(G)) if filtre_SSL is None else np.flatnonzero(filtre_SSL(G, HPBW))
    nb_repli = 0
    if len(lignes) > 0:
        lignes_t = torch.as_tensor(lignes, device=dev)
        x = AF_norm[lignes_t]
        centre = x[:, 1:-1]
        pic = (x[:, :-2] < centre) & (x[:, 2:] < centre)          # maxima stricts (indices 1 à 998)
        # Cas où find_peaks pourrait différer des maxima stricts : plateau, ou deux pics à moins de 5 points
        repli = (x[:, 1:] == x[:, :-1]).any(dim=1)
        for d in range(1, 5):
            repli |= (pic[:, :-d] & pic[:, d:]).any(dim=1)
        moins_inf = torch.full_like(centre, -np.inf)
        pic_dB = torch.where(pic, 20 * torch.log10(centre + EPS), moins_inf)
        sommet = pic_dB.max(dim=1).values
        lobe_sec = pic & (pic_dB < sommet[:, None] - 1)
        ssl = torch.where(lobe_sec, pic_dB, moins_inf).max(dim=1).values
        ssl = torch.where(torch.isfinite(ssl), ssl, torch.zeros_like(ssl)).cpu().numpy()
        repli = repli.cpu().numpy()
        # Repli exact : lignes de l'enseignant (find_peaks) sur le facteur de réseau de ces lignes
        for i in np.flatnonzero(repli):
            ssl[i] = metriques_depuis_AF(AF_t[lignes[i]].cpu().numpy())[3]
        nb_repli = int(repli.sum())
        SSL[lignes] = ssl
    return G, SSL, HPBW, nb_repli


def metriques_lot(archis, P_t, filtre_SSL=None):
    """Lot d'architectures (B, 5) pour un même θ0 (P_t = vers_torch(diagrammes_anneaux(θ0))).
    Retourne G, SSL, HPBW (numpy, (B,)) et le nombre de replis exacts."""
    a = torch.as_tensor(np.asarray(archis, dtype=np.int64), device=P_t.device)
    AF = P_t[0, a[:, 0]]
    for ring in range(1, max_rings):
        AF = AF + P_t[ring, a[:, ring]]          # même ordre d'addition que facteur_reseau_rapide
    return _metriques_AF_lot(AF, filtre_SSL)


def est_valide(G, HPBW, G_min, HPBW_max):
    """Contraintes du second TP, avec les tolérances de l'en-tête."""
    return (G >= G_min - TOL_G_DB) & (HPBW <= HPBW_max + TOL_HPBW_DEG)


# Les 11^4 combinaisons (n1, n2, n3, n4), dans l'ordre lexicographique
GRILLE_4_ANNEAUX = np.array(list(itertools.product(range(max_elements + 1), repeat=4)), dtype=np.int64)


def recherche_exhaustive(theta0deg, G_min=-np.inf, HPBW_max=np.inf, device=DEVICE_AF):
    """Les 161 050 architectures à θ0 : SSL minimal parmi les architectures valides (sans contrainte par défaut).
    Le SSL n'est calculé que pour les architectures valides. Retourne un dictionnaire."""
    t_debut = time.time()
    P_t = vers_torch(diagrammes_anneaux(theta0deg), device)
    # Sommes partielles des anneaux 1 à 4, même ordre d'addition que facteur_reseau_rapide
    S = P_t[0][:, None, :] + P_t[1][None, :, :]
    S = S[:, :, None, :] + P_t[2][None, None, :, :]
    S = (S[:, :, :, None, :] + P_t[3][None, None, None, :, :]).reshape(-1, NB_THETA)    # (14641, 1000)

    meilleur = {'SSL': np.inf, 'archi': None, 'G': np.nan, 'HPBW': np.nan}
    nb_valides, nb_repli = 0, 0
    for n5 in range(max_elements + 1):
        AF = S + P_t[4, n5]

        def filtre(G, HPBW, n5=n5):
            v = est_valide(G, HPBW, G_min, HPBW_max)
            if n5 == 0:
                v[0] = False                      # architecture vide [0 0 0 0 0]
            return v

        G, SSL, HPBW, r = _metriques_AF_lot(AF, filtre)
        nb_repli += r
        valides = ~np.isnan(SSL)
        nb_valides += int(valides.sum())
        if valides.any():
            i = int(np.nanargmin(SSL))
            if SSL[i] < meilleur['SSL']:
                meilleur = {'SSL': float(SSL[i]), 'archi': np.append(GRILLE_4_ANNEAUX[i], n5),
                            'G': float(G[i]), 'HPBW': float(HPBW[i])}
    if device.type == "cuda":
        torch.cuda.synchronize()
    meilleur.update({'nb_valides': nb_valides, 'nb_repli': nb_repli, 'duree_s': time.time() - t_debut})
    return meilleur


def tirer_architecture(rng):
    """Architecture aléatoire, même loi que dataset.py : n_r uniforme dans {0..10}, au moins un anneau actif."""
    while True:
        archi = rng.randint(0, max_elements + 1, size=max_rings)
        if np.sum(archi > 0) > 0:
            return archi


# =============================================================================
# === Vérification (exécution directe) ===
# =============================================================================
def _lignes_fonction(chemin):
    """Lignes de calcul_AF_performance_metrics dans un fichier (sans les fins de ligne)."""
    with open(chemin, encoding='utf-8') as f:
        lignes = f.read().splitlines()
    debut = lignes.index("def calcul_AF_performance_metrics(elements_per_ring):")
    fin = lignes.index("    return AF_dB_az, HPBW, maxVal_non_norm, true_SSL_gain", debut)
    return lignes[debut:fin + 1]


if __name__ == "__main__":
    # Console en UTF-8 même si la sortie est redirigée vers un fichier (θ, Δ, ≥ n'existent pas en cp1252)
    sys.stdout.reconfigure(encoding="utf-8")

    ### Affichage visuel de démarcation (convention de l'enseignant)
    def clear_console():
        print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")

    clear_console()
    CONFIG_REDUITE = "--rapide" in sys.argv[1:]
    NB_COUPLES = 200 if CONFIG_REDUITE else 2000
    THETA0_SANS_CONTRAINTE = [] if CONFIG_REDUITE else [5.0, 30.0, 60.0, 85.0, 90.0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.RandomState(SEED)
    print(f"Configuration : {'RÉDUITE (--rapide)' if CONFIG_REDUITE else 'COMPLÈTE'} | "
          f"{NB_COUPLES} couples (architecture, θ0) | calcul par lot sur : {DEVICE_AF}")

    # --- 0) La copie de la fonction de l'enseignant est-elle identique à celle de dataset.py ? ---
    chemin_dataset = os.path.join(script_dir, '..', 'dataset.py')
    if os.path.exists(chemin_dataset):
        copie_identique = _lignes_fonction(chemin_dataset) == _lignes_fonction(os.path.abspath(__file__))
        print(f"\nCopie de calcul_AF_performance_metrics identique à celle de dataset.py : "
              f"{'OUI' if copie_identique else 'NON'}")
    else:
        copie_identique = False
        print("\ndataset.py introuvable : comparaison de la copie impossible.")

    # --- 1) Couples aléatoires (même loi que dataset.py), chacun avec son propre θ0 ---
    archis = np.array([tirer_architecture(rng) for _ in range(NB_COUPLES)])
    theta0s = rng.uniform(0, 180, size=NB_COUPLES)
    ref = np.zeros((NB_COUPLES, 3)); rap = np.zeros((NB_COUPLES, 3)); lot = np.zeros((NB_COUPLES, 3))
    ecart_AF_rap = 0.0
    duree_ens = duree_P = duree_rap = duree_lot = 0.0
    nb_repli_total = 0
    print(f"\n--- 1) {NB_COUPLES} couples : fonction de l'enseignant / version rapide / calcul par lot ---")
    for i in range(NB_COUPLES):
        t = time.perf_counter(); AF_ref, *ref[i] = metriques_enseignant(archis[i], theta0s[i]); duree_ens += time.perf_counter() - t
        t = time.perf_counter(); P = diagrammes_anneaux(theta0s[i]); duree_P += time.perf_counter() - t
        t = time.perf_counter(); AF_rap, *rap[i] = metriques_rapides(archis[i], P); duree_rap += time.perf_counter() - t
        t = time.perf_counter()
        G, SSL, HPBW, r = metriques_lot(archis[i:i + 1], vers_torch(P))
        duree_lot += time.perf_counter() - t
        lot[i] = [G[0], SSL[0], HPBW[0]]
        nb_repli_total += r
        ecart_AF_rap = max(ecart_AF_rap, np.abs(AF_ref - AF_rap).max())
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{NB_COUPLES}")
    ecarts_rap = np.abs(rap - ref).max(axis=0)
    ecarts_lot = np.abs(lot - ref).max(axis=0)
    nb_diff_rap = int((np.abs(rap - ref) > 1e-9).any(axis=1).sum())
    nb_diff_lot = int((np.abs(lot - ref) > 1e-9).any(axis=1).sum())

    # --- 2) Un lot de NB_COUPLES architectures à un même θ0 (vérifie le calcul par lot à grande taille) ---
    theta0_lot = 30.4
    P_t = vers_torch(diagrammes_anneaux(theta0_lot))
    G, SSL, HPBW, nb_repli_lot = metriques_lot(archis, P_t)
    ref_lot = np.array([metriques_enseignant(a, theta0_lot)[1:] for a in archis])
    ecarts_lot_meme_theta0 = np.abs(np.column_stack([G, SSL, HPBW]) - ref_lot).max(axis=0)
    nb_diff_lot_meme_theta0 = int((np.abs(np.column_stack([G, SSL, HPBW]) - ref_lot) > 1e-9).any(axis=1).sum())

    # --- 3) Symétrie θ0 / 180° − θ0 (version rapide) ---
    nb_sym = min(500, NB_COUPLES)
    sym = np.zeros((nb_sym, 3))
    for i in range(nb_sym):
        sym[i] = np.array(metriques_rapides(archis[i], diagrammes_anneaux(180 - theta0s[i]))[1:]) - rap[i]
    ecarts_sym = np.abs(sym).max(axis=0)

    # --- 4) Temps de la recherche exhaustive (une requête d'exemple, sans lien avec le jeu de test) ---
    print("\n--- 4) Recherche exhaustive sur les 161 050 architectures, requête d'exemple "
          "(θ0 = 30°, G ≥ 26,02 dB soit 20 éléments, HPBW ≤ 40°) ---")
    recherche_exhaustive(30.0, 26.02, 40.0)            # premier appel : initialisation du GPU, non chronométré
    ex = recherche_exhaustive(30.0, 26.02, 40.0)
    G_ens_ex, SSL_ens_ex, HPBW_ens_ex = metriques_enseignant(ex["archi"], 30.0)[1:]
    print(f"  optimum {ex['archi']} : SSL {ex['SSL']:.3f} dB ({ex['nb_valides']} architectures valides, "
          f"{ex['nb_repli']} replis exacts) en {ex['duree_s']:.2f} s")
    print(f"  recalculé par la fonction de l'enseignant : G {G_ens_ex:.3f} dB, SSL {SSL_ens_ex:.3f} dB, HPBW {HPBW_ens_ex:.2f}°")

    # --- 5) Optimum sans contrainte pour quelques θ0 (pourquoi des contraintes sont nécessaires) ---
    sans_contrainte = []
    if THETA0_SANS_CONTRAINTE:
        print("\n--- 5) SSL minimal sans aucune contrainte (161 050 architectures) ---")
    for t0 in THETA0_SANS_CONTRAINTE:
        ex0 = recherche_exhaustive(t0)
        G_e, SSL_e, HPBW_e = metriques_enseignant(ex0['archi'], t0)[1:]
        sans_contrainte.append([t0, *ex0['archi'], ex0['archi'].sum(), SSL_e, G_e, HPBW_e])
        print(f"  θ0 = {t0:5.1f}° : {ex0['archi']} ({ex0['archi'].sum():2d} éléments) | SSL {SSL_e:7.2f} dB | "
              f"G {G_e:5.2f} dB | HPBW {HPBW_e:6.2f}° (recalculés par l'enseignant ; {ex0['duree_s']:.1f} s)")
    sans_contrainte = np.array(sans_contrainte).reshape(-1, 10)

    # --- Résumé ---
    print("\n" + "="*80 + "\nRÉSUMÉ DE LA VÉRIFICATION\n" + "="*80)
    print(f"Copie de la fonction de l'enseignant identique à dataset.py : {'OUI' if copie_identique else 'NON'}")
    print(f"{NB_COUPLES} couples (architecture, θ0) aléatoires, écart maximal avec la fonction de l'enseignant :")
    print(f"  version rapide   : G {ecarts_rap[0]:.2e} dB | SSL {ecarts_rap[1]:.2e} dB | HPBW {ecarts_rap[2]:.2e}° | "
          f"diagramme en dB {ecart_AF_rap:.2e} | couples avec un écart > 1e-9 : {nb_diff_rap}")
    print(f"  calcul par lot   : G {ecarts_lot[0]:.2e} dB | SSL {ecarts_lot[1]:.2e} dB | HPBW {ecarts_lot[2]:.2e}° | "
          f"couples avec un écart > 1e-9 : {nb_diff_lot} | replis exacts : {nb_repli_total}")
    print(f"  lot de {NB_COUPLES} à θ0 = {theta0_lot}° : G {ecarts_lot_meme_theta0[0]:.2e} dB | "
          f"SSL {ecarts_lot_meme_theta0[1]:.2e} dB | HPBW {ecarts_lot_meme_theta0[2]:.2e}° | "
          f"écarts > 1e-9 : {nb_diff_lot_meme_theta0} | replis exacts : {nb_repli_lot}")
    print(f"Symétrie θ0 / 180° − θ0 ({nb_sym} architectures) : écart max G {ecarts_sym[0]:.2e} dB, "
          f"SSL {ecarts_sym[1]:.2e} dB, HPBW {ecarts_sym[2]:.2e}°")
    print(f"Temps moyens : enseignant {1000 * duree_ens / NB_COUPLES:.3f} ms par architecture | "
          f"diagrammes d'anneau {1000 * duree_P / NB_COUPLES:.3f} ms par θ0 | "
          f"version rapide {1000 * duree_rap / NB_COUPLES:.3f} ms par architecture | "
          f"lot de 1 {1000 * duree_lot / NB_COUPLES:.3f} ms")
    print(f"Recherche exhaustive (161 050 architectures, {DEVICE_AF}) : {ex['duree_s']:.2f} s par θ0 "
          f"(estimation avec la fonction de l'enseignant : {NB_ARCHITECTURES * duree_ens / NB_COUPLES:.0f} s)")

    # === Enregistrement en .npy et .mat dans le dossier du script ===
    noms = ['copie_identique', 'nb_couples',
            'ecart_max_G_rapide_dB', 'ecart_max_SSL_rapide_dB', 'ecart_max_HPBW_rapide_deg', 'ecart_max_AF_dB_rapide',
            'nb_couples_ecart_rapide', 'ecart_max_G_lot_dB', 'ecart_max_SSL_lot_dB', 'ecart_max_HPBW_lot_deg',
            'nb_couples_ecart_lot', 'nb_replis_lot', 'ecart_max_G_lot_meme_theta0_dB', 'ecart_max_SSL_lot_meme_theta0_dB',
            'ecart_max_HPBW_lot_meme_theta0_deg', 'nb_ecarts_lot_meme_theta0',
            'ecart_max_symetrie_G_dB', 'ecart_max_symetrie_SSL_dB', 'ecart_max_symetrie_HPBW_deg',
            'temps_enseignant_ms', 'temps_diagrammes_anneaux_ms', 'temps_rapide_ms', 'temps_exhaustive_s']
    valeurs = np.array([float(copie_identique), NB_COUPLES, *ecarts_rap, ecart_AF_rap, nb_diff_rap, *ecarts_lot,
                        nb_diff_lot, nb_repli_total, *ecarts_lot_meme_theta0, nb_diff_lot_meme_theta0, *ecarts_sym,
                        1000 * duree_ens / NB_COUPLES, 1000 * duree_P / NB_COUPLES, 1000 * duree_rap / NB_COUPLES,
                        ex['duree_s']])
    np.save(os.path.join(script_dir, 'verification_physique.npy'), valeurs)
    scipy.io.savemat(os.path.join(script_dir, 'verification_physique.mat'), {
        **dict(zip(noms, valeurs)), 'noms': np.array(noms, dtype=object),
        'optimum_sans_contrainte': sans_contrainte,
        'colonnes_optimum_sans_contrainte': np.array(['theta0_deg', 'n1', 'n2', 'n3', 'n4', 'n5', 'nb_elements',
                                                      'SSL_dB', 'G_dB', 'HPBW_deg'], dtype=object),
        'device': str(DEVICE_AF)})
    print(f"\nFichiers enregistrés dans {script_dir} :")
    print("  verification_physique.npy, verification_physique.mat")
