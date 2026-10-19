# -*- coding: utf-8 -*-
"""
Script 1/4 — Génération du dataset (VQ-VAE, conception inverse d'un réseau d'antennes circulaire concentrique)

Reprend EXACTEMENT la logique de dataset.py de l'enseignant (N. ISMAIL) :
  - mêmes paramètres physiques, même graine (46), même taille (1000), même séquence aléatoire ;
  - nr ∈ {0,...,10} éléments par anneau (0 = anneau vide), au moins un anneau actif ;
  - θ0 tiré uniformément dans [0°, 180°].
On obtient donc les mêmes matrices Minput / Moutput que celles utilisées pour le FFNN
(vérifié : écart 0,0), ce qui garantit une comparaison équitable.

Conventions de l'enseignant (conservées) :
  - Minput  (4 × M) : spécifications [Gain lobe principal (dB), SSL (dB), HPBW (°), θ0 (°)]
  - Moutput (5 × M) : architectures (nombre d'éléments par anneau)
  - une colonne = un échantillon.

Fichiers produits (dans le dossier du script) : Minput.mat, Minput.npy, Moutput.mat, Moutput.npy

Exécution : python dataset.py            (configuration complète, 1000 échantillons)
            python dataset.py --rapide   (configuration réduite pour les tests, 50 échantillons)
"""
import numpy as np
import scipy.io
from scipy.signal import find_peaks
import os
import sys
import time
import random


### Affichage visuel de démarcation (convention de l'enseignant)
def clear_console():
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")

clear_console()


# === Paramètres modifiables (identiques à dataset.py de l'enseignant) ===
CONFIG_REDUITE = "--rapide" in sys.argv[1:]   # configuration réduite pour vérifier la chaîne complète
nb_samples = 50 if CONFIG_REDUITE else 1000
max_rings = 5
max_elements = 10
theta0_max_deg = 180
carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
k = 2 * np.pi / lambda_
r0 = 0.2 * lambda_
delta_r = 0.5 * lambda_
SEED = 46

# Graines fixées (seul le générateur numpy est utilisé ici, comme chez l'enseignant)
random.seed(SEED)
np.random.seed(SEED)

# Grille angulaire et angles de pointage (variables globales lues par la fonction de l'enseignant)
theta = np.linspace(0, 2 * np.pi, 1000)
theta_deg = np.rad2deg(theta)
phi = 0
phi0 = 0
theta0 = 0.0   # mis à jour pour chaque échantillon avant l'appel de la fonction


# =============================================================================
# Fonction de l'enseignant — COPIÉE TELLE QUELLE depuis test_model_FFNN.py (l. 133-207), non modifiée.
# (Le fichier de l'enseignant exécute du code à l'import : chargement de fichiers, test, plt.show().
#  Il ne peut donc pas être importé.)
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


print(f"Configuration : {'RÉDUITE (--rapide)' if CONFIG_REDUITE else 'COMPLÈTE'} - {nb_samples} échantillons")

# === Initialisation des matrices (mêmes noms et formes que l'enseignant) ===
Moutput = np.zeros((max_rings, nb_samples))  # Architecture
Minput = np.zeros((4, nb_samples))           # [MainLobe, SSL, HPBW, theta0]

# === Boucle principale (même séquence de tirages que dataset.py de l'enseignant) ===
t_debut = time.time()
for sample_idx in range(nb_samples):
    # --- Architecture complètement aléatoire : nr ∈ {0..10}, au moins un anneau actif ---
    while True:
        elements_per_ring = np.random.randint(0, max_elements + 1, size=max_rings)
        if np.sum(elements_per_ring > 0) > 0:
            break
    Moutput[:, sample_idx] = elements_per_ring

    # --- Theta0 aléatoire (direction de pointage) ---
    theta0deg = np.random.uniform(0, theta0_max_deg)
    theta0 = np.deg2rad(theta0deg)   # variable globale lue par calcul_AF_performance_metrics

    # --- Spécifications calculées avec le code de l'enseignant ---
    _, HPBW, main_lobe_gain, true_SSL_gain = calcul_AF_performance_metrics(elements_per_ring)

    # --- Stockage ---
    Minput[0, sample_idx] = main_lobe_gain
    Minput[1, sample_idx] = true_SSL_gain
    Minput[2, sample_idx] = HPBW
    Minput[3, sample_idx] = theta0deg

    if (sample_idx + 1) % 200 == 0:
        print(f"  {sample_idx + 1}/{nb_samples} architectures évaluées")

duree = time.time() - t_debut
print(f"\nTemps de génération : {duree:.2f} s, soit {1000 * duree / nb_samples:.2f} ms par architecture")


# === Affichage d'une partie du Minput et Moutput ===
print("\nCaractéristiques de la matrice Minput (spécifications) :")
print(f"- Shape (lignes, colonnes) : {Minput.shape}")
print("- Lignes : [Gain lobe principal (dB), SSL (dB), HPBW (°), theta0 (°)]")
print("-> Premières colonnes (extraits) :")
print(np.round(Minput[:, :5], 4))
print("- Min par ligne :", np.round(Minput.min(axis=1), 2))
print("- Max par ligne :", np.round(Minput.max(axis=1), 2))

print("\nCaractéristiques de la matrice Moutput (architectures) :")
print(f"- Shape (lignes, colonnes) : {Moutput.shape}")
print("-> Premières colonnes (extraits) :")
print(Moutput[:, :5].astype(int))
print(f"- Architectures avec au moins un anneau vide : {int((Moutput == 0).any(axis=0).sum())}/{nb_samples}")


# === Enregistrement en .mat et .npy dans le dossier contenant le script ===
script_dir = os.path.dirname(os.path.abspath(__file__))

scipy.io.savemat(os.path.join(script_dir, 'Minput.mat'), {'Minput': Minput})
scipy.io.savemat(os.path.join(script_dir, 'Moutput.mat'), {'Moutput': Moutput})
np.save(os.path.join(script_dir, 'Minput.npy'), Minput)
np.save(os.path.join(script_dir, 'Moutput.npy'), Moutput)

print(f"\nFichiers enregistrés dans {script_dir} :")
print("  Minput.mat, Minput.npy, Moutput.mat, Moutput.npy")
