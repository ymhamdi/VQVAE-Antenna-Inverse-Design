# -*- coding: utf-8 -*-
"""
Script 2/4 — Répartition training / test (VQ-VAE)

Reprend EXACTEMENT la logique de repartition_training_test.py de l'enseignant :
  - graine 46, mélange par np.random.permutation ;
  - 10 % test / 90 % training.
Sur le dataset complet (1000 échantillons), on obtient donc le MÊME split que celui du FFNN
(900 training / 100 test). Le FFNN de l'enseignant n'a jamais vu nos exemples de test,
ce qui permet de le comparer au VQ-VAE sur les mêmes exemples.

Entrées  : Minput.npy (4 × M), Moutput.npy (5 × M) produits par dataset.py
Fichiers produits (dans le dossier du script), en .mat ET en .npy :
  Minput_training, Minput_test, Moutput_training, Moutput_test   (une colonne = un échantillon)
  train_indices, test_indices   (indices des colonnes de Minput/Moutput, numérotés à partir de 0)

Exécution : python repartition_training_test.py
"""
import numpy as np
import scipy.io
import os
import random


### Affichage visuel de démarcation (convention de l'enseignant)
def clear_console():
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")

clear_console()


# === Paramètres modifiables ===
SEED = 46              # même graine que l'enseignant
FRACTION_TEST = 0.10   # même proportion que l'enseignant (10 % test / 90 % training)

random.seed(SEED)
np.random.seed(SEED)

# === Chargement des matrices Minput et Moutput ===
script_dir = os.path.dirname(os.path.abspath(__file__))
Minput = np.load(os.path.join(script_dir, 'Minput.npy'))
Moutput = np.load(os.path.join(script_dir, 'Moutput.npy'))

# === Mélange aléatoire des indices ===
total_cols = Minput.shape[1]
rand_indices = np.random.permutation(total_cols)

# === Séparation test / training ===
nb_test = round(FRACTION_TEST * total_cols)
test_indices = rand_indices[:nb_test]
train_indices = rand_indices[nb_test:]

# === Sélection des colonnes ===
Minput_test = Minput[:, test_indices]
Minput_training = Minput[:, train_indices]
Moutput_test = Moutput[:, test_indices]
Moutput_training = Moutput[:, train_indices]


# === Affichage ===
print(f"Nombre total d'échantillons : {total_cols}")
print(f"Training : {len(train_indices)} échantillons | Test : {len(test_indices)} échantillons")

print("\nCaractéristiques de la matrice Minput_training :")
print(f"- Shape (lignes, colonnes) : {Minput_training.shape}")
print("-> Premières colonnes (extraits) :")
print(np.round(Minput_training[:, :5], 4))

print("\nCaractéristiques de la matrice Moutput_training :")
print(f"- Shape (lignes, colonnes) : {Moutput_training.shape}")
print("-> Premières colonnes (extraits) :")
print(Moutput_training[:, :5].astype(int))

print("\nCaractéristiques de la matrice Minput_test :")
print(f"- Shape (lignes, colonnes) : {Minput_test.shape}")
print("-> Premières colonnes (extraits) :")
print(np.round(Minput_test[:, :5], 4))

print("\nCaractéristiques de la matrice Moutput_test :")
print(f"- Shape (lignes, colonnes) : {Moutput_test.shape}")
print("-> Premières colonnes (extraits) :")
print(Moutput_test[:, :5].astype(int))


# === Enregistrement en .mat et .npy dans le dossier contenant le script ===
matrices = {
    'Minput_training': Minput_training,
    'Moutput_training': Moutput_training,
    'Minput_test': Minput_test,
    'Moutput_test': Moutput_test,
    'train_indices': train_indices.astype(np.int64),
    'test_indices': test_indices.astype(np.int64),
}
for nom, matrice in matrices.items():
    scipy.io.savemat(os.path.join(script_dir, nom + '.mat'), {nom: matrice})
    np.save(os.path.join(script_dir, nom + '.npy'), matrice)

print(f"\nFichiers enregistrés dans {script_dir} :")
for nom in matrices:
    print(f"  {nom}.mat, {nom}.npy")
print("(train_indices / test_indices : indices Python, numérotés à partir de 0)")
