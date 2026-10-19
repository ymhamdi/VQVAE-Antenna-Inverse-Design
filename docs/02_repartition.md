# Script 2 : répartition training / test (`repartition_training_test.py`)

[← Retour à l'index de la documentation](README.md)

## Rôle

Ce script mélange les 1000 échantillons et les sépare en **900 pour le training (90 %)** et **100 pour le test (10 %)**. La logique et la graine (46) sont celles de l'enseignant. On obtient donc **le même split que celui du FFNN** ; on l'a vérifié fichier par fichier.

Ce point est essentiel pour la comparaison : le FFNN de l'enseignant (`ffnn_model.pth`) a été entraîné sur ces 900 exemples et n'a **jamais vu les 100 exemples de test**. Les deux modèles sont donc évalués sur les mêmes données inédites.

```
python repartition_training_test.py     # moins d'une seconde
```

## Déroulement

1. Charger `Minput.npy` et `Moutput.npy` (produits par le [script 1](01_dataset.md)).
2. `np.random.seed(46)`, puis `rand_indices = np.random.permutation(1000)`.
3. Les 100 premiers indices vont au test, les 900 suivants au training.
4. Sélectionner les colonnes correspondantes et sauvegarder.

## Fichiers produits

Chaque fichier existe en `.mat` et en `.npy`. Dans le `.mat`, la variable porte le nom du fichier. Une colonne = un échantillon.

| Fichier | Forme | Contenu |
|---|---|---|
| `Minput_training` | 4 × 900 | spécifications du training [G, SSL, HPBW, θ0] |
| `Moutput_training` | 5 × 900 | architectures du training |
| `Minput_test` | 4 × 100 | spécifications du test, c'est-à-dire les « cahiers des charges » s* à satisfaire |
| `Moutput_test` | 5 × 100 | architectures de référence du test |
| `train_indices` | 900 | colonnes de `Minput`/`Moutput` mises dans le training |
| `test_indices` | 100 | colonnes de `Minput`/`Moutput` mises dans le test |

Les indices sont **numérotés à partir de 0** (convention Python). Sous MATLAB, il faut ajouter 1.
