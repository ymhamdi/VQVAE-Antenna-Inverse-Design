# VQ-VAE : conception inverse d'un réseau d'antennes circulaire concentrique

TP d'intelligence artificielle, INSAT, filière IIA3. Enseignant : N. ISMAIL.

On cherche l'architecture d'un réseau de **5 anneaux concentriques** (f = 2,45 GHz), c'est-à-dire le nombre d'éléments sur chaque anneau, de 0 à 10. Elle doit atteindre des spécifications données : gain du lobe principal G, niveau du lobe secondaire SSL, ouverture à −3 dB HPBW, pour une direction de pointage θ0.

Ce problème inverse a **plusieurs solutions**. Un VQ-VAE associé à un prior conditionnel propose donc environ 300 architectures candidates. Le facteur de réseau, calculé avec le code de l'enseignant, sert à retenir la meilleure. Les résultats sont comparés au FFNN fourni par l'enseignant, sur les mêmes données.

## Résultat principal (100 exemples de test inédits)

| | VQ-VAE | FFNN de l'enseignant |
|---|---|---|
| Erreur globale moyenne (formule de l'enseignant) | **0,61 %** | 3,37 % |
| MAE G / SSL / HPBW | 0,13 dB / 0,24 dB / 1,49° | 0,33 dB / 2,98 dB / 2,70° |
| Temps par exemple | 1,04 s | 0,003 s |

À nuancer : une recherche aléatoire de 301 architectures, avec la même vérification physique, atteint déjà 0,72 %. Voir [Résultats et limites](docs/05_resultats.md).

## Démarrage rapide

Dépendances : `numpy`, `scipy`, `torch`, `matplotlib`. Tout tourne sur CPU ; le GPU est utilisé s'il est disponible.

```
python dataset.py                     # 1. génération du dataset            (environ 2 à 5 s)
python repartition_training_test.py   # 2. répartition 90 % / 10 %          (< 1 s)
python VQVAE_training.py              # 3. entraînement VQ-VAE puis prior   (environ 25 s)
python test_model_VQVAE.py            # 4. test et comparaison avec le FFNN (environ 2 min)
```

Chaque script lit et écrit ses fichiers **dans le dossier des scripts**, quel que soit le dossier d'où on le lance. Pour tester la chaîne rapidement, utiliser `python dataset.py --rapide` (50 échantillons) et `python VQVAE_training.py --rapide` (2 époques par étage). Les scripts 2 et 4 s'adaptent seuls.

Les fichiers de résultats de la configuration complète sont déjà dans le dépôt. On peut donc les consulter sans rien relancer.

## Documentation : une page par élément

| Élément | Fichiers | Documentation |
|---|---|---|
| Script 1 : dataset | `dataset.py`, puis `Minput`, `Moutput` (`.mat` / `.npy`) | [docs/01_dataset.md](docs/01_dataset.md) |
| Script 2 : répartition | `repartition_training_test.py`, puis `Minput_training`, `Minput_test`, `Moutput_training`, `Moutput_test`, `train_indices`, `test_indices` | [docs/02_repartition.md](docs/02_repartition.md) |
| Script 3 : entraînement | `VQVAE_training.py`, puis `vqvae_model.pt`, `prior_model.pt`, `normalization_params_VQVAE`, `historique_*`, `courbes_VQVAE.png`, `courbe_prior.png` | [docs/03_entrainement.md](docs/03_entrainement.md) |
| Script 4 : test | `test_model_VQVAE.py`, puis `*_VQVAE`, `*_FFNN`, `comparaison_VQVAE_FFNN`, `diagramme_exemple_*.png`, `distribution_erreurs.png` | [docs/04_test.md](docs/04_test.md) |
| Résultats | tableaux, figures, limites, pistes | [docs/05_resultats.md](docs/05_resultats.md) |
| FFNN de l'enseignant | `ffnn_model.pth`, `normalization_params.npz` | [docs/04_test.md](docs/04_test.md#comparaison-avec-le-ffnn) |

## Choix importants

- **Même dataset et même split que le FFNN** (graine 46, 1000 échantillons, 900 / 100), vérifiés fichier par fichier. La comparaison est donc équitable.
- **n_r ∈ {0,…,10}** comme chez l'enseignant, où 0 = anneau vide. Il y a donc 11 classes par anneau.
- **Convention de noms de l'enseignant** : `Minput` = spécifications, `Moutput` = architectures, une colonne = un échantillon.
- **Le code de l'enseignant est copié, pas réimplémenté** : la fonction du facteur de réseau et le calcul de l'erreur ont été vérifiés identiques à l'original. Ses fichiers ne peuvent pas être importés, car ils exécutent du code à l'import.
- Les numéros d'équations cités dans le code renvoient au rapport de synthèse théorique du groupe, qui n'est pas inclus dans le dépôt.

## Crédits et licence

- La fonction `calcul_AF_performance_metrics`, les lignes de calcul de l'erreur de prédiction, la classe `FFNN`, ainsi que les fichiers `ffnn_model.pth` et `normalization_params.npz`, sont l'œuvre de l'enseignant, **N. ISMAIL** (codes du TP). Ils sont reproduits pour la comparaison et signalés dans le code par des commentaires.
- Le reste du code est publié sous [licence MIT](LICENSE).
