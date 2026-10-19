# Script 1 : génération du dataset (`dataset.py`)

[← Retour à l'index de la documentation](README.md)

## Rôle

Ce script tire 1000 architectures de réseau au hasard. Pour chacune, il calcule ses spécifications avec le code de l'enseignant. Le résultat est **exactement le dataset du FFNN de l'enseignant** (même graine, même séquence de tirages). On l'a vérifié fichier par fichier, avec un écart de 0,0. La comparaison VQ-VAE / FFNN se fait donc sur les mêmes données.

```
python dataset.py            # configuration complète : 1000 échantillons (environ 2 à 5 s)
python dataset.py --rapide   # configuration réduite : 50 échantillons, pour tester la chaîne
```

## Le problème physique

Le réseau comporte **5 anneaux concentriques**, à f = 2,45 GHz (λ ≈ 12,24 cm) :

| Anneau r | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Rayon a_r = r0 + (r−1)·Δr | 0,2λ | 0,7λ | 1,2λ | 1,7λ | 2,2λ |

L'anneau r porte **n_r ∈ {0, 1, …, 10} éléments**, régulièrement espacés (φ_n = 2πn / n_r) ; n_r = 0 veut dire que l'anneau est vide. Une **architecture** est le vecteur n = [n1, …, n5], avec au moins un anneau non vide. Cela fait 11⁵ − 1 = 161 050 architectures possibles.

Le faisceau est pointé dans la direction θ0 (avec φ0 = 0). Le facteur de réseau est calculé dans la coupe φ = 0, sur 1000 angles θ ∈ [0, 2π] :

$$\mathrm{AF}(\theta)=\sum_{r=1}^{5}\sum_{n=0}^{n_r-1}\exp\Big(j\,k\,a_r\big[\sin\theta\cos(\phi-\phi_n)-\sin\theta_0\cos(\phi_0-\phi_n)\big]\Big),\qquad k=\frac{2\pi}{\lambda}$$

### Les quatre spécifications s = (G, SSL, HPBW, θ0)

| Spécification | Définition (code de l'enseignant) | Unité | Plage dans le dataset |
|---|---|---|---|
| **G**, gain du lobe principal | 20·log10(max \|AF\|), non normalisé | dB | 12,0 à 33,4 |
| **SSL**, lobe secondaire | niveau du plus haut lobe secondaire, relatif au lobe principal. Les pics à moins de 1 dB du maximum comptent comme lobes principaux | dB (négatif) | −38,7 à −1,3 |
| **HPBW**, ouverture à −3 dB | largeur angulaire du lobe principal à mi-puissance (180 si incalculable) | ° | 11,2 à 113,5 |
| **θ0**, direction de pointage | tirée uniformément dans [0°, 180°] | ° | 0,1 à 179,7 |

> **À retenir.** G ne dépend que du **nombre total d'éléments** : dans la direction de pointage, toutes les phases s'alignent et G = 20·log10(Σ n_r). Par exemple, [8, 4, 10, 0, 9] et [9, 1, 9, 5, 7] ont chacune 31 éléments, donc le même G = 29,83 dB. Beaucoup d'architectures différentes ont ainsi des spécifications voisines : **le problème inverse n'a pas de solution unique.** C'est la raison d'être du VQ-VAE (voir [script 3](03_entrainement.md)).

## Le code de l'enseignant (copié, non modifié)

La fonction `calcul_AF_performance_metrics(elements_per_ring)` est **copiée telle quelle** depuis `test_model_FFNN.py` (lignes 133 à 207). On ne peut pas l'importer, car le fichier de l'enseignant exécute du code à l'import (chargement de fichiers, test, `plt.show()`). Une vérification automatique a confirmé que la copie est identique à l'original.

- **Entrée** : `elements_per_ring`, 5 entiers.
- **Sorties, dans cet ordre** : `AF_dB_az` (diagramme normalisé en dB, 1000 points, écrêté à −40 dB), `HPBW` (°), `maxVal_non_norm` (= G, en dB), `true_SSL_gain` (= SSL, en dB).
- ⚠️ **Piège** : la fonction lit la direction de pointage dans la **variable globale `theta0`, en radians**. Il faut donc écrire `theta0 = np.deg2rad(theta0deg)` avant chaque appel. Le script le fait dans sa boucle, comme l'enseignant.
- **Temps de calcul** : 1,6 à 4,6 ms par architecture selon la charge de la machine, et environ 7,7 ms au pire (10 éléments sur chaque anneau).

## Déroulement

1. Fixer les graines (`np.random.seed(46)`).
2. Pour chaque échantillon :
   - tirer `np.random.randint(0, 11, size=5)`, jusqu'à obtenir au moins un anneau non vide ;
   - tirer `θ0 ~ U(0°, 180°)` ;
   - calculer (G, SSL, HPBW) avec la fonction de l'enseignant.
3. Sauvegarder en `.mat` et en `.npy`.

## Fichiers produits

La convention de l'enseignant est conservée : **une colonne = un échantillon**.

| Fichier | Forme | Contenu |
|---|---|---|
| `Minput.mat` / `Minput.npy` | 4 × 1000 | spécifications, lignes [G (dB), SSL (dB), HPBW (°), θ0 (°)] |
| `Moutput.mat` / `Moutput.npy` | 5 × 1000 | architectures, nombre d'éléments par anneau (stocké en flottants, valeurs entières de 0 à 10) |

Dans le `.mat`, chaque variable porte le nom du fichier (`Minput`, `Moutput`).

> **Attention au nommage** : « input » désigne l'**entrée du modèle inverse**, c'est-à-dire les spécifications. « output » désigne ce que le modèle doit proposer, c'est-à-dire l'architecture. C'est la convention de l'enseignant.

372 architectures sur 1000 ont au moins un anneau vide.
