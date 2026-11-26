# Documentation

Une page par élément du projet, à lire dans l'ordre :

| Page | Contenu |
|---|---|
| [1. Dataset](01_dataset.md) | le problème physique, les 4 spécifications (G, SSL, HPBW, θ0), le code de l'enseignant et ses pièges, `Minput` / `Moutput` |
| [2. Répartition training / test](02_repartition.md) | le split 90 / 10, identique à celui du FFNN, et les indices |
| [3. Entraînement](03_entrainement.md) | le VQ-VAE (étage A), le prior conditionnel (étage B), les pertes, les hyperparamètres, les historiques |
| [4. Test](04_test.md) | la conception inverse (candidats et critère (25)), l'erreur de l'enseignant, la comparaison avec le FFNN, les fichiers de résultats |
| [5. Résultats et limites](05_resultats.md) | les chiffres, les figures, ce qu'il faut en conclure et ce qu'on peut améliorer |
| [Second TP : DQN](../rl_dqn/README.md) | apprentissage par renforcement sans dataset : minimiser le SSL sous contraintes de gain et d'ouverture, comparé à la recherche aléatoire et à l'optimum exhaustif |

Pour démarrer vite, lire le [README principal](../README.md), puis la page 5.
