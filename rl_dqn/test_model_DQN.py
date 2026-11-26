# -*- coding: utf-8 -*-
"""
Script R2/R2 — Évaluation du DQN sur les 100 spécifications de test du premier TP et comparaison avec d'autres méthodes

Les 100 spécifications de test (Minput_test, split du FFNN de l'enseignant) ne sont lues QUE dans ce script, pour
l'évaluation finale : le DQN ne les a jamais vues pendant l'entraînement (R1 ne lit aucun fichier de données).

Requête n° i : θ0 = θ0 de test, G_min = G de test, HPBW_max = HPBW de test (tolérances : 0,05 dB et 0,5°,
voir physique_antenne.py). L'architecture de référence du test satisfait ces contraintes par construction.
Objectif : l'architecture valide de SSL minimal.

Méthodes comparées :
  - Référence, VQ-VAE et FFNN : architectures lues dans les fichiers du premier TP (Moutput_test,
    Moutput_pred_VQVAE, Moutput_pred_FFNN ; modèles non relancés), notées avec la fonction de l'enseignant.
    ATTENTION : le VQ-VAE et le FFNN visent le SSL DEMANDÉ (et le G demandé), pas le SSL le plus bas.
    Leur colonne compare des objectifs différents, pas la qualité des modèles.
  - DQN : déroulé glouton (argmax Q) depuis un départ aléatoire tiré avec une graine fixe (le même pour toutes
    les graines du DQN et pour la recherche locale), réponse = meilleure architecture valide vue ;
  - recherche aléatoire : autant d'évaluations du facteur de réseau que le DQN sur cette spécification ;
  - recherche locale aléatoire : mêmes coups ±1 que le DQN, même départ, même budget d'évaluations ;
    un voisin est accepté s'il améliore le score J ; nouveau départ aléatoire dans un optimum local ;
  - recherche exhaustive : les 161 050 architectures (version rapide, GPU si disponible) = l'optimum.
  Toutes les réponses finales sont recalculées avec la fonction de l'enseignant.

Fichiers produits (dans le dossier du script) :
  resultats_DQN.npy / .mat           par méthode (et graine) et par spécification : architecture, G, SSL, HPBW,
                                     validité, évaluations, temps (noms dans le .mat), trajectoire du DQN
  comparaison_methodes.npy / .mat    tableau comparatif (moyenne sur les graines ; écart-type dans le .mat)
  boxplot_SSL_methodes.png           SSL et écart à l'optimum par méthode
  diagrammes_DQN_XX.png              diagrammes Référence / VQ-VAE / DQN / optimum
  trajectoire_DQN.png                déroulé pas à pas du DQN sur une spécification

Exécution : python rl_dqn/test_model_DQN.py            (les 100 spécifications de test)
            python rl_dqn/test_model_DQN.py --rapide   (5 spécifications)
"""
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import os
import sys
import time
import random
import matplotlib

import physique_antenne as pa
import env_antenne as ea

sys.stdout.reconfigure(encoding="utf-8")   # console en UTF-8 même si la sortie est redirigée


### Affichage visuel de démarcation (convention de l'enseignant)
def clear_console():
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")

clear_console()


# =============================================================================
# === Paramètres ===
# =============================================================================
CONFIG_REDUITE = "--rapide" in sys.argv[1:]
AFFICHER_FIGURES = False                 # True : ouvre les fenêtres matplotlib (bloquant)
SEED = 46                                # départs des épisodes de test (et graine de référence pour les figures)
NB_SPECS = 5 if CONFIG_REDUITE else 100
INDICES_FIGURES = [0, 10, 42, 77]        # mêmes exemples que les figures du VQ-VAE
INDICE_TRAJECTOIRE = 10                  # exemple utilisé par l'enseignant

if not AFFICHER_FIGURES:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"Configuration : {'RÉDUITE (--rapide)' if CONFIG_REDUITE else 'COMPLÈTE'} | {NB_SPECS} spécifications | "
      f"recherche exhaustive sur : {pa.DEVICE_AF}")


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


# =============================================================================
# === Chargement (fichiers du premier TP, à la racine du dépôt, en lecture seule) ===
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
racine = os.path.normpath(os.path.join(script_dir, '..'))
Minput_test = np.load(os.path.join(racine, 'Minput_test.npy')).T[:NB_SPECS]                       # G, SSL, HPBW, θ0
archis_lues = {'Référence': np.load(os.path.join(racine, 'Moutput_test.npy')).T[:NB_SPECS],
               'VQ-VAE': np.load(os.path.join(racine, 'Moutput_pred_VQVAE.npy')).T[:NB_SPECS],
               'FFNN': np.load(os.path.join(racine, 'Moutput_pred_FFNN.npy')).T[:NB_SPECS]}
archis_lues = {nom: np.rint(a).astype(np.int64) for nom, a in archis_lues.items()}
theta0s, G_mins, HPBW_maxs = Minput_test[:, 3], Minput_test[:, 0], Minput_test[:, 2]
N_ref = archis_lues['Référence'].sum(axis=1)                     # G_min ⇔ au moins N_ref éléments
n_specs = len(theta0s)

# Coûts du VQ-VAE et du FFNN : run committé du premier TP (non relancés ici)
evals_lues = {'Référence': np.nan, 'VQ-VAE': np.nan, 'FFNN': 1.0}
temps_lus = {'Référence': np.nan, 'VQ-VAE': np.nan, 'FFNN': np.nan}
chemin_pred = os.path.join(racine, 'predictions_VQVAE.mat')
if os.path.exists(chemin_pred):
    evals_lues['VQ-VAE'] = float(np.mean(scipy.io.loadmat(chemin_pred)['nb_candidats_uniques']))
chemin_comp = os.path.join(racine, 'comparaison_VQVAE_FFNN.npy')
if os.path.exists(chemin_comp):
    temps_lus['VQ-VAE'], temps_lus['FFNN'] = np.load(chemin_comp)[9]      # ligne « temps d'inférence par exemple »

modele = torch.load(os.path.join(script_dir, 'dqn_model.pt'), map_location='cpu')
GRAINES = list(modele['graines'])
reseaux = {}
for g in GRAINES:
    q = ReseauQ(modele['dim_etat'], modele['nb_actions'], modele['hyperparametres']['HIDDEN'])
    q.load_state_dict(modele['etats'][g])
    q.eval()
    reseaux[g] = q
print(f"DQN chargé : graines {GRAINES}{' (modèle --rapide)' if modele['config_reduite'] else ''}")


# =============================================================================
# === Outils ===
# =============================================================================
def cle(a):
    return tuple(int(v) for v in a)


def meilleure_reponse(cache, G_min, HPBW_max):
    """Réponse d'une recherche : architecture valide de SSL minimal, sinon meilleur score J. (archi, valide)"""
    valides = [(SSL, a) for a, (G, SSL, HPBW) in cache.items() if pa.est_valide(G, HPBW, G_min, HPBW_max)]
    if valides:
        return np.array(min(valides)[1]), True
    J = [(ea.score_J(G, SSL, HPBW, G_min, HPBW_max), a) for a, (G, SSL, HPBW) in cache.items()]
    return np.array(max(J)[1]), False


def deroule_DQN(q, requete, depart, P):
    """Déroulé glouton du DQN. Retourne (réponse, valide, nb d'évaluations, trajectoire)."""
    env = ea.EnvAntenne(np.random.RandomState(0))
    etat = env.reset(requete=requete, depart=depart, P=P)
    trajectoire = [[-1, *env.archi, env.G, env.SSL, env.HPBW]]
    termine = False
    while not termine:
        with torch.no_grad():
            action = int(q(torch.as_tensor(etat)[None]).argmax(dim=1).item())
        etat, _, termine, _ = env.step(action)
        trajectoire.append([action, *env.archi, env.G, env.SSL, env.HPBW])
    archi, valide = env.reponse()
    return archi, valide, len(env.cache), np.array(trajectoire)


def recherche_aleatoire(G_min, HPBW_max, budget, P, rng):
    """`budget` architectures distinctes tirées au hasard (loi de dataset.py)."""
    cache = {}
    while len(cache) < budget:
        a = pa.tirer_architecture(rng)
        if cle(a) not in cache:
            cache[cle(a)] = pa.metriques_rapides(a, P)[1:]
    return (*meilleure_reponse(cache, G_min, HPBW_max), len(cache))


def recherche_locale(G_min, HPBW_max, depart, budget, P, rng):
    """Recherche locale aléatoire : voisin ±1 non évalué tiré au hasard, accepté s'il améliore J ;
    nouveau départ aléatoire quand tous les voisins sont évalués sans amélioration (optimum local)."""
    cache = {}

    def evaluer(a):
        if cle(a) not in cache:
            cache[cle(a)] = pa.metriques_rapides(a, P)[1:]
        return ea.score_J(*cache[cle(a)], G_min, HPBW_max)

    courant = np.asarray(depart, dtype=np.int64)
    J_courant = evaluer(courant)
    while len(cache) < budget:
        voisins = [v for v in (ea.appliquer_action(courant, a) for a in range(ea.NB_ACTIONS - 1))
                   if v is not None and cle(v) not in cache]
        if not voisins:
            courant = pa.tirer_architecture(rng)
            J_courant = evaluer(courant)
            continue
        v = voisins[rng.randint(len(voisins))]
        J_v = evaluer(v)
        if J_v > J_courant:
            courant, J_courant = v, J_v
    return (*meilleure_reponse(cache, G_min, HPBW_max), len(cache))


# =============================================================================
# === 1) Recherches sur chaque spécification de test ===
# =============================================================================
rng_departs = np.random.RandomState(SEED)
departs = np.array([pa.tirer_architecture(rng_departs) for _ in range(n_specs)])

# reponses[methode] : (n_specs, 5) ; evals / temps : (n_specs,) ; une entrée par graine pour les méthodes aléatoires
reponses, valides_rapide, evaluations, temps = {}, {}, {}, {}
for g in GRAINES:
    for m in ['DQN', 'Aléatoire', 'Locale']:
        reponses[(m, g)] = np.zeros((n_specs, pa.max_rings), dtype=np.int64)
        valides_rapide[(m, g)] = np.zeros(n_specs, dtype=bool)
        evaluations[(m, g)] = np.zeros(n_specs)
        temps[(m, g)] = np.zeros(n_specs)
reponses['Optimum'] = np.zeros((n_specs, pa.max_rings), dtype=np.int64)
evaluations['Optimum'] = np.full(n_specs, float(pa.NB_ARCHITECTURES))
temps['Optimum'] = np.zeros(n_specs)
nb_valides_exhaustive = np.zeros(n_specs, dtype=int)
specs_rapides_optimum = np.zeros((n_specs, 3))       # G, SSL, HPBW de l'optimum par la version rapide
trajectoires = {}

pa.recherche_exhaustive(90.0, 30.0, 30.0)            # initialisation du GPU (non chronométrée)
print(f"\n--- Recherches sur {n_specs} spécifications (DQN, aléatoire, locale, exhaustive) ---")
t_debut = time.time()
for i in range(n_specs):
    requete = (theta0s[i], G_mins[i], HPBW_maxs[i])
    t = time.perf_counter()
    P = pa.diagrammes_anneaux(theta0s[i])
    duree_P = time.perf_counter() - t
    for g in GRAINES:
        rng_alea, rng_locale = np.random.RandomState(g + 2000 + i), np.random.RandomState(g + 3000 + i)
        t = time.perf_counter()
        a, v, nb, traj = deroule_DQN(reseaux[g], requete, departs[i], P)
        temps[('DQN', g)][i] = duree_P + time.perf_counter() - t
        reponses[('DQN', g)][i], valides_rapide[('DQN', g)][i], evaluations[('DQN', g)][i] = a, v, nb
        trajectoires[(g, i)] = traj
        # Mêmes budgets que le DQN de cette graine, sur cette spécification
        for m, fonction in [('Aléatoire', lambda: recherche_aleatoire(G_mins[i], HPBW_maxs[i], nb, P, rng_alea)),
                            ('Locale', lambda: recherche_locale(G_mins[i], HPBW_maxs[i], departs[i], nb, P, rng_locale))]:
            t = time.perf_counter()
            a, v, nb_m = fonction()
            temps[(m, g)][i] = duree_P + time.perf_counter() - t
            reponses[(m, g)][i], valides_rapide[(m, g)][i], evaluations[(m, g)][i] = a, v, nb_m
    ex = pa.recherche_exhaustive(*requete)
    reponses['Optimum'][i], temps['Optimum'][i], nb_valides_exhaustive[i] = ex['archi'], ex['duree_s'], ex['nb_valides']
    specs_rapides_optimum[i] = [ex['G'], ex['SSL'], ex['HPBW']]
    if (i + 1) % 10 == 0 or i + 1 == n_specs:
        print(f"  {i + 1}/{n_specs} | {time.time() - t_debut:6.1f} s écoulées")


# =============================================================================
# === 2) Toutes les réponses recalculées avec la fonction de l'enseignant ===
# =============================================================================
def noter(archis):
    """(n, 5) -> (n, 3) : G, SSL, HPBW calculés par la fonction de l'enseignant aux θ0 de test."""
    return np.array([pa.metriques_enseignant(archis[i], theta0s[i])[1:] for i in range(n_specs)])


for nom, a in archis_lues.items():
    reponses[nom] = a
specs = {m: noter(a) for m, a in reponses.items()}
valides = {m: pa.est_valide(s[:, 0], s[:, 2], G_mins, HPBW_maxs) for m, s in specs.items()}

# Écart version rapide / enseignant sur les réponses des recherches
ecart_max = 0.0
for m in valides_rapide:
    P_specs = np.array([pa.metriques_rapides(reponses[m][i], pa.diagrammes_anneaux(theta0s[i]))[1:]
                        for i in range(n_specs)]) if m[1] == GRAINES[0] else None
    if P_specs is not None:
        ecart_max = max(ecart_max, np.abs(P_specs - specs[m]).max())
    assert (valides_rapide[m] == valides[m]).all(), "validité différente entre version rapide et enseignant"
ecart_max = max(ecart_max, np.abs(specs_rapides_optimum - specs['Optimum']).max())
assert valides['Optimum'].all(), "l'optimum recalculé par l'enseignant doit être valide"
SSL_opt = specs['Optimum'][:, 1]
ecart_ref = np.abs(specs['Référence'] - Minput_test[:, :3]).max()


# =============================================================================
# === 3) Tableau comparatif ===
# =============================================================================
METHODES = ['Référence', 'VQ-VAE', 'FFNN', 'DQN', 'Aléatoire', 'Locale', 'Optimum']
NOMS_AFFICHES = {'Référence': 'Référence (test)', 'VQ-VAE': 'VQ-VAE (SSL demandé)*', 'FFNN': 'FFNN (enseignant)*',
                 'DQN': 'DQN', 'Aléatoire': 'Recherche aléatoire', 'Locale': 'Recherche locale',
                 'Optimum': 'Optimum (exhaustive)'}
METRIQUES = ['SSL moyen (dB)', 'SSL médian (dB)', 'Valides (%)', 'Écart moyen à l\'optimum (dB)',
             'Écart médian à l\'optimum (dB)', 'Optimum atteint (%)', 'Évaluations du AF par spec', 'Temps par spec (s)']
NOMS_MAT = ['SSL_moyen_dB', 'SSL_median_dB', 'valides_pct', 'ecart_moyen_optimum_dB', 'ecart_median_optimum_dB',
            'optimum_atteint_pct', 'evaluations_AF_par_spec', 'temps_par_spec_s']


def metriques(cle_m, nom):
    s, v = specs[cle_m], valides[cle_m]
    ecart = s[v, 1] - SSL_opt[v]                          # écart à l'optimum, réponses valides seulement
    ev = evaluations.get(cle_m, np.full(n_specs, evals_lues.get(nom, np.nan)))
    tp = temps.get(cle_m, np.full(n_specs, temps_lus.get(nom, np.nan)))
    return np.array([s[:, 1].mean(), np.median(s[:, 1]), 100 * v.mean(),
                     ecart.mean() if v.any() else np.nan, np.median(ecart) if v.any() else np.nan,
                     100 * np.mean(v & (np.abs(s[:, 1] - SSL_opt) < 1e-9)), np.mean(ev), np.mean(tp)])


tableau, tableau_std = np.zeros((len(METHODES), len(METRIQUES))), np.zeros((len(METHODES), len(METRIQUES)))
for j, m in enumerate(METHODES):
    if m in ['DQN', 'Aléatoire', 'Locale']:
        par_graine = np.array([metriques((m, g), m) for g in GRAINES])
        tableau[j], tableau_std[j] = par_graine.mean(axis=0), par_graine.std(axis=0)
    else:
        tableau[j] = metriques(m, m)

print("\n" + "="*80 + "\nRÉSUMÉ DU TEST\n" + "="*80)
print(f"{n_specs} requêtes du jeu de test : θ0 de test, G ≥ G de test − {pa.TOL_G_DB} dB, "
      f"HPBW ≤ HPBW de test + {pa.TOL_HPBW_DEG}°. Valeurs recalculées par la fonction de l'enseignant.")
print(f"DQN, recherches aléatoire et locale : moyenne sur {len(GRAINES)} graine(s) {GRAINES} (± écart-type entre graines)\n")
entete = f"{'Méthode':<24}" + "".join(f"{c:>12}" for c in ['SSL moy.', 'SSL méd.', 'Valides %', 'Écart moy.',
                                                            'Écart méd.', 'Optimum %', 'Éval. AF', 'Temps (s)'])
print(entete)
print("-" * len(entete))
for j, m in enumerate(METHODES):
    ligne = f"{NOMS_AFFICHES[m]:<24}"
    for c in range(len(METRIQUES)):
        val = tableau[j, c]
        txt = '—' if np.isnan(val) else (f"{val:.4f}" if c == 7 else f"{val:.2f}")
        if m in ['DQN', 'Aléatoire', 'Locale'] and len(GRAINES) > 1 and c in [0, 2, 3]:
            txt += f"±{tableau_std[j, c]:.2f}"
        ligne += f"{txt:>12}"
    print(ligne)
print("Écarts à l'optimum : SSL − SSL optimal, sur les réponses valides seulement (dB).")
print("* Le VQ-VAE et le FFNN visent le SSL DEMANDÉ (et le G demandé), pas le SSL le plus bas : ces lignes comparent "
      "des objectifs, pas la qualité des modèles. Leurs coûts viennent du run committé du premier TP.")

print(f"\nContrôle : écart max specs de référence recalculées / Minput_test = {ecart_ref:.2e}")
print(f"Contrôle : écart max version rapide / fonction de l'enseignant sur les réponses (graine {GRAINES[0]} et optimum) "
      f"= {ecart_max:.2e} ; validités identiques")
N_opt = reponses['Optimum'].sum(axis=1)
print(f"Optimum : {100 * np.mean(N_opt == N_ref):.0f} % des spécifications avec exactement N_ref éléments "
      f"(le minimum imposé par G_min), {100 * np.mean(N_opt > N_ref):.0f} % avec plus ; "
      f"N_opt − N_ref moyen = {np.mean(N_opt - N_ref):.2f}")
print(f"Optimum plus bas que la référence de {np.mean(specs['Référence'][:, 1] - SSL_opt):.2f} dB en moyenne "
      f"(architectures valides par spécification : médiane {int(np.median(nb_valides_exhaustive))} sur 161 050)")
N_dqn = reponses[('DQN', GRAINES[0])].sum(axis=1)
print(f"DQN (graine {GRAINES[0]}) : N − N_ref moyen = {np.mean(N_dqn - N_ref):.2f} ; "
      f"évaluations par spécification : min {int(evaluations[('DQN', GRAINES[0])].min())}, "
      f"max {int(evaluations[('DQN', GRAINES[0])].max())}")


# =============================================================================
# === 4) Sauvegardes ===
# =============================================================================
runs = ['Référence', 'VQ-VAE', 'FFNN'] + [(m, g) for m in ['DQN', 'Aléatoire', 'Locale'] for g in GRAINES] + ['Optimum']
noms_runs = [r if isinstance(r, str) else f"{r[0]}_{r[1]}" for r in runs]
noms_runs = [n.replace('é', 'e').replace('É', 'E').replace('-', '') for n in noms_runs]
COLONNES_RESULTATS = ['n1', 'n2', 'n3', 'n4', 'n5', 'G_dB', 'SSL_dB', 'HPBW_deg', 'valide', 'evaluations_AF', 'temps_s']
resultats = np.zeros((len(runs), n_specs, len(COLONNES_RESULTATS)))
for k_run, r in enumerate(runs):
    nom = r if isinstance(r, str) else r[0]
    resultats[k_run] = np.column_stack([reponses[r], specs[r], valides[r],
                                        evaluations.get(r, np.full(n_specs, evals_lues.get(nom, np.nan))),
                                        temps.get(r, np.full(n_specs, temps_lus.get(nom, np.nan)))])
np.save(os.path.join(script_dir, 'resultats_DQN.npy'), resultats)
i_traj = INDICE_TRAJECTOIRE if INDICE_TRAJECTOIRE < n_specs else 0
scipy.io.savemat(os.path.join(script_dir, 'resultats_DQN.mat'), {
    'resultats': resultats, 'runs': np.array(noms_runs, dtype=object),
    'colonnes': np.array(COLONNES_RESULTATS, dtype=object),
    'requetes': np.column_stack([theta0s, G_mins, HPBW_maxs]),
    'colonnes_requetes': np.array(['theta0_deg', 'G_min_dB', 'HPBW_max_deg'], dtype=object),
    'departs': departs, 'N_ref': N_ref, 'nb_valides_exhaustive': nb_valides_exhaustive,
    'trajectoire_DQN': trajectoires[(GRAINES[0], i_traj)], 'indice_trajectoire': i_traj,
    'colonnes_trajectoire': np.array(['action', 'n1', 'n2', 'n3', 'n4', 'n5', 'G_dB', 'SSL_dB', 'HPBW_deg'],
                                     dtype=object),
    'tolerance_G_dB': pa.TOL_G_DB, 'tolerance_HPBW_deg': pa.TOL_HPBW_DEG, 'graines': np.array(GRAINES)})
np.save(os.path.join(script_dir, 'comparaison_methodes.npy'), tableau)
scipy.io.savemat(os.path.join(script_dir, 'comparaison_methodes.mat'), {
    'tableau': tableau, 'ecart_type_graines': tableau_std,
    'lignes': np.array([n.replace('é', 'e').replace('É', 'E').replace('-', '') for n in METHODES], dtype=object),
    'colonnes': np.array(NOMS_MAT, dtype=object), 'graines': np.array(GRAINES)})
fichiers = ['resultats_DQN.npy', 'resultats_DQN.mat', 'comparaison_methodes.npy', 'comparaison_methodes.mat']


# =============================================================================
# === 5) Figures ===
# =============================================================================
# Couleurs fixes par méthode (celles du premier TP pour Référence / VQ-VAE / FFNN)
COULEURS = {'Référence': '#2a78d6', 'VQ-VAE': '#eb6834', 'FFNN': '#1baf7a', 'DQN': '#6a4ce2',
            'Aléatoire': '#ee7fac', 'Locale': '#f2a600', 'Optimum': '#ec4647'}
cle_fig = {m: (m, GRAINES[0]) if m in ['DQN', 'Aléatoire', 'Locale'] else m for m in METHODES}

# --- SSL et écart à l'optimum par méthode ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ETIQUETTES = {'Référence': 'Référence\n(test)', 'VQ-VAE': 'VQ-VAE*', 'FFNN': 'FFNN*', 'DQN': 'DQN',
              'Aléatoire': 'Recherche\naléatoire', 'Locale': 'Recherche\nlocale', 'Optimum': 'Optimum\n(exhaustive)'}
etiquettes = [ETIQUETTES[m] for m in METHODES]
MEDIANE = {'color': '#191919', 'linewidth': 1.5}     # médiane en encre neutre (pas une couleur de méthode)
bp = axes[0].boxplot([specs[cle_fig[m]][:, 1] for m in METHODES], patch_artist=True, widths=0.55,
                     medianprops=MEDIANE)
for patch, m in zip(bp['boxes'], METHODES):
    patch.set_facecolor(COULEURS[m])
    patch.set_alpha(0.7)
axes[0].set_xticks(range(1, len(METHODES) + 1), etiquettes, fontsize=8)
axes[0].set_ylabel("SSL (dB), plus bas = meilleur")
axes[0].set_title("SSL de la réponse (toutes les spécifications)")
methodes_ecart = [m for m in METHODES if m != 'Optimum']
ecarts = [specs[cle_fig[m]][valides[cle_fig[m]], 1] - SSL_opt[valides[cle_fig[m]]] for m in methodes_ecart]
bp = axes[1].boxplot(ecarts, patch_artist=True, widths=0.55, medianprops=MEDIANE)
for patch, m in zip(bp['boxes'], methodes_ecart):
    patch.set_facecolor(COULEURS[m])
    patch.set_alpha(0.7)
axes[1].set_xticks(range(1, len(methodes_ecart) + 1),
                   [f"{e}\n{100 * valides[cle_fig[m]].mean():.0f} % valides"
                    for e, m in zip(etiquettes[:-1], methodes_ecart)], fontsize=8)
axes[1].set_ylabel("SSL − SSL optimal (dB)")
axes[1].set_title("Écart à l'optimum (réponses valides seulement)")
for ax in axes:
    ax.grid(True, axis='y', alpha=0.3)
fig.suptitle(f"{n_specs} spécifications de test - DQN et recherches : graine {GRAINES[0]} - "
             f"* vise le SSL demandé, pas le plus bas")
fig.tight_layout()
fig.savefig(os.path.join(script_dir, 'boxplot_SSL_methodes.png'), dpi=150)
fichiers.append('boxplot_SSL_methodes.png')

# --- Diagrammes de rayonnement : Référence / VQ-VAE / DQN / optimum ---
STYLES = {'Référence': '-', 'VQ-VAE': '--', 'DQN': '-.', 'Optimum': ':'}
indices_figures = [i for i in INDICES_FIGURES if i < n_specs] or [0]
for i in indices_figures:
    fig = plt.figure(figsize=(14, 6.5))
    ax_pol = fig.add_subplot(1, 2, 1, polar=True)
    ax_cart = fig.add_subplot(1, 2, 2)
    for m in ['Référence', 'VQ-VAE', 'DQN', 'Optimum']:
        archi = reponses[cle_fig[m]][i]
        AF_dB = pa.metriques_enseignant(archi, theta0s[i])[0]
        s, v = specs[cle_fig[m]][i], valides[cle_fig[m]][i]
        label = f"{m} {archi} : SSL {s[1]:.2f} dB" + ("" if v else " (non valide)")
        ax_pol.plot(pa.theta, AF_dB, color=COULEURS[m], linestyle=STYLES[m], linewidth=2, label=label)
        ax_cart.plot(pa.theta_deg, AF_dB, color=COULEURS[m], linestyle=STYLES[m], linewidth=2, label=label)
    ax_pol.set_rlim([-40, 0])
    ax_pol.set_title("Diagramme de rayonnement (polaire)")
    ax_cart.set_xlabel("θ dans la coupe φ = 0 (°)")
    ax_cart.set_ylabel("Gain normalisé (dB)")
    ax_cart.set_xlim([0, 360])
    ax_cart.set_ylim([-40, 0])
    ax_cart.grid(True, alpha=0.3)
    ax_cart.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
    ax_cart.set_title(f"Requête : G ≥ {G_mins[i]:.2f} dB ({N_ref[i]} éléments), HPBW ≤ {HPBW_maxs[i]:.1f}°, "
                      f"SSL le plus bas possible", fontsize=10)
    fig.suptitle(f"Exemple de test {i} - θ0 = {theta0s[i]:.1f}° (DQN : graine {GRAINES[0]})")
    fig.tight_layout()
    nom = f'diagrammes_DQN_{i:02d}.png'
    fig.savefig(os.path.join(script_dir, nom), dpi=150)
    fichiers.append(nom)

# --- Trajectoire du DQN pas à pas sur une spécification ---
traj = trajectoires[(GRAINES[0], i_traj)]
pas = np.arange(len(traj))
valide_traj = pa.est_valide(traj[:, 6], traj[:, 8], G_mins[i_traj], HPBW_maxs[i_traj])
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
grandeurs = [(7, "SSL (dB)", None, None), (6, "G (dB)", G_mins[i_traj] - pa.TOL_G_DB, "G_min − tolérance"),
             (8, "HPBW (°)", HPBW_maxs[i_traj] + pa.TOL_HPBW_DEG, "HPBW_max + tolérance")]
for k_ax, (ax, (col, ylabel, seuil, nom_seuil)) in enumerate(zip(axes, grandeurs)):
    ax.plot(pas, traj[:, col], color=COULEURS['DQN'], linewidth=2)
    ax.scatter(pas[valide_traj], traj[valide_traj, col], s=60, color=COULEURS['DQN'], zorder=3,
               edgecolor='white', linewidth=1.5, label='architecture valide' if k_ax == 0 else None)
    ax.scatter(pas[~valide_traj], traj[~valide_traj, col], s=60, facecolor='white', edgecolor=COULEURS['DQN'],
               linewidth=1.5, zorder=3, label='architecture non valide' if k_ax == 0 else None)
    if seuil is not None:
        ax.axhline(seuil, color='0.4', linewidth=1, linestyle='--', label=nom_seuil)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
axes[0].axhline(SSL_opt[i_traj], color=COULEURS['Optimum'], linewidth=1.5, linestyle=':',
                label=f"optimum {reponses['Optimum'][i_traj]} : {SSL_opt[i_traj]:.2f} dB")
axes[0].axhline(specs['Référence'][i_traj, 1], color=COULEURS['Référence'], linewidth=1.5, linestyle='-',
                label=f"référence {reponses['Référence'][i_traj]} : {specs['Référence'][i_traj, 1]:.2f} dB")
for ax in axes:
    ax.legend(frameon=False, fontsize=8, loc='best')
# Axe des pas : action jouée et architecture obtenue
etiquettes_pas = [('départ' if p == 0 else ea.NOMS_ACTIONS[int(traj[p, 0])].replace(' anneau ', ' a'))
                  + '  ' + ' '.join(str(int(v)) for v in traj[p, 1:6]) for p in pas]
axes[-1].set_xticks(pas, etiquettes_pas, rotation=90, fontsize=8)
axes[-1].set_xlabel("Pas : action jouée et architecture obtenue")
archi_rep = reponses[('DQN', GRAINES[0])][i_traj]
fig.suptitle(f"Déroulé du DQN (graine {GRAINES[0]}), exemple de test {i_traj}, θ0 = {theta0s[i_traj]:.1f}° - "
             f"réponse {archi_rep} : SSL {specs[('DQN', GRAINES[0])][i_traj, 1]:.2f} dB, "
             f"{int(evaluations[('DQN', GRAINES[0])][i_traj])} évaluations")
fig.tight_layout()
fig.savefig(os.path.join(script_dir, 'trajectoire_DQN.png'), dpi=150)
fichiers.append('trajectoire_DQN.png')

if AFFICHER_FIGURES:
    plt.show()
plt.close('all')

print(f"\nFichiers enregistrés dans {script_dir} :")
for nom in fichiers:
    print(f"  {nom}")
