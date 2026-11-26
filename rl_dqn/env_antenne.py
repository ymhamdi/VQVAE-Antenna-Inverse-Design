# -*- coding: utf-8 -*-
"""
Module R0 bis — Environnement d'apprentissage par renforcement (second TP : DQN), interface minimale à la Gym

Objectif : pour une requête (θ0, G_min, HPBW_max), trouver l'architecture (5 entiers de 0 à 10) qui minimise le SSL
sous les contraintes G ≥ G_min et HPBW ≤ HPBW_max (tolérances de physique_antenne.py).
G = 20·log10(nombre total d'éléments) : G_min est donc un nombre minimal d'antennes (le « nombre ») ;
la répartition sur les anneaux fixe le SSL et le HPBW (l'« emplacement »).

Épisode :
  - requête : θ0 ~ U[0°, 180°] ; G_min et HPBW_max tirés uniformément dans les plages atteintes, à ce θ0, par
    NB_ARCHIS_PLAGE architectures aléatoires générées à la volée (même loi que dataset.py). Aucun fichier lu.
    Le couple (G_min, HPBW_max) peut être impossible à satisfaire : l'agent reçoit alors des pénalités.
  - départ : une architecture aléatoire (même loi), au moins un élément ;
  - 11 actions : +1 élément sur l'anneau r (0 à 4), −1 élément sur l'anneau r (5 à 9), arrêt (10) ;
    un coup illégal (n_r < 0, n_r > 10, réseau vide) ne change rien et coûte PENALITE_ILLEGAL ;
  - fin : action d'arrêt ou T pas ;
  - réponse : la meilleure architecture valide vue pendant l'épisode (SSL minimal) ; s'il n'y en a aucune,
    l'architecture de meilleur score J, marquée non valide.

Score et récompense :
  J(n) = −SSL(n) − α·max(0, G_min − tol_G − G(n)) − β·max(0, HPBW(n) − HPBW_max − tol_HPBW) − P·1[n non valide]
  Le terme fixe P = 5 dB a été ajouté après des essais : sans lui, l'agent acceptait de petites violations des
  contraintes en échange de quelques dB de SSL (59 % de réponses valides sur le test) ; avec P = 20 dB, il
  s'arrêtait dès la première architecture valide (pas mieux que la recherche aléatoire).
  r    = (J(n') − J(n)) / ECHELLE_J − COUT_PAS        (coup légal)
  r    = −PENALITE_ILLEGAL                            (coup illégal)    ;    r = 0 pour l'arrêt

État (14 valeurs) : n/10 (5), sin θ0 (θ0 n'agit que par sin θ0, voir physique_antenne.py), G_min/34,
  HPBW_max/180, G/34, SSL/40, HPBW/180, déficit de gain/10, excès de HPBW/90, pas restants/T.

Ce fichier ne s'exécute pas seul : il est importé par DQN_training.py (R1) et test_model_DQN.py (R2).
"""
import numpy as np
import physique_antenne as pa


# === Paramètres de l'environnement ===
T_MAX = 30                 # nombre maximal de pas par épisode
NB_ARCHIS_PLAGE = 64       # architectures aléatoires qui définissent les plages de G_min et HPBW_max à θ0
ALPHA = 10.0               # pénalité par dB de gain manquant (un élément manquant ≈ 8,7·α/N ≥ 1,7 dB, N ≤ 50)
BETA = 1.0                 # pénalité par degré de HPBW en trop
ECHELLE_J = 10.0           # J en dB → récompense de l'ordre de l'unité
COUT_PAS = 0.005           # coût d'un pas (0,05 dB de score) : s'arrêter a un sens
PENALITE_ILLEGAL = 0.1     # coup illégal (1 dB de score)
PENALITE_INVALIDE = 5.0    # pénalité fixe (dB de score) pour toute architecture qui viole une contrainte
NB_ACTIONS = 2 * pa.max_rings + 1
ACTION_ARRET = NB_ACTIONS - 1
DIM_ETAT = 14
NOMS_ACTIONS = [f"+1 anneau {r + 1}" for r in range(pa.max_rings)] + \
               [f"-1 anneau {r + 1}" for r in range(pa.max_rings)] + ["arrêt"]


def score_J(G, SSL, HPBW, G_min, HPBW_max):
    """Score J (dB) : −SSL moins les pénalités de contrainte (nulles si l'architecture est valide)."""
    deficit_G = max(0.0, G_min - pa.TOL_G_DB - G)
    exces_HPBW = max(0.0, HPBW - HPBW_max - pa.TOL_HPBW_DEG)
    invalide = deficit_G > 0 or exces_HPBW > 0
    return -SSL - ALPHA * deficit_G - BETA * exces_HPBW - PENALITE_INVALIDE * invalide


def appliquer_action(archi, action):
    """Architecture obtenue après l'action, ou None si le coup est illégal (ou si c'est l'arrêt)."""
    if action == ACTION_ARRET:
        return None
    ring, delta = (action, +1) if action < pa.max_rings else (action - pa.max_rings, -1)
    nouvelle = archi.copy()
    nouvelle[ring] += delta
    if nouvelle[ring] < 0 or nouvelle[ring] > pa.max_elements or nouvelle.sum() == 0:
        return None
    return nouvelle


class EnvAntenne:
    """Environnement minimal à la Gym : reset() -> état ; step(action) -> état, récompense, terminé, info."""

    def __init__(self, rng, T=T_MAX, device=pa.DEVICE_AF):
        self.rng = rng            # np.random.RandomState : requêtes et départs
        self.T = T
        self.device = device

    # --- requête et départ ---
    def tirer_requete(self):
        """θ0 ~ U[0°, 180°], puis G_min et HPBW_max uniformes dans les plages atteintes à ce θ0
        par NB_ARCHIS_PLAGE architectures aléatoires (calcul par lot)."""
        theta0deg = self.rng.uniform(0, 180)
        P = pa.diagrammes_anneaux(theta0deg)
        pool = np.array([pa.tirer_architecture(self.rng) for _ in range(NB_ARCHIS_PLAGE)])
        G, _, HPBW, _ = pa.metriques_lot(pool, pa.vers_torch(P, self.device),
                                         filtre_SSL=lambda G, H: np.zeros(len(G), dtype=bool))
        G_min = self.rng.uniform(G.min(), G.max())
        HPBW_max = self.rng.uniform(HPBW.min(), HPBW.max())
        return (theta0deg, G_min, HPBW_max), P

    def reset(self, requete=None, depart=None, P=None):
        """requete = (θ0 en °, G_min en dB, HPBW_max en °) ; sinon tirée. depart : architecture initiale ;
        sinon tirée. P : diagrammes d'anneau déjà calculés pour ce θ0 (facultatif)."""
        if requete is None:
            requete, P = self.tirer_requete()
        self.theta0deg, self.G_min, self.HPBW_max = requete
        self.P = pa.diagrammes_anneaux(self.theta0deg) if P is None else P
        self.archi = (pa.tirer_architecture(self.rng) if depart is None
                      else np.asarray(depart, dtype=np.int64)).copy()
        self.cache = {}                      # architecture -> (G, SSL, HPBW) : évaluations uniques du facteur de réseau
        self.meilleure_valide = None         # (SSL, architecture)
        self.meilleure_J = None              # (J, architecture)
        self.pas = 0
        self.J = self._evaluer(self.archi)
        return self._etat()

    def _evaluer(self, archi):
        """Évalue (avec cache) G, SSL, HPBW et J ; met à jour les meilleures architectures vues."""
        cle = tuple(int(v) for v in archi)
        if cle not in self.cache:
            _, G, SSL, HPBW = pa.metriques_rapides(archi, self.P)
            self.cache[cle] = (G, SSL, HPBW)
        G, SSL, HPBW = self.cache[cle]
        J = score_J(G, SSL, HPBW, self.G_min, self.HPBW_max)
        self.G, self.SSL, self.HPBW = G, SSL, HPBW
        if pa.est_valide(G, HPBW, self.G_min, self.HPBW_max):
            if self.meilleure_valide is None or SSL < self.meilleure_valide[0]:
                self.meilleure_valide = (SSL, archi.copy())
        if self.meilleure_J is None or J > self.meilleure_J[0]:
            self.meilleure_J = (J, archi.copy())
        return J

    def _etat(self):
        deficit_G = max(0.0, self.G_min - pa.TOL_G_DB - self.G)
        exces_HPBW = max(0.0, self.HPBW - self.HPBW_max - pa.TOL_HPBW_DEG)
        return np.array([*(self.archi / pa.max_elements), np.sin(np.deg2rad(self.theta0deg)),
                         self.G_min / 34, self.HPBW_max / 180, self.G / 34, self.SSL / 40, self.HPBW / 180,
                         deficit_G / 10, exces_HPBW / 90, (self.T - self.pas) / self.T], dtype=np.float32)

    def step(self, action):
        self.pas += 1
        if action == ACTION_ARRET:
            recompense, termine = 0.0, True
        else:
            nouvelle = appliquer_action(self.archi, action)
            if nouvelle is None:                                   # coup illégal : rien ne change
                recompense = -PENALITE_ILLEGAL
            else:
                J_avant = self.J
                self.archi = nouvelle
                self.J = self._evaluer(nouvelle)
                recompense = (self.J - J_avant) / ECHELLE_J - COUT_PAS
            termine = self.pas >= self.T
        return self._etat(), recompense, termine, {'nb_evaluations': len(self.cache)}

    def reponse(self):
        """(architecture, valide) : meilleure valide vue, sinon meilleur score J."""
        if self.meilleure_valide is not None:
            return self.meilleure_valide[1].copy(), True
        return self.meilleure_J[1].copy(), False
