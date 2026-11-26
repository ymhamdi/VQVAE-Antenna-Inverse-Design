# -*- coding: utf-8 -*-
"""
Images du README principal (pas un script des TP) : python assets/readme/make_readme_images.py

  hero.jpg                  composite de deux figures déjà committées, recadrées sur leur vue cartésienne :
                            diagramme_exemple_77.png (script 4, VQ-VAE) et rl_dqn/diagrammes_DQN_00.png (script R2, DQN)
  before-reference-spec00.jpg   diagramme de l'architecture de référence de l'exemple de test 0
  after-dqn-spec00.jpg          diagramme de la réponse du DQN (graine 46) sur le même exemple

Les deux diagrammes « avant / après » sont recalculés avec la fonction de l'enseignant (rl_dqn/physique_antenne.py),
à partir des architectures committées (Moutput_test.npy, rl_dqn/resultats_DQN.mat) : aucune valeur n'est recopiée.
Le script n'écrit que dans assets/readme/.
"""
import os
import sys
import numpy as np
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.normpath(os.path.join(ICI, '..', '..'))
sys.path.insert(0, os.path.join(RACINE, 'rl_dqn'))
import physique_antenne as pa      # fonction de l'enseignant, copiée telle quelle

JPG = {'quality': 88, 'optimize': True}


# === 1) hero.jpg : deux vues cartésiennes côte à côte ===
def recadrer(chemin, x0, y0):
    """Vue cartésienne (moitié droite) d'une figure committée, sans son titre général."""
    image = mpimg.imread(os.path.join(RACINE, chemin))[..., :3]
    return image[y0:, x0:]


gauche = recadrer('diagramme_exemple_77.png', 990, 80)            # script 4 : demandé / VQ-VAE / FFNN
droite = recadrer(os.path.join('rl_dqn', 'diagrammes_DQN_00.png'), 1000, 85)   # R2 : référence / VQ-VAE / DQN / optimum
LARGEUR, BANDEAU, ECART = 1600, 56, 24
h = (LARGEUR - ECART) / (gauche.shape[1] / gauche.shape[0] + droite.shape[1] / droite.shape[0])
w1, w2 = gauche.shape[1] * h / gauche.shape[0], droite.shape[1] * h / droite.shape[0]
HAUTEUR = h + BANDEAU
fig = plt.figure(figsize=(LARGEUR / 100, HAUTEUR / 100), dpi=100, facecolor='white')
for x, w, image in [(0, w1, gauche), (w1 + ECART, w2, droite)]:
    ax = fig.add_axes([x / LARGEUR, 0, w / LARGEUR, h / HAUTEUR])
    ax.imshow(image, interpolation='antialiased')
    ax.axis('off')
fig.text(w1 / 2 / LARGEUR, 1 - 30 / HAUTEUR, "Match the beam you ask for  ·  VQ-VAE, test spec 77",
         ha='center', va='center', fontsize=17, fontweight='bold', color='#0b0b0b')
fig.text((w1 + ECART + w2 / 2) / LARGEUR, 1 - 30 / HAUTEUR, "Push the side lobes down  ·  DQN, test spec 0",
         ha='center', va='center', fontsize=17, fontweight='bold', color='#0b0b0b')
fig.savefig(os.path.join(ICI, 'hero.jpg'), dpi=100, facecolor='white', pil_kwargs=JPG)
plt.close(fig)


# === 2) avant / après sur l'exemple de test 0 (fonction de l'enseignant) ===
Minput_test = np.load(os.path.join(RACINE, 'Minput_test.npy'))
Moutput_test = np.load(os.path.join(RACINE, 'Moutput_test.npy'))
res = scipy.io.loadmat(os.path.join(RACINE, 'rl_dqn', 'resultats_DQN.mat'))
runs = [str(r[0]) for r in res['runs'].ravel()]
i = 0
theta0deg, G_min, HPBW_max = Minput_test[3, i], Minput_test[0, i], Minput_test[2, i]
cas = [('before-reference-spec00.jpg', "Before · reference array", Moutput_test[:, i].astype(int), '#2a78d6'),
       ('after-dqn-spec00.jpg', "After · DQN answer (seed 46)",
        res['resultats'][runs.index('DQN_46')][i, :5].astype(int), '#6a4ce2')]
for nom, titre, archi, couleur in cas:
    AF_dB, G, SSL, HPBW = pa.metriques_enseignant(archi, theta0deg)
    valide = pa.est_valide(G, HPBW, G_min, HPBW_max)
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=100)
    ax.plot(pa.theta_deg, AF_dB, color=couleur, linewidth=2)
    ax.axhline(SSL, color='#52514e', linewidth=1.2, linestyle='--')
    ax.text(358, SSL + 0.8, f"highest side lobe: {SSL:.2f} dB", ha='right', va='bottom', fontsize=11, color='#52514e')
    ax.set_xlim(0, 360)
    ax.set_ylim(-40, 0.5)
    ax.set_xlabel("θ in the φ = 0 cut (°)")
    ax.set_ylabel("Normalised gain (dB)")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{titre} {archi}\nSSL {SSL:.2f} dB · G {G:.2f} dB · HPBW {HPBW:.1f}° · "
                 f"{'valid' if valide else 'not valid'}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, nom), dpi=100, pil_kwargs=JPG)
    plt.close(fig)
    print(f"{nom}: {archi} SSL {SSL:.2f} dB, G {G:.2f} dB, HPBW {HPBW:.2f}°, {'valide' if valide else 'non valide'}")

for nom in ['hero.jpg', 'before-reference-spec00.jpg', 'after-dqn-spec00.jpg']:
    taille = os.path.getsize(os.path.join(ICI, nom))
    largeur = mpimg.imread(os.path.join(ICI, nom)).shape[1]
    print(f"{nom}: {largeur} px de large, {taille / 1024:.0f} Ko")
