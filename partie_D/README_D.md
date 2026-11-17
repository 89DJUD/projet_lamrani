# Partie D — Optimisation multicritère RIM, TOPSIS et VIKOR

Module autonome d'aide à la décision pour le projet
*Human-Centered AI for Migrant Integration Through LLM and RAG
Optimization*. Il classe les configurations expérimentales
(`results/resultats_pour_D.csv`) selon trois méthodes multicritères et
produit une analyse comparative, une analyse de robustesse et des
visualisations.

## Données utilisées

`results/resultats_pour_D.csv` — une ligne par configuration (C1–C6),
scores médians sur 20 questions.

| Critère | Sens |
|---|---|
| `bleu`, `rouge1`, `rouge2`, `rougeL`, `semantic_similarity` | à maximiser |
| `hit_page`, `mrr` | à maximiser |
| `seconds_median` | **à minimiser** |

`config`, `model`, `chunk_size`, `top_k`, `temperature`, `n_questions`
sont des identifiants/métadonnées, **pas** des critères.

**Absents du CSV** : métriques de biais et d'impact environnemental
(`bias_detection.py` et `energy_metrics.py` sont des squelettes de la
partie I). Elles ne sont donc pas optimisées — limite documentée.

## Installation

Prérequis : Python **3.12**, Git. À faire une fois, depuis la racine
du projet :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pandas numpy scipy plotly streamlit pytest
```

(ou `pip install -r requirements.txt` du projet, qui couvre déjà
l'essentiel ; `scipy`, `plotly` et `pytest` sont ajoutés pour la D.)

## Tests

```powershell
python -m pytest partie_D/tests -v
```

Les tests utilisent des exemples numériques calculés à la main
(TOPSIS 2×2, VIKOR 3×2, RIM 3×2) plus des tests de robustesse
(valeurs manquantes, critères constants, ex æquo, conditions de
compromis VIKOR, somme des poids).

## Exécution de l'analyse complète (CLI)

```powershell
python -m partie_D.run_analysis
```

Exporte dans `partie_D/results/` : classements CSV des trois méthodes,
tableau des rangs, corrélations Spearman/Kendall, scénarios de poids,
Monte-Carlo (500 itérations, graine 42), sensibilité du paramètre v,
matrice normalisée.

## Application Streamlit

```powershell
streamlit run partie_D/app.py
```

Six pages : Vue d'ensemble, Matrice de décision, Méthodes
multicritères (formules + tableaux + graphiques), Comparaison
(corrélations, Top-3, désaccords), Robustesse (scénarios, Monte-Carlo,
v de VIKOR), Conclusion (recommandation + limites + téléchargements
CSV/rapport).

Paramètres disponibles dans la barre latérale : scénario de
pondération (uniforme, qualité, rapidité, équité/environnement,
personnalisée), paramètre v de VIKOR.

## Architecture

```
partie_D/
├── app.py                  # interface Streamlit (6 pages)
├── pipeline.py             # orchestration commune app / CLI
├── data_loader.py          # chargement CSV (séparateur + encodage auto)
├── preprocessing.py        # matrice de décision, directions, NA, rapport
├── weighting.py            # 5 scénarios de pondération + Dirichlet
├── methods/
│   ├── topsis.py           # Hwang & Yoon (1981)
│   ├── vikor.py            # Opricovic & Tzeng (2004)
│   └── rim.py              # Sałabun et al. (2019), Reference Ideal Method
├── analysis/
│   ├── comparison.py       # rangs, Spearman, Kendall, Top-k, désaccords
│   └── sensitivity.py      # scénarios, Monte-Carlo, sensibilité v
├── visualization/plots.py  # 9 familles de graphiques Plotly
├── tests/                  # pytest
├── results/                # exports CSV produits par run_analysis
└── README_D.md             # ce fichier
```

## Méthodes — points clés

- **TOPSIS** : normalisation vectorielle, distances aux solutions
  idéale positive/négative, `C_i = D_i−/(D_i+ + D_i−)` ∈ [0,1],
  rang décroissant.
- **VIKOR** : utilité collective `S_i`, regret `R_i`,
  `Q_i = v·S_i/S* + (1−v)·R_i/R*`, rang croissant ; conditions de
  compromis (avantage `Q1−Qk ≥ 1/(n−1)`, stabilité `Rk−R1 ≥ 1/(n−1)`)
  respectées et affichées.
- **RIM** : distance à un intervalle idéal de référence par critère,
  normalisation par l'étendue, distances aux solutions idéale et
  anti-idéale, score de proximité. **Mode exploratoire** : faute de
  seuils métier, l'intervalle idéal = meilleure valeur observée
  (limite explicitement signalée ; des intervalles personnalisés
  peuvent être fournis via `ideal_intervals`).

## Hypothèses et limites

- Toutes les métriques de score sont maximisées sauf
  `seconds_median` (minimisée).
- Les scénarios de pondération non uniformes sont des **hypothèses
  d'analyse**, pas des préférences validées par les utilisateurs.
- Le scénario « équité/environnement » bascule sur la fiabilité
  (`hit_page`, `mrr`) car aucune métrique dédiée n'existe dans le CSV.
- `seconds_median` de C4 (79 s) est probablement gonflé par une
  perturbation matérielle (cf. README projet).
- Les scores bruts des trois méthodes ne sont pas comparables entre
  eux : la comparaison porte sur les **rangs**.
- La robustesse est évaluée vis-à-vis des poids uniquement.
