# Assistant pour les migrants — LLM + RAG

Projet d'équipe inspiré de Castellanos-Nieves & García-Forte (2025),
*Human-Centered AI for Migrant Integration Through LLM and RAG Optimization*.

| Partie | Responsable | Contenu |
|---|---|---|
| **B** | Corpus et RAG | `rag.py`, `corpus/`, `index/`, `experiments_rag.py`, `experiments_llm.py` — **terminée** |
| **I** | LLM et évaluation | `llm.py`, `evaluation.py`, `bias_detection.py`, `energy_metrics.py` |
| **D** | Décision multicritère (RIM, TOPSIS, VIKOR) | `partie_D/` — analyse multicritère de modèles |

## Installation (Windows, à faire une fois)

**Prérequis :** Python **3.12** (pas 3.13 ni 3.14), Git, ~10 Go libres.

```powershell
git clone https://github.com/89DJUD/projet_lamrani.git
cd projet_lamrani
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- Si PowerShell refuse `Activate.ps1` :
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, puis réessayer.
- Le premier lancement télécharge les modèles d'embeddings (~500 Mo) : il faut Internet.

### Llama 3 (seulement pour générer de nouvelles réponses)

1. Installer Ollama : <https://ollama.com/download>
2. Dans un **nouveau** terminal : `ollama pull llama3:8b` (4,7 Go).
3. Ollama doit être ouvert (icône près de l'horloge) avant de lancer l'application.

Sans Ollama, on peut choisir `flan-t5-small` dans l'application (petit modèle, réponses faibles).

## Lancer l'application principale

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Onglets : **Poser une question** (RAG + sources), **Documents** (corpus et index),
**Évaluation** (le jeu de test `test_set.csv` → BLEU, ROUGE, similarité, export CSV).

## Lancer l'application de la partie D (Analyse multicritère)

```powershell
.\.venv\Scripts\Activate.ps1
cd partie_D
streamlit run app.py
```

L'application de la partie D permet d'analyser les performances des configurations
RAG+LLM avec les méthodes multicritères :
- **RIM** (Reference Ideal Method)
- **TOPSIS** (Technique for Order Preference by Similarity to Ideal Solution)
- **VIKOR** (VIseKriterijumsko KOmpromisno Rangiranje)

## Ce qui est déjà fourni (rien à recalculer)

| Fichier | Contenu |
|---|---|
| `corpus/` | 8 PDF : 7 directives de l'UE (en français) + guide du HCR (en anglais) |
| `index/chunk_{50,100,300,500,1000}/` | 5 index FAISS déjà construits (passages de 50 à 1000 mots) |
| `test_set.csv` | 20 questions + réponse de référence + page source (dont 1 hors sujet) |
| `results/retrieval_results.csv` | Qualité de la recherche : 5 tailles × 5 top-K (Hit@K, MRR) |
| `results/top1_scores.csv` | Justification du seuil de similarité (0,40) |
| `results/llm_raw.csv` | Chaque réponse de Llama 3, par question et par configuration |
| `results/resultats_pour_D.csv` | **Une ligne par configuration (C1–C6), scores médians** |
| `partie_D/results/` | Résultats de l'analyse multicritère (classements, corrélations, sensibilité) |

### Colonnes de `resultats_pour_D.csv`

| Colonne | Sens | Sens du critère |
|---|---|---|
| `chunk_size`, `top_k`, `temperature` | Paramètres de la configuration | — |
| `bleu`, `rouge1`, `rouge2`, `rougeL`, `semantic_similarity` | Qualité de la réponse vs référence (médiane, 0 à 1) | à maximiser |
| `hit_page` | Part des questions où la bonne page est retrouvée | à maximiser |
| `mrr` | Rang moyen inverse de la bonne page | à maximiser |
| `seconds_median` | Temps de réponse médian (CPU, sans carte graphique) | à minimiser |

Remarque : le temps de C4 (79 s) est probablement gonflé par une perturbation du PC pendant la mesure.

## Choix techniques de la partie B

- Extraction du texte : `pypdf`, nettoyage des césures et espaces.
- Découpage : par page, en passages de N **mots** avec 20 % de recouvrement ; passages < 25 mots écartés.
- Embeddings : `paraphrase-multilingual-MiniLM-L12-v2` (questions en français, un document en anglais).
- Index : FAISS `IndexFlatIP` sur vecteurs normalisés (= similarité cosinus).
- Seuil de similarité 0,40 : en dessous, la question est jugée hors corpus et le LLM n'est pas appelé.
- Llama 3 8B via Ollama, graine 42, 150 tokens max, consigne en français.

## Méthodes de la partie D

L'analyse multicritère compare les configurations RAG+LLM selon plusieurs critères :
- **Pondération** : attribution de poids aux critères (importance)
- **Normalisation** : mise à l'échelle des critères sur [0,1]
- **Agrégation** : combinaison des scores selon la méthode choisie
- **Analyse de sensibilité** : test de robustesse aux variations de poids
- **Comparaison** : corrélations entre les méthodes (Kendall, Spearman)

## Relancer les expériences (facultatif)

### Partie B (RAG + LLM)
```powershell
python experiments_rag.py   # ~1 min, sans LLM
python experiments_llm.py   # plusieurs heures sur CPU, reprend où il s'est arrêté
```

### Partie D (Analyse multicritère)
```powershell
cd partie_D
python run_analysis.py     # lance l'analyse complète
```

Ajouter des PDF : les mettre dans `corpus/`, puis onglet **Documents** → *Construire les 5 index* (~10 min).

## Problèmes fréquents

| Message | Solution |
|---|---|
| `ollama n'est pas reconnu` | Fermer et rouvrir le terminal après l'installation d'Ollama |
| « Ollama ne tourne pas » dans l'application | Ouvrir l'application Ollama, puis recharger la page |
| `Une stratégie de contrôle d'application a bloqué ce fichier` | Windows Smart App Control : Sécurité Windows → Contrôle des applications → désactiver |
| `No module named ...` | L'environnement n'est pas activé : `.\.venv\Scripts\Activate.ps1` |
| Calcul bloqué pendant des heures | PC en veille : le laisser branché, couvercle ouvert |
