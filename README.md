# Assistant pour l’intégration des migrants — LLM et RAG

Projet de classe inspiré de Castellanos-Nieves et García-Forte,
[Human-Centered AI for Migrant Integration Through LLM and RAG Optimization](https://doi.org/10.3390/app15010325).

L’application recherche des passages dans un corpus de PDF, génère une réponse
avec Llama 3 puis évalue sa qualité, les signaux de biais et son impact environnemental.
Les scripts et données déjà présents restent disponibles ; les contributions
suivantes peuvent s’appuyer sur les mêmes index et résultats.

## Installation

Prérequis : Python 3.12 et [Ollama](https://ollama.com/download).
Depuis la racine du dépôt, dans PowerShell :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-application.txt
ollama pull llama3:8b-instruct-q2_K
```

`requirements-application.txt` contient les dépendances de la version intégrée.
`requirements.txt` conserve la configuration d’origine. `requirements-validated.txt`
consigne l’environnement utilisé pour les vérifications locales.

La variante Q2_K est plus compressée (environ 3,2 Go de poids) et utilise un
contexte de 2048 tokens. Elle réduit les besoins mémoire, avec un compromis
possible sur la qualité. Son démarrage dépend de la RAM disponible. La variante
standard `llama3:8b` reste proposée dans l’interface.

## Lancer l’application commune

```powershell
$env:MIGRANT_LLM_MODEL = "llama3:8b-instruct-q2_K"
$env:MIGRANT_OLLAMA_KEEP_ALIVE = "0"
python -m streamlit run application.py
```

Les onglets permettent de consulter les documents, poser une question avec ses
sources et évaluer le jeu de test. Les réglages sociaux et environnementaux se
trouvent dans la barre latérale. Les modèles d’embeddings et le classifieur NLI
optionnel sont téléchargés au premier usage.

## Organisation

| Fonction | Emplacement |
|---|---|
| Corpus, découpage, embeddings, index et recherche | `rag.py`, `corpus/`, `index/` |
| Interface commune | `application.py`, `migrant_ai/interface.py` |
| Génération Llama 3 | `migrant_ai/generation.py` |
| BLEU, ROUGE et similarité | `migrant_ai/quality.py` |
| Indicateurs de biais | `migrant_ai/fairness.py` |
| Énergie et CO2 | `migrant_ai/footprint.py` |
| Campagnes et export | `migrant_ai/experiments.py` |
| Résultats et données de référence | `results/`, `test_set.csv` |
| Vérification et transmission des résultats | `tests/`, `docs/` |

Les fichiers Python existants conservent leur contenu et leurs interfaces.
L’historique Git conserve les contributions de l’équipe.

## Expériences

Réévaluer les réponses déjà fournies, sans les régénérer :

```powershell
python -m migrant_ai.experiments --from-csv results/llm_raw.csv --reevaluate --output results/evaluations/reevaluation_multilingue
```

Tester deux nouvelles questions avant une campagne complète :

```powershell
python -m migrant_ai.experiments --model llama3:8b-instruct-q2_K --chunk-sizes 50 --top-k 2 --temperatures 0.1 --limit 2 --energy --country MAR --output results/evaluations/essai_court
```

Le paramètre `--bias-method model` active le classifieur NLI multilingue.
Le mode lexical par défaut fonctionne sans ce modèle supplémentaire.
Une campagne limitée ne figure pas dans le tableau des configurations complètes.

## Transmission pour l’optimisation

Les CSV historiques, dont **`results/resultats_pour_D.csv`**, sont conservés.
Le [guide des données pour l’optimisation](docs/optimisation.md) décrit les colonnes,
les critères, les configurations incomplètes et les valeurs manquantes.
Les méthodes RIM, TOPSIS et VIKOR seront ajoutées lors de cette prochaine étape.

## Vérification

```powershell
python -m unittest discover -s tests -v
python -m migrant_ai.diagnostic
python -m migrant_ai.validate_generation --wait
```

Le dernier contrôle attend les poids puis essaie deux questions réelles. Son état
est enregistré dans `reports/generation_validation.json`. Consulter le
[compte rendu de validation](docs/validation.md) pour distinguer les vérifications
réelles, les simulations et les étapes encore en attente.

Les réponses peuvent contenir des erreurs et ne constituent pas un avis juridique.
Les scores de biais demandent une validation humaine. CodeCarbon estime la
consommation de la machine locale, autres processus inclus ; il ne mesure pas
l’inférence d’un serveur distant. Ne pas remplacer une mesure absente par zéro.

L’[audit de la réponse incorrecte](docs/diagnostic_reponse.md) détaille une limite constatée de recherche et de génération, ainsi que les contrôles de provenance ajoutés.

## Campagne complète disponible

Les 120 réponses des six configurations Llama 3 Q2_K ont été évaluées : qualité, biais NLI et énergie. Le tableau complet pour l’optimisation est `results/evaluations/campagne_complete/configurations_completes.csv`. Consulter [les résultats et limites](docs/resultats_campagne.md).

Pour exécuter ou reprendre la campagne : `python -m migrant_ai.campaign`. Voir [le protocole de reprise](docs/campagne_complete.md). Les erreurs du modèle sont conservées dans les résultats.

## Périmètre par rapport à l’article

La campagne dite complète couvre les six configurations du protocole local sur 20 questions. Elle ne reproduit pas l’ensemble des expériences du PDF (125 configurations sur 91 questions). Les méthodes supplémentaires sont désormais proposées dans le lanceur expérimental et leur état d’exécution est documenté séparément ; cela ne transforme pas les anciens résultats en reproduction du PDF. Voir [l’audit de correspondance](docs/conformite_article.md) pour le détail et les responsabilités.

## Méthodes complémentaires et essais réels

La perplexité causale, les catégories sociales du PDF, le filtrage des passages et le résumé BART sont disponibles dans la même application Python, via les [commandes expérimentales documentées](docs/protocole_experiences.md). Le profil `protocols/article.json` décrit la grille de 125 configurations ; le profil `protocols/local.json` est une adaptation française pour ce PC. Aucun modèle n’est chargé par la simple consultation du plan.

Le pilote apparié dans `results/evaluations/essai_integration/` couvre une question réelle dans les deux variantes. Il comprend des mesures effectives de qualité, de biais, d’énergie et de perplexité. Les deux réponses sont identiques sur ce cas : ce pilote ne démontre pas un gain de qualité. Les passages courts n’ont pas été résumés ; BART a été testé séparément sur le texte anglais du PDF. Le pilote est exclu du tableau des configurations complètes sur 20 questions.

Les tableaux de `data/article/` sont les données publiées par les auteurs, attribuées et vérifiées ; ce ne sont pas nos mesures. Les 91 textes question/référence originaux sont absents de ces fichiers. Ils ne sont pas inventés et cette campagne n’est pas exécutée, conformément au périmètre décidé par l’équipe.

Les compléments sont terminés sur les 120 réponses : perplexité et cinq indicateurs sociaux français. Le tableau local le plus complet est `results/evaluations/complements_complets/configurations_completes.csv`. Lire [les méthodes, résultats et limites](docs/resultats_complementaires.md) avant de comparer ces critères.
