# Exécuter et vérifier les méthodes de l’article

## Périmètre décidé par l’équipe

Les 91 questions originales ne figurent pas dans les fichiers publics identifiés. Elles ne seront pas inventées ; la campagne correspondante n’est pas exécutée. Le jeu `test_set.csv` de 20 questions françaises est conservé. Les 125 configurations du PDF sont disponibles comme plan ; un plan ne constitue pas un résultat expérimental.

La campagne historique de 120 réponses reste dans `results/evaluations/campagne_complete/`. Les compléments de précision/rappel ROUGE et les écarts-types calculés sur ces mêmes réponses sont dans `results/evaluations/complements_rouge/`. Aucun appel Llama ni mesure énergétique supplémentaire n’a été fabriqué pour ces compléments.

## Modules intégrés

- `paper_metrics.py` : classificateurs exacts Narrativa et Hate-speech-CNERG avec toutes leurs classes ; variantes ROUGE F1/précision/rappel ; médianes, écarts-types et nombres de valeurs disponibles. Les labels sont lus explicitement, jamais reconstruits par complément d’un score manquant.
- `perplexity.py` : calcul teacher-forcing à fenêtres glissantes, sans compter deux fois les tokens recouvrants. Le score porte sur le texte prédit non conditionné par le contexte ; ce choix est explicite car le conditionnement du calcul publié n’est pas suffisamment précisé. Le backend GGUF réutilise les poids quantifiés de la génération ; le backend Transformers permet le modèle original sur matériel adapté. Les perplexités de modèles/tokenizers différents ne sont pas directement comparables.
- `context_processing.py` : filtrage par indicateurs de stéréotype/haine, traçabilité des passages rejetés, résumé BART de chaque fenêtre trop longue. Les passages courts sont conservés. Un résumé est un texte généré, pas une citation littérale et pas une garantie de fidélité.
- `run_study.py` : chaîne témoin et chaîne intégrée, mêmes questions et paramètres, sauvegarde par réponse et par étape, données appariées et différences. Aucun tableau complet n’est exporté si une étape requise manque. Les erreurs restent visibles et une relance reprend les étapes sauvegardées.

Le seuil de filtrage 0,5, les consignes, la taille des fenêtres, la limite de sortie et la stratégie d’agrégation des fenêtres sont des choix du projet, le PDF ne donnant pas tous ces détails. Les modèles du PDF sont disponibles dans le profil `article.json`, mais les textes/corpus originaux sont requis pour l’exécuter. Aucun profil n’affirme garantir une reproduction numérique exacte.

## Préparation explicite

Depuis la racine du dépôt avec les dépendances de l’application installées :

```powershell
python -m migrant_ai.prepare_models
python -m pip install -r requirements-gguf.txt
```

La préparation télécharge environ 2,7 Go et vérifie les empreintes des poids publiés. Les modèles restent dans `.runtime/`, exclus de Git. Pour Windows, une roue CPU compatible peut être nécessaire : consulter les releases officielles de `abetlen/llama-cpp-python` plutôt que compiler ou installer un binaire non identifié.

Pour la perplexité GGUF, renseigner dans une copie du profil le chemin vers le fichier utilisé par Ollama. Le lanceur vérifie que son SHA-256 correspond au modèle de génération. Ne pas utiliser un autre modèle à la place sans changer le protocole et expliquer le rôle du score.

## Plan, petit essai puis campagne

```powershell
python -m migrant_ai.run_study --profile protocols/article.json
python -m migrant_ai.run_study --profile protocols/local.json
python -m migrant_ai.run_study --profile protocols/local.json --run --limit 1 --output results/evaluations/essai_integration
```

Sans `--run`, aucun modèle n’est chargé et aucune expérience n’est exécutée. `--limit 1` évalue la première question réelle dans les deux variantes et porte explicitement le statut de pilote. Retirer la limite uniquement après vérification de cet essai. Le profil local est une adaptation française Q2_K ; BART est entraîné en anglais et les scores NLI sont exploratoires. Le profil de l’article conserve les modèles publiés, dont le détecteur de haine espagnol ; l’application automatique de ce détecteur au français ne fournit pas une validation linguistique.

Dans le profil local, la faible RAM impose de libérer les modèles entre les étapes ; cela ralentit le calcul. La mesure CodeCarbon du nouveau lanceur comprend le chargement, la recherche, le filtrage, le résumé et la génération, mais pas les métriques calculées ensuite. Son périmètre est différent de l’ancienne campagne. Ne pas mélanger leurs valeurs énergétiques comme si le protocole était identique. L’estimation porte sur la machine entière ; les autres applications l’influencent.

## Compléter les réponses déjà sauvegardées

```powershell
python -m migrant_ai.complete_evaluation --input results/evaluations/campagne_complete/responses.csv --output results/evaluations/nouveaux_complements
```

Par défaut cette commande ne calcule que les nouveaux détails ROUGE et la dispersion. `--social paper` et `--social multilingual_nli` sont des choix explicites qui chargent les modèles ; `--perplexity-gguf CHEMIN` mesure réellement la perplexité. Sans ces options, les valeurs absentes restent absentes et leur statut indique `not_run`. Cette commande conserve les anciens scores et l’énergie historique ; elle ne reconstitue pas une mesure manquante.

## Sources techniques

- Article : DOI 10.3390/app15010325, §§3.4–3.8 et 5.4–5.5.
- Données des auteurs, attribution et SHA-256 : `data/article/README.md`.
- Classificateurs : https://huggingface.co/Narrativa/distilroberta-finetuned-stereotype-detection et https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-spanish .
- BART et limites linguistiques : https://huggingface.co/facebook/bart-large-cnn .
- Calcul causal à fenêtres glissantes : https://huggingface.co/docs/transformers/perplexity .
- Moteur CPU : https://github.com/abetlen/llama-cpp-python .

## Enrichissement réellement exécuté

`python -m migrant_ai.enrich_results --input results/evaluations/campagne_complete/responses.csv --output NOUVEAU_DOSSIER --gguf CHEMIN_DU_MEME_MODELE` calcule les compléments avec reprise. Le dossier `results/evaluations/complements_complets` contient l’exécution terminée. Son code exact est archivé ; une nouvelle version du code nécessite un nouveau dossier. Voir `resultats_complementaires.md`. Le profil de l’article exige un GPU CUDA adapté, l’accès autorisé au modèle Meta et les données originales ; ce profil n’a pas été exécuté sur ce PC.
