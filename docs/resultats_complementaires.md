# Évaluation complémentaire terminée

Les 120 réponses de la campagne locale ont maintenant leur perplexité réelle et cinq indicateurs sociaux. Les textes prédits, les références, les contextes, les scores de qualité précédents et les estimations environnementales initiales sont conservés. Aucune réponse Llama n’a été régénérée pour cet enrichissement.

| Configuration | Mots / top-k / température | Perplexité médiane | Stéréotypes | Anti-stéréotypes | Neutralité | Haine | Non-haine |
|---|---|---:|---:|---:|---:|---:|---:|
| C1 | 50 / 2 / 0.1 | 11.496 | 0.0136 | 0.0422 | 0.0488 | 0.0002 | 0.6751 |
| C2 | 50 / 2 / 0.5 | 10.157 | 0.0196 | 0.1477 | 0.1162 | 0.0002 | 0.6030 |
| C3 | 50 / 2 / 0.9 | 10.456 | 0.0157 | 0.0597 | 0.1133 | 0.0002 | 0.5935 |
| C4 | 100 / 5 / 0.1 | 12.030 | 0.0071 | 0.0825 | 0.0398 | 0.0001 | 0.5378 |
| C5 | 100 / 5 / 0.5 | 9.177 | 0.0074 | 0.0549 | 0.0383 | 0.0002 | 0.5525 |
| C6 | 100 / 5 / 0.9 | 9.237 | 0.0023 | 0.0332 | 0.0424 | 0.0002 | 0.5312 |

Chaque configuration couvre les 20 questions. Ces scores sont calculés sur les réponses enregistrées, avec toutes les limites factuelles déjà constatées.

## Méthode et interprétation

La perplexité est calculée par teacher-forcing sur le même fichier Llama Q2_K que la génération (SHA-256 eaa3ffa9267f65b2beebac4803927e752bd5ac6cc8d0e256e38e9df2ec214af4). Elle porte sur le texte prédit non conditionné par le contexte ; fenêtre de 256 tokens, pas de 128, chaque token cible compté une fois. Les textes strictement identiques réutilisent leur score déterministe. Ce score mesure la prédictibilité du texte, pas sa justesse juridique.

Les cinq indicateurs français utilisent le modèle NLI multilingue déjà présent, avec cinq hypothèses indépendantes. Ils ne constituent pas une distribution multiclasses et ne sont pas les probabilités des deux détecteurs de l’article. Les modèles exacts de l’article restent disponibles dans le profil dédié, mais leurs performances sur le corpus français ne sont pas établies.

Le traitement des cinq hypothèses en un lot a montré un écart de 0,002972 sur le diagnostic par rapport au traitement séquentiel. Il n’a donc pas été déclaré équivalent au seuil 1e-5. Les 120 scores complémentaires ont été entièrement recalculés avec une seule configuration enregistrée : `independent-nli-mean-batch5-v3`. Les anciens scores sont conservés dans les colonnes suffixées `_previous` et dans la campagne initiale. Ne pas interpréter leur différence comme une réduction réelle des biais : les réponses n’ont pas changé.

Le seuil 0,5 signale 3 réponses sur 120 dans cette version. Ces signaux demandent une lecture humaine ; ils ne sont pas des biais confirmés. Les scores nuls ou faibles ne prouvent pas une absence de biais.

L’énergie et le CO₂ restent les mesures originales sur recherche/génération. Le coût des nouvelles métriques n’a pas été mesuré ; aucune consommation supplémentaire n’est inventée ni remplacée par zéro.

## Fichiers pour la suite

- `results/evaluations/complements_complets/responses.csv` : les 120 réponses et leurs métriques, anciennes valeurs conservées.
- `results/evaluations/complements_complets/configurations_completes.csv` : six alternatives complètes avec versions d’évaluation, médianes, écarts-types et nombres de valeurs disponibles.
- `protocol.json`, les trois sauvegardes JSONL et `source_execution.zip` : provenance et code réellement exécuté.

Le pilote avec/sans filtrage/résumé reste limité à une question et exclu du classement complet. Les 91 questions originales n’ont pas été retrouvées dans les fichiers publiés consultés : elles ne sont pas inventées et leur campagne n’est pas exécutée. La grille des 125 configurations est un plan disponible, pas un résultat local réalisé.
