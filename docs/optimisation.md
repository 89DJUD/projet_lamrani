# Données pour l’optimisation multicritère

Le classement RIM, TOPSIS et VIKOR reste à implémenter par la personne chargée
de l’optimisation. Les fichiers ci-dessous constituent son interface de travail.

## Résultats historiques conservés

- `results/llm_raw.csv` : 128 réponses et leurs scores d’origine.
- `results/resultats_pour_D.csv` : six configurations complètes, 20 questions chacune.
- `results/retrieval_results.csv` : qualité de la recherche (Hit@K et MRR).
- `results/top1_scores.csv` : justification du seuil de recherche.

Ces fichiers et leurs colonnes ne sont pas modifiés. La configuration de
100 mots / top-10 / température 0,1 n’a que 8 réponses ; elle ne doit pas être
comparée comme une configuration complète aux autres.

## Réévaluation multilingue

`results/evaluations/reevaluation_multilingue/` contient :

- `responses.csv` : les 128 textes originaux, avec les scores recalculés.
- `configurations.csv` : toutes les configurations, y compris les incomplètes.
- `configurations_completes.csv` : six configurations de 20 questions.

Les identifiants C1, C2… sont propres à chaque tableau. Pour rapprocher deux
exports, utiliser `model`, `chunk_size`, `top_k` et `temperature`.
Les anciens scores de similarité utilisaient un modèle principalement anglophone ;
les nouveaux utilisent le modèle multilingue de la recherche. Ne pas mélanger
ces deux versions de métriques dans une même comparaison.

| Critère | Colonne des nouveaux résumés | Sens |
|---|---|---|
| BLEU | `bleu_median` | Maximiser |
| ROUGE | `rouge1_median`, `rouge2_median`, `rougeL_median` | Maximiser |
| Similarité | `semantic_similarity_median` | Maximiser |
| Latence | `seconds_median` | Minimiser |
| Stéréotypes et haine | `stereotype_score_median`, `hate_score_median` | Minimiser après validation |
| Énergie | `energy_kwh_median` | Minimiser |
| Empreinte carbone | `co2_kg_median` | Minimiser |

L’énergie et le CO2 des réponses historiques sont inconnus : les cellules sont
vides, jamais remplacées par zéro. Vérifier `n_energy_available` avant d’utiliser
ces critères, exclure les critères sans mesures et renormaliser les poids.
Les scores lexicaux sont exploratoires ; leur valeur nulle ne prouve pas
l’absence de biais. Une validation humaine reste nécessaire.

La variante `llama3:8b-instruct-q2_K` a une quantification et un contexte différents
de `llama3:8b`. Les nouvelles expériences enregistrent le nom exact du modèle ;
conserver cette différence dans les analyses et dans le rapport de classe.

## Essai réel avec mesures environnementales

`results/evaluations/essai_court/` contient deux générations réelles avec
Llama 3 Q2_K, leurs scores et leurs mesures d’énergie et de CO2. Il s’agit d’un
contrôle de fonctionnement. Le fichier `configurations_completes.csv` y est vide,
car le jeu de vingt questions n’a pas été entièrement évalué. Ne pas classer
cet essai partiel contre les configurations complètes historiques.
Une réponse diffère de la référence : conserver cet écart dans l’analyse de
qualité. L’optimisation doit comparer des campagnes complètes et des métriques
calculées avec les mêmes méthodes.

## Traçabilité des nouvelles expériences

Les nouvelles réponses conservent le contexte fourni au modèle et les sources
réellement retrouvées. La colonne `source` du jeu initial reste la source de la
référence ; `retrieved_sources` est une liste JSON des passages effectivement
utilisés. Ces deux colonnes peuvent différer.
`prompt_version` distingue les consignes de génération et fait partie des clés
d’agrégation quand elle est disponible. Une version historique non enregistrée
est notée `unrecorded`. Ne pas fusionner des versions différentes dans un même
classement sans les traiter comme des configurations distinctes.
