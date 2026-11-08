# Résultats de la campagne complète

La campagne Llama 3 8B Q2_K est terminée : 120 réponses sur six configurations, avec 20 questions par configuration. Elle comprend 114 appels LLM et six abstentions automatiques hors corpus. Les références et les réponses, y compris incorrectes, sont conservées.

| Configuration | Mots / top-k / température | BLEU | ROUGE-L | Similarité | Secondes | Énergie (Wh) |
|---|---|---:|---:|---:|---:|---:|
| C1 | 50 / 2 / 0.1 | 0.087 | 0.248 | 0.735 | 44.8 | 0.236 |
| C2 | 50 / 2 / 0.5 | 0.106 | 0.249 | 0.748 | 43.8 | 0.223 |
| C3 | 50 / 2 / 0.9 | 0.076 | 0.228 | 0.742 | 45.8 | 0.235 |
| C4 | 100 / 5 / 0.1 | 0.095 | 0.249 | 0.667 | 79.3 | 0.415 |
| C5 | 100 / 5 / 0.5 | 0.080 | 0.231 | 0.726 | 68.6 | 0.385 |
| C6 | 100 / 5 / 0.9 | 0.079 | 0.280 | 0.637 | 67.9 | 0.373 |

Les valeurs du tableau sont des médianes sur les 20 questions, abstentions comprises. C2 obtient les médianes BLEU et similarité les plus élevées et les médianes de latence et d’énergie les plus faibles de cette campagne. C6 obtient la meilleure médiane ROUGE-L. Cela ne constitue pas un classement multicritère : les poids et les méthodes RIM, TOPSIS et VIKOR restent à étudier.

## Limites observées

29 réponses sur 120 ont une similarité inférieure à 0,5. Ce seuil est descriptif : la similarité ne mesure pas directement la justesse factuelle et 0,74 ne signifie pas 74 % de réponses correctes. Les scores BLEU et ROUGE comparent les formulations aux références, sans garantir la validité des chiffres ou des citations.

La confusion connue sur le délai de regroupement familial demeure pour quatre configurations. Deux autres réponses mentionnent neuf mois, mais contiennent encore des affirmations ou citations problématiques. Le cas ne peut donc pas être déclaré résolu. Aucun correctif des références ni sélection des réponses favorables n’a été appliqué.

Le classificateur NLI signale six réponses au seuil 0,5 sur au moins un des deux indicateurs. Il s’agit de signaux exploratoires, pas de six cas de biais confirmés. Une lecture humaine des réponses signalées est nécessaire ; les médianes faibles ne prouvent pas l’absence de biais.

Les 120 mesures CodeCarbon sont disponibles. Leur somme sur les seuls intervalles mesurés est de 0,038132 kWh et 0,024024 kg de CO₂. Ce total exclut les téléchargements, l’initialisation des capteurs, les calculs de qualité et la seconde étape NLI. Les estimations concernent toute la machine pendant la recherche et la génération ; les autres applications peuvent les influencer. Elles ne sont pas des mesures au wattmètre.

## Fichiers et vérification

- `results/evaluations/campagne_complete/responses.csv` : réponses, références, contexte, sources, qualité, biais NLI et énergie.
- `results/evaluations/campagne_complete/configurations_completes.csv` : tableau de six configurations utilisable pour l’optimisation.
- `protocol.json` : modèle et digest, paramètres, empreintes du jeu de test, des index et du code utilisé.
- `generation.jsonl` et `bias.jsonl` : sauvegardes des deux étapes. `responses_generation.csv` conserve les indicateurs lexicaux calculés avant NLI.

Vérifications réalisées : 120 identifiants uniques, aucune réponse vide, couverture des 20 questions dans chaque configuration, scores bornés et finis, 120 mesures environnementales, 120 évaluations NLI et conformité des empreintes du protocole. Les tests automatisés ne remplacent pas une validation factuelle humaine exhaustive.

Le modèle Q2_K, le prompt scoped-context-v2 et les indicateurs NLI distinguent cette campagne des résultats historiques. Ne pas fusionner les campagnes comme s’il s’agissait du même protocole. Une seule exécution par configuration ne permet pas de quantifier la variabilité entre répétitions.

## Périmètre par rapport à l’article

La campagne dite complète couvre les six configurations du protocole local sur 20 questions. Elle ne reproduit pas l’ensemble des expériences du PDF (125 configurations sur 91 questions). Les méthodes supplémentaires sont désormais proposées dans le lanceur expérimental et leur état d’exécution est documenté séparément ; cela ne transforme pas les anciens résultats en reproduction du PDF. Voir [l’audit de correspondance](conformite_article.md) pour le détail et les responsabilités.
