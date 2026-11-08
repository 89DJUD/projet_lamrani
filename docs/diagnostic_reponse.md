# Audit de la réponse incorrecte

La deuxième question de l’essai attend un délai général de neuf mois d’après
la référence fournie et l’article 5(4) de la directive 2003/86/CE. La réponse
initiale indiquait 90 jours en utilisant une dérogation relative aux chercheurs.

La référence a été contrôlée dans la page 4 du corpus et dans le
[texte officiel](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32003L0086).
Les questions, références et réponses sont alignées ; les scores calculés
portent sur les bons couples. Le contrôle technique de génération ne suffit
pas à valider l’exactitude d’une réponse.

## Reproduction

| Contexte | Consigne | Réponse observée |
|---|---|---|
| 50 mots, top-2 : règle générale absente | Initiale | 90 jours, incorrect pour la référence |
| Mêmes deux extraits | Portée renforcée | 90 jours, erreur persistante |
| 100 mots, top-5 : règle générale en quatrième position | Portée renforcée | 90 jours, erreur persistante |
| Paragraphe correct seul, extrait de la page de référence | Portée renforcée | Neuf mois |

Les journaux serveur ne signalent pas de troncature pour ces essais. Le modèle
peut lire la bonne règle, mais privilégie ici un passage spécial quand plusieurs
régimes sont mélangés. Ce test ne permet pas d’attribuer ce comportement à la
seule quantification : aucune comparaison à précision différente n’a été faite.

Le dernier contrôle utilise volontairement la source de référence pour isoler
le problème. Cette source n’est jamais injectée automatiquement dans les
expériences ordinaires : ce serait une fuite du jeu de test.

## Changements apportés

- Consigne versionnée `scoped-context-v2` : champ d’application, règles générales,
  exceptions et abstention explicites. Cette consigne seule n’a pas suffi ici.
- Enregistrement de `context`, `retrieved_sources`, `context_origin`,
  `generation_skipped` et `prompt_version` dans les nouvelles expériences.
- `source` reste la source attendue de la référence ; elle ne représente pas
  nécessairement les passages effectivement présentés au modèle.
- Agrégation séparée des versions de consigne pour éviter de mélanger les résultats.
- 27 tests de code réussis, dont conservation des sources et contrôle de provenance.

Le code original de recherche, les index, les références et les résultats
historiques restent inchangés. Les mauvaises réponses sont conservées pour
l’évaluation. La qualité de réponse n’est pas encore déclarée validée sur ce cas.
Il reste à comparer les configurations et à améliorer la sélection des passages
applicables ; aucune réponse attendue n’est codée en dur.

Les réponses détaillées et les sources de chaque essai sont dans
`answer_error_audit.json`. Ces contrôles diagnostiques sont séparés des campagnes
complètes utilisées pour le classement multicritère.
