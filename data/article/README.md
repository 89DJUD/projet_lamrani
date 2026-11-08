# Données publiées par les auteurs

Auteurs : Dagoberto Castellanos Nieves et Luis García-Forte. Licence CC BY 4.0 : https://creativecommons.org/licenses/by/4.0/ . Les fichiers CSV sont conservés sans modification ; leurs empreintes sont dans `source.json` et `summary_sources.json`.

- `HCAI_Data.csv` : DOI 10.17632/x4x86r6tzd.1, 11 375 lignes de métriques, 125 configurations de 91 observations.
- `HCAI_Data_125.csv` : DOI 10.17632/rjdt5nmm88.2, tableau de 125 alternatives agrégées publié par les auteurs.
- `Integrated_data.csv` : même DOI, métriques publiées avec/sans intégration.

Ce sont **les résultats des auteurs**, pas des réponses générées par notre application. Ils ne contiennent pas les textes des 91 questions, les réponses de référence ni le corpus complet de l’expérience originale. La reproduction sur ces textes est donc exclue tant qu’ils restent introuvables, conformément à la décision de l’équipe.

Les cellules vides des catégories sociales restent vides ; elles ne sont pas remplacées par zéro. Les unités de `CO2e` restent celles de la publication, dont la présentation g/Wh est ambiguë : ne pas les convertir en kWh ou kg de nos mesures sans information supplémentaire. Le tableau agrégé publié nomme les colonnes ROUGE `fmeasure` ; le PDF décrit aussi des formules de rappel. Notre export distingue explicitement F1, précision et rappel.

Ne pas fusionner ces tableaux avec les mesures locales et ne pas présenter les améliorations chiffrées des auteurs comme des résultats de notre équipe. Les adaptations/modifications de code ne proviennent pas des auteurs et ne constituent pas leur validation du projet.
