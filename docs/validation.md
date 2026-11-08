# État de validation

- 30 tests unitaires exécutés avec succès dans la nouvelle organisation.
- Interface commune ouverte sans exception après réorganisation, avec la variante Q2_K.
- Parcours hors corpus vérifié avec la vraie recherche et les vrais embeddings.
- CodeCarbon exécuté sur une charge CPU de diagnostic de trois secondes.
- 128 réponses réévaluées avec les vrais embeddings multilingues.
- Aucune consommation énergétique historique inventée.

Les appels Llama et NLI des tests unitaires sont simulés. Le classifieur NLI a aussi
été exécuté réellement sur trois phrases synthétiques : voir validation_classifier.json.
Ce contrôle confirme le fonctionnement, pas une performance scientifique calibrée.
Llama 3 Q2_K a terminé son téléchargement et a généré deux réponses réelles.
BLEU, ROUGE, similarité et mesures CodeCarbon sont disponibles pour les deux.
Le test technique est réussi ; voir results/evaluations/essai_court/ et
validation_generation.json. Une réponse ne correspond pas à la référence : cette
limite est conservée pour l’analyse, sans sélection des seules bonnes réponses.
L’essai de deux questions ne remplace pas une campagne complète sur les vingt questions. Le diagnostic et le
contrôle de génération permettent de vérifier ces étapes après installation.
Le test CodeCarbon de diagnostic ne constitue pas une mesure d’inférence LLM.

Les tests de conservation vérifient les sources et données du dépôt initial.
Les fins de ligne des fichiers texte sont normalisées pour fonctionner sous
Windows et Linux. Le README et les règles Git sont maintenus comme documentation
et configuration du projet.

Un audit approfondi du cas incorrect est disponible dans [diagnostic_reponse.md](diagnostic_reponse.md). Le contrôle technique a réussi, mais la fidélité à la règle générale reste insuffisante sur ce cas, même après renforcement de la consigne.

## Campagne complète terminée

Les six configurations ont maintenant chacune 20 réponses avec scores de qualité, biais NLI et mesures CodeCarbon : 120/120 pour chaque étape. Voir [les résultats et leurs limites](resultats_campagne.md). Les paragraphes précédents décrivent les contrôles initiaux, notamment l’essai court, conservés pour la traçabilité. La campagne complète ne garantit pas la justesse de toutes les réponses.
