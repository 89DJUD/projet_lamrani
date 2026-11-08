# Audit de correspondance avec l’article

## Conclusion

La chaîne fonctionnelle LLM et évaluation est implémentée et une campagne réduite est terminée. La reproduction complète du protocole de l’article n’est pas terminée. Ces deux états ne doivent pas être confondus. Les 120 réponses existantes restent valides comme résultats du protocole adapté, sans garantir leur justesse factuelle.

Source examinée : Castellanos-Nieves et García-Forte, *Human-Centered AI for Migrant Integration Through LLM and RAG Optimization*, Applied Sciences 2025, 15, 325, DOI 10.3390/app15010325, PDF de 26 pages fourni par l’équipe. Les pages ci-dessous correspondent aux pages imprimées du PDF. L’article est une référence méthodologique, pas une instruction autorisant à modifier le dépôt initial.

Audit réalisé le 8 novembre 2026 sur la version Git cbdd870. Aucun code ni résultat initial n’a été modifié pour cet audit. Aucun nouveau calcul de génération n’a été lancé.

## Correspondance détaillée

| Élément | Article | Dépôt actuel | État et suite |
|---|---|---|---|
| Génération | Meta-Llama-3-8B-Instruct, §5.4 p.12 | `migrant_ai/generation.py` ; campagne `llama3:8b-instruct-q2_K`, contexte 2048, sortie limitée à 150 tokens | Fonction présente, quantification et paramètres adaptés ; aucune équivalence expérimentale démontrée avec le modèle de l’article |
| Questions et corpus | 91 questions, sources principalement UE/Espagne, §5.2 p.10 | `test_set.csv` : 20 questions françaises ; huit PDF du corpus existant | Jeu local réduit ; utiliser les données originales pour une reproduction fidèle, sans écraser le jeu local ni les fichiers initiaux |
| Grille expérimentale | 125 configurations : cinq tailles 50/100/300/500/1000, cinq top-k 1/2/5/8/10 et cinq températures 0.1/0.3/0.5/0.7/0.9, §5.4 p.12 et tableau 1 p.14 | Six configurations : (50,2) et (100,5), chacune aux températures 0.1/0.5/0.9 ; 120 lignes | Campagne complète pour ce sous-ensemble seulement. Le lanceur avec reprise `campaign.py` fixe six configurations et 20 questions ; il doit être généralisé pour la grille de l’article |
| Plan expérimental | Croisement des trois facteurs | Taille et top-k changent ensemble dans les six configurations | Les résultats ne permettent pas d’isoler l’effet de la taille de celui du top-k. D peut classer six alternatives, mais pas reproduire les analyses factorielles de l’article |
| Recherche et embeddings | BAAI/bge-base-en-v1.5, §5.4 p.12 | `rag.py` : paraphrase-multilingual-MiniLM-L12-v2, index FAISS existants | Adaptation au français. Relève du corpus/RAG ; conserver les fichiers initiaux et ajouter toute variante séparément |
| Similarité | all-MiniLM-L6-v2, §5.4 p.12 | Même modèle multilingue que la recherche, cosinus borné à [0,1] | Fonction présente, modèle et traitement adaptés ; scores non directement comparables à l’article |
| BLEU | Comparaison des n-grammes, §3.4 p.7 | SacreBLEU par réponse, score divisé par 100 puis médiane | Fonction présente ; consigner la tokenisation et le lissage pour une comparaison numérique fidèle |
| ROUGE | Formules de rappel ROUGE-N et ROUGE-L, §3.5 pp.7–8 | ROUGE-1/2/L F1 avec tokenisation Unicode française, puis médiane | Fonction présente, variante différente des formules publiées. Ajouter le rappel à côté du F1, sans remplacer les scores déjà calculés |
| Perplexité | §3.6 p.8, §5.3 p.10, tableau 1 p.14 | Pas de calcul ni colonne de perplexité dans les nouveaux exports | Manquant pour une reproduction du PDF ; calculer à partir des probabilités token par token, avec modèle et conditionnement explicités. Ne pas déduire la perplexité de BLEU, de la similarité ou du temps |
| Stéréotypes | Narrativa/distilroberta-finetuned-stereotype-detection, §5.4 p.12 ; Stereo/Anti-Stereo/Neutral dans les résultats | NLI multilingue mDeBERTa, hypothèse de généralisation négative ; seul `stereotype_score` | Indicateur exploratoire différent ; Anti-Stereo et Neutral absents. Les trois catégories ne se reconstruisent pas par complément à 1 d’un score NLI |
| Haine | Hate-speech-CNERG/dehatebert-mono-spanish, §5.4 p.12 ; Hate/Non_Hate dans les résultats | NLI multilingue avec hypothèse d’exclusion ; `hate_score` | Adaptation au français, pas le classificateur de l’article. Ne pas renommer les scores ni les présenter comme des probabilités calibrées. Une reproduction du modèle espagnol exige des données/langues compatibles |
| Contrôle des biais du corpus | RoBERTa et expertise humaine, §5.2 p.10 | Pas de protocole équivalent constaté dans la chaîne active | Travail partagé corpus/évaluation ; annotations humaines à préparer et valider par l’équipe |
| Filtrage avant génération | Filtrage des documents par les détecteurs, §5.5 p.13 | La recherche fournit directement les passages au générateur ; les biais sont évalués sur les réponses ensuite | Manquant pour reproduire le processus intégré. Évaluer et tracer une nouvelle variante sans changer les références ni les index initiaux |
| Résumé du contexte | facebook/bart-large-cnn, §5.5 p.13 | Pas d’étape de résumé avant Llama | Manquant. L’usage français et la capacité mémoire imposent un choix explicite ; un changement de modèle serait une adaptation, pas une reproduction identique |
| Impact environnemental | Énergie CPU/GPU/TPU/RAM, intensité carbone et PUE, §3.3 pp.6–7 | CodeCarbon, CPU local, pays MAR, énergie kWh et émissions kg CO₂, sur recherche + génération | Mesure estimée présente ; matériel, outil, périmètre et unités différents. PUE et composants détaillés non exportés. Pas de comparaison absolue directe avec les tableaux de l’article |
| Matériel et logiciel | i7-12700, 32 Go RAM, RTX 4090 24 Go ; Python 3.8, CUDA 11.8, §5.5 p.13 | PC environ 8 Go, CPU, Python 3.12, Q2_K via Ollama | Adaptation matérielle substantielle. Les temps et émissions ne sont pas reproductibles à l’identique sur ce PC |
| Agrégation | Médianes des 91 questions, §5.4 p.12 ; écarts-types des scores sociaux, tableau 2 p.15 | Médianes des 20 questions ; pas d’export des écarts-types | Médianes présentes ; dispersion sociale à ajouter pour des tableaux comparables. Les lignes individuelles sont conservées |
| Comparaison avant/après intégration | Filtrage + résumé comparés au système sans intégration, §6.1 pp.18–19 | Une seule chaîne actuelle évaluée ; ni campagne appariée avec filtrage/résumé ni démonstration de leur gain | Manquant. Les anciens résultats et la nouvelle campagne changent plusieurs facteurs : ce n’est pas une comparaison contrôlée avant/après |
| Optimisation | RIM/TOPSIS/VIKOR sur 125 alternatives, §4 p.9 et §6.1 p.18 | CSV pour six alternatives ; méthodes non implémentées par cette contribution | Relève de la prochaine contribution. D peut commencer sur six alternatives, sans prétendre reproduire le classement des 125 alternatives de l’article |

## Ce qui est acquis

- 120 réponses, dont six abstentions automatiques ; 114 appels LLM réels.
- Six groupes de 20 questions, avec BLEU, ROUGE F1, similarité multilingue, scores NLI et 120 estimations environnementales.
- Provenance du contexte, version de consigne, modèle exact et empreintes du protocole conservés.
- Sauvegarde par question et reprise ; 30 tests automatisés réussis avant publication.
- Les fichiers initiaux protégés sont conservés. Les tests ne prouvent ni l’équivalence au PDF ni la justesse factuelle des réponses.

## Travail restant et responsabilités

1. **LLM et évaluation** : ajouter une vraie perplexité ; exporter les variantes de rappel ROUGE et la dispersion ; définir puis valider la méthode sociale (catégories manquantes et compatibilité linguistique). Garder les anciens scores et les nouvelles versions séparés.
2. **Corpus/RAG et évaluation, conjointement** : décider si l’objectif est une adaptation française ou une reproduction des données de l’article ; obtenir et vérifier les 91 questions et leurs références si nécessaire. Ne pas fabriquer les questions originales ni transformer automatiquement 20 questions en prétendu jeu de 91.
3. **Intégration** : implémenter, dans de nouveaux modules, les variantes de filtrage et de résumé ; garder une chaîne témoin pour mesurer leur effet sur les mêmes questions et configurations.
4. **Expérimentation** : généraliser le lanceur avec reprise à une grille configurable et à un jeu de taille quelconque ; vérifier les limites de contexte et tracer les troncatures éventuelles avant une campagne plus grande. La grille complète représente 11 375 couples configuration/question par variante (125 × 91), ce qui dépasse largement la campagne actuelle. Aucun téléchargement lourd ni calcul de cette ampleur n’est lancé par cet audit.
5. **Optimisation** : D implémente RIM, TOPSIS et VIKOR, documente les poids/sens des critères et compare les classements. D ne remplace pas les métriques ou expériences manquantes ; les tableaux d’entrée doivent expliciter leur périmètre.
6. **Rapport d’équipe** : présenter les écarts méthodologiques et matériels, les erreurs factuelles et les limites des mesures. Les réductions de CO₂ et de biais publiées dans l’article ne sont pas des résultats de notre projet et ne doivent pas être reprises comme tels.

## Statut à communiquer

« Notre contribution LLM et évaluation fonctionne, et la campagne réduite de six configurations sur 20 questions est terminée. Elle est une adaptation française et matérielle du protocole. La reproduction complète de l’article reste à compléter, notamment pour la perplexité, les catégories sociales, le filtrage/résumé et l’étendue des expériences. L’optimisation multicritère constitue l’étape suivante de l’équipe. »

## Suite de l’audit : code ajouté et périmètre retenu

La matrice précédente décrit l’état au commit cbdd870. Depuis cet état, des modules ont été ajoutés pour la perplexité teacher-forcing (GGUF ou Transformers), toutes les classes des deux modèles sociaux du PDF, les détails ROUGE et écarts-types, le filtrage, BART et un lanceur apparié avec reprise. Voir `protocole_experiences.md` pour les commandes et les adaptations.

Des contrôles réels séparés ont été exécutés sur le même GGUF Llama Q2, les deux classificateurs officiels et BART. Ils ne constituent pas une validation scientifique de l’exactitude des détecteurs. Le détecteur de stéréotypes a notamment attribué un score très élevé à une phrase inclusive, et le détecteur de haine n’a pas dépassé 0,5 sur la phrase d’exclusion du diagnostic. Ces limites sont conservées dans les rapports, sans retoucher les scores.

Le pilote apparié sur une question française a terminé les deux variantes et leurs métriques. Il ne démontre pas une amélioration et ne remplace pas une étude appariée des 20 questions. BART étant entraîné en anglais, son application au corpus français demeure une adaptation à évaluer, pas une équivalence établie au protocole publié.

Les fichiers publics des auteurs ont été retrouvés et vérifiés : ils contiennent 11 375 observations de métriques et des tableaux agrégés, mais pas les textes des 91 questions/réponses. L’équipe a explicitement décidé de ne pas inventer ni exécuter ces questions introuvables. La grille complète est donc un plan disponible, pas une campagne réalisée. Les jeux de données publiés restent séparés de nos mesures locales.

Précision sur ROUGE : les formules du PDF sont présentées en rappel, tandis que le CSV agrégé des auteurs porte explicitement les noms `rouge*_fmeasure`. Notre export conserve le F1 initial et ajoute précision/rappel ; on ne prétend pas qu’une seule variante prouve une reproduction numérique exacte.
