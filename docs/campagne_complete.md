# Campagne complète avec reprise

Depuis la racine du dépôt, avec les dépendances installées et Ollama démarré :

```powershell
python -m migrant_ai.campaign
```

Le modèle requis est `llama3:8b-instruct-q2_K`, évalué séparément du modèle original. Les six configurations sont (50 mots, top-k 2) et (100 mots, top-k 5), chacune aux températures 0.1, 0.5 et 0.9. Les 20 questions sont conservées, y compris les abstentions : 120 lignes attendues.

Chaque résultat est sauvegardé atomiquement dans `results/evaluations/campagne_complete/generation.jsonl`. Une interruption permet de relancer la même commande : les lignes déjà sauvegardées ne sont pas recalculées. Une modification du code, des index, du jeu de test ou du modèle exige un nouveau dossier `--output`. Ne pas modifier les fichiers du protocole pendant une campagne.

Les scores BLEU, ROUGE et similarité multilingue accompagnent chaque réponse. CodeCarbon estime la consommation de toute la machine pendant la recherche et la génération, en excluant l'initialisation des capteurs. Les autres applications peuvent influencer cette estimation. Les valeurs indisponibles restent manquantes. Le classificateur de biais est exécuté ensuite, hors de cette mesure énergétique, avec sa propre sauvegarde `bias.jsonl`. Ses scores sont exploratoires et nécessitent une interprétation humaine.

Les fichiers définitifs `responses.csv`, `configurations.csv` et `configurations_completes.csv` ne sont produits qu'après les deux étapes et la vérification de la couverture. Ils ne garantissent pas la justesse factuelle des réponses. Le cas de confusion entre neuf mois et 90 jours reste un échec connu à observer sans modifier les références.

Le PC doit rester allumé, branché et sans mise en veille. Les journaux locaux se trouvent dans `reports/campaign.log` et `reports/campaign-error.log`. Les calculs peuvent durer plusieurs heures sur une machine de 8 Go. Les modèles, environnements locaux et journaux ne sont pas destinés à GitHub.

## Cache et première exécution

Le lanceur utilise par défaut le cache Hugging Face hors ligne pour éviter les téléchargements pendant les mesures. Sur une nouvelle machine, précharger les modèles ou autoriser leur premier téléchargement avec `$env:HF_HUB_OFFLINE = "0"` avant de lancer la commande. Les modèles nécessaires sont `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` et `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`. Une fois les modèles disponibles, utiliser `$env:HF_HUB_OFFLINE = "1"` pour une campagne sans accès réseau.
