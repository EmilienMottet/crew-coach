# n8n Workflows

Ce dossier contient les définitions de workflows n8n qui sont automatiquement déployés via GitHub Actions.

## 📁 Workflows disponibles

### 1. `update-strava-activity-from-interval-icu.json`
**Statut** : Inactif par défaut
**Description** : Met à jour automatiquement les activités Strava avec des descriptions générées par CrewAI basées sur les données Intervals.icu

**Déclencheur** : Webhook Strava (création d'activité)
**Flux** :
1. Strava Trigger → Nouvelle activité créée
2. HTTP Request → Appelle le service crew (`http://crew:8000/process`)
3. Update Activity → Met à jour le titre et la description sur Strava

### 2. `meal-planning-weekly.json`
**Statut** : Actif
**Description** : Génère automatiquement un plan de repas hebdomadaire personnalisé basé sur les données d'entraînement Hexis

**Déclencheur** : Cron schedule (Dimanches 20:00 Europe/Paris)
**Flux** :
1. Schedule Trigger → Exécution hebdomadaire
2. Calculate Next Monday → Calcule la date de début de semaine
3. Execute Meal Planning Crew → Lance `crew_mealy.py`
4. Parse Crew Output → Parse le JSON de sortie
5. Check Success → Vérifie le succès de l'exécution
6. Success/Error Notification → Envoie notification Telegram
7. Log Execution Result → Log pour debugging

**Variables d'environnement requises** :
- `TELEGRAM_CHAT_ID` : ID du chat Telegram pour les notifications
- Standard crew environment variables (dans le container)

## 🚀 Déploiement automatique

Les workflows sont automatiquement déployés vers n8n lors de :
- Push sur la branche `main` avec modifications dans `n8n/workflows/`
- Déclenchement manuel du workflow GitHub Actions
- Workflow appelé depuis un autre workflow

### Prérequis GitHub Secrets

Les secrets suivants doivent être configurés dans GitHub :
- `N8N_API_URL` : URL de l'API n8n (ex: `https://n8n.example.com`)
- `N8N_API_KEY` : Clé API n8n pour l'authentification

### Workflow GitHub Actions

Fichier : `.github/workflows/n8n-deploy.yml`

Le workflow :
1. Clone le repository
2. Valide la configuration
3. Pour chaque fichier JSON dans `n8n/workflows/` :
   - Extrait l'`id` du workflow
   - Tente de mettre à jour le workflow existant (PUT)
   - Si 404, crée un nouveau workflow (POST)
   - Log le résultat

## 📝 Ajouter un nouveau workflow

1. **Créer le fichier JSON** dans ce dossier
2. **S'assurer qu'il contient un `id` unique** (ex: `"id": "UniqueWorkflowId123"`)
3. **Commit et push** sur la branche `main`
4. **Le workflow sera automatiquement déployé** via GitHub Actions

### Structure minimale requise

```json
{
  "id": "unique-workflow-id",
  "name": "My Workflow Name",
  "active": true,
  "settings": {
    "executionOrder": "v1"
  },
  "nodes": [...],
  "connections": {...},
  "meta": {
    "templateCredsSetupCompleted": true
  },
  "tags": []
}
```

## 🔧 Configuration des credentials n8n

Certains workflows requièrent des credentials configurés dans n8n :
- **Strava OAuth2** : Pour les workflows Strava
- **Telegram Bot** : Pour les notifications (ID: `telegram-bot-credentials`)
- **HTTP Auth** : Si nécessaire pour les endpoints HTTP

### Configuration Telegram Bot

1. Créer un bot via [@BotFather](https://t.me/botfather)
2. Obtenir le Bot Token
3. Obtenir votre Chat ID via [@userinfobot](https://t.me/userinfobot)
4. Configurer dans n8n :
   - Settings → Credentials → Add Credential
   - Type : Telegram
   - Name : `Telegram Bot`
   - Access Token : [Votre Bot Token]
5. Définir `TELEGRAM_CHAT_ID` comme variable d'environnement n8n

## 🐛 Dépannage

### Workflow non déployé
- **Vérifier** : Les secrets GitHub sont configurés
- **Vérifier** : Le fichier JSON est valide (utiliser `jq` pour valider)
- **Vérifier** : L'`id` est unique et présent
- **Voir** : Les logs GitHub Actions pour les erreurs

### Workflow déployé mais n'exécute pas
- **Vérifier** : Le workflow est `"active": true` dans le JSON
- **Vérifier** : Les credentials sont configurés dans n8n
- **Vérifier** : Les variables d'environnement sont définies
- **Voir** : Les logs d'exécution dans l'interface n8n

### Erreurs de permission
- **Vérifier** : La clé API n8n a les permissions suffisantes
- **Vérifier** : L'URL de l'API n8n est correcte (pas de trailing slash)

## 📚 Documentation

- **n8n Docs** : https://docs.n8n.io
- **n8n API** : https://docs.n8n.io/api/
- **CrewAI Meal Planning** : Voir `MEAL_PLANNING_README.md` à la racine
- **CrewAI Strava** : Voir `CLAUDE.md` à la racine

## 🔄 Workflow de mise à jour

1. Modifier le fichier JSON localement
2. Tester localement si possible (importer dans n8n dev)
3. Commit et push sur `main`
4. Vérifier le déploiement dans GitHub Actions
5. Tester l'exécution dans n8n production
6. Monitorer les logs

## 📊 Monitoring

Pour monitorer les workflows :
- **Interface n8n** : Dashboard des exécutions
- **GitHub Actions** : Logs de déploiement
- **Telegram** : Notifications de succès/échec (meal planning)
- **Logs applicatifs** : stdout/stderr des containers crew

---

**Note** : Les workflows sont versionnés dans Git. Toujours modifier les fichiers JSON ici plutôt que directement dans l'interface n8n pour maintenir la synchronisation.
