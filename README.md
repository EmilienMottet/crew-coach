# Strava Activity Description Crew 🏃‍♂️

Réseau d'agents CrewAI pour générer automatiquement des titres et descriptions d'activités Strava basés sur les données d'Intervals.icu.

## 📋 Fonctionnalités

### Agents

1. **Activity Description Writer** 📝
   - Analyse les données d'Intervals.icu
   - Génère des titres accrocheurs (max 50 caractères)
   - Rédige des descriptions informatives (max 500 caractères)
   - Identifie le type d'entraînement (tempo, intervalles, sortie facile, etc.)

2. **Privacy & Compliance Officer** 🔒
   - Détecte les informations sensibles (noms, adresses, etc.)
   - Vérifie les horaires de travail
   - Recommande le niveau de confidentialité (public/privé)
   - Propose des versions nettoyées du contenu

3. **Sports Content Translator** 🌐 *(Optionnel)*
   - Traduit les titres et descriptions dans la langue cible
   - Préserve les emojis et la mise en forme
   - Adapte la terminologie sportive de manière appropriée
   - Respecte les limites de caractères (50 pour le titre, 500 pour la description)
   - S'active via la variable d'environnement `TRANSLATION_ENABLED=true`

### Règles de confidentialité

- **Horaires de travail** : 08:30-12:00 et 14:00-17:00
- Les activités pendant ces horaires sont automatiquement mises en **privé**
- Détection d'informations sensibles :
  - Noms complets (seuls les prénoms sont acceptés)
  - Adresses exactes
  - Numéros de téléphone, emails
  - Informations médicales détaillées

## 🚀 Installation

1. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

2. **Configurer l'environnement** :
```bash
cp .env.example .env
# Le fichier .env est déjà configuré avec vos paramètres
```

### Authentication Configuration

The crew supports two authentication methods:

1. **Basic Authentication** (recommended for external providers):
   ```bash
   # Generate base64 token from username:password
   echo -n "username:password" | base64
   
   # Add to .env file
   OPENAI_API_AUTH_TOKEN=your_base64_token_here
   ```

2. **API Key**:
   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```

If `OPENAI_API_AUTH_TOKEN` is set, it takes precedence and will be used as `Authorization: Basic <token>`.

### Example curl request
```bash
curl https://ghcopilot.emottet.com/v1/chat/completions \
  -H "Authorization: Basic b2NvOjc2d3VudFk4Q3QzR2szRFU=" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

## Configuration

### Variables d'environnement

```bash
# Endpoint LLM local (votre serveur à 192.168.0.141:8181)
OPENAI_API_BASE=http://192.168.0.141:8181/v1
OPENAI_API_KEY=dummy-key-not-needed-for-local
OPENAI_MODEL_NAME=gpt-5-mini

# Serveur MCP
MCP_SERVER_URL=https://mcp.emottet.com/metamcp/stravaDescriptionAgent/mcp?api_key=...

# Horaires de travail
WORK_START_MORNING=08:30
WORK_END_MORNING=12:00
WORK_START_AFTERNOON=14:00
WORK_END_AFTERNOON=17:00

# Traduction (optionnel)
TRANSLATION_ENABLED=false
TRANSLATION_TARGET_LANGUAGE=English
TRANSLATION_SOURCE_LANGUAGE=French
```

## 🐳 Déploiement Docker & GitHub Packages

### Construction et exécution locales

Copier le fichier d'environnement :

```bash
cp .env.example .env
```

Construire l'image containerisée :

```bash
docker compose build
```

Traiter une activité en injectant le JSON sur l'entrée standard :

```bash
docker compose run --rm crew < input.json
```

> ℹ️ Le service `crew` charge automatiquement les variables définies dans `.env`. Le conteneur attend toujours les données Strava sur `stdin`, comme en exécution locale.

### Publication automatique sur le registre privé GitHub

- Un workflow GitHub Actions (`.github/workflows/docker-publish.yml`) construit l'image Docker et la pousse dans le registre privé GitHub Container Registry (`ghcr.io/emilienmottet/crew-coach`).
- Le workflow se déclenche à chaque `push` sur `main` (et peut être lancé manuellement). Aucune configuration supplémentaire n'est nécessaire : le token `GITHUB_TOKEN` intégré fournit les droits `packages:write`.
- Les images sont taguées avec `latest` et le SHA du commit (`ghcr.io/emilienmottet/crew-coach:<sha>`). Vous pouvez ensuite les consommer via `docker pull ghcr.io/emilienmottet/crew-coach:latest`.

### Modèles LLM disponibles

Le système utilise votre endpoint local avec le modèle `gpt-5-mini` par défaut. Le modèle est configuré pour être compatible avec LiteLLM (utilisé par CrewAI) en utilisant le préfixe `openai/`.

**Important** : Assurez-vous que :

- Votre endpoint local (`OPENAI_API_BASE`) est compatible OpenAI API
- Le modèle (`OPENAI_MODEL_NAME`) correspond exactement au nom exposé par votre serveur
- Pour LM Studio : utilisez le nom du modèle affiché dans l'interface
- Pour Ollama : activez l'API compatible OpenAI sur le port 11434

Autres modèles disponibles sur votre endpoint :

- `gpt-4.1`
- `gpt-5`
- `gpt-4o-mini`
- `claude-sonnet-4.5`
- `gemini-2.5-pro`

## 📖 Utilisation

### Ligne de commande

```bash
# Traiter une activité depuis stdin
cat input.json | python crew.py

# Exemple avec l'input fourni
python crew.py < input.json
```

### Intégration n8n

1. **Nœud Execute Command** :

  ```bash
  Command: python /home/emottet/Documents/Perso/Sport/crew/crew.py
  ```

1. **Workflow suggéré** :

  ```text
  Webhook Strava → Execute Command (Python crew.py) → Parse JSON → Update Strava
  ```

1. **Input** : Passer les données du webhook Strava via stdin

1. **Output** : JSON sur stdout avec le résultat

## 📥 Format d'entrée

Le script attend des données au format webhook Strava :

```json
[
  {
    "object_type": "activity",
    "object_id": 16284886069,
    "aspect_type": "create",
    "object_data": {
      "id": 16284886069,
      "name": "Lunch Run",
      "distance": 12337,
      "moving_time": 3601,
      "type": "Run",
      "start_date_local": "2025-10-27T11:54:41Z",
      ...
    }
  }
]
```

## 📤 Format de sortie

```json
{
  "activity_id": 16284886069,
  "title": "🏃 12.3K Tempo Run - Strong Effort",
  "description": "Solid tempo run focusing on pace control...",
  "should_be_private": false,
  "privacy_check": {
    "approved": true,
    "during_work_hours": false,
    "issues": [],
    "reasoning": "No privacy issues. Activity outside work hours."
  },
  "workout_analysis": {
    "type": "Tempo Run",
    "metrics": {
      "average_pace": "4:53 /km",
      "average_hr": "141 bpm",
      "max_hr": "169 bpm"
    }
  }
}
```

## 🔧 Architecture

```
crew/
├── agents/
│   ├── description_agent.py    # Génère titre et description
│   ├── privacy_agent.py        # Vérifie confidentialité
│   └── translation_agent.py    # Traduit le contenu (optionnel)
├── tasks/
│   ├── description_task.py     # Tâche de génération
│   ├── privacy_task.py         # Tâche de vérification
│   └── translation_task.py     # Tâche de traduction (optionnel)
├── tools/                     # Package conservé (plus de helpers legacy)
│   └── __init__.py
├── crew.py                     # Point d'entrée principal
├── requirements.txt
├── .env.example
├── input.json
└── README.md
```

## 🛠️ Outils MCP disponibles

### Intervals.icu

- `IntervalsIcu__get_activity_details` : Détails complets d'une activité
- `IntervalsIcu__get_activity_intervals` : Données des intervalles/segments
- `IntervalsIcu__get_activities` : Liste des activités récentes

> ℹ️ Ces outils sont exposés automatiquement à l'agent de description via le champ `mcps` de CrewAI. Il suffit de définir `MCP_SERVER_URL` (ou plusieurs URL séparées par des virgules) dans l'environnement. Par défaut, l'auto-découverte est utilisée. Définissez `INTERVALS_MCP_TOOL_NAMES` pour verrouiller une liste spécifique d'outils si nécessaire.

### Autres sources (via MCP)

- Strava : Détails activités, segments, zones
- Hexis.live : Données nutritionnelles
- Spotify : Playlists d'entraînement
- OpenWeatherMap : Conditions météo

## 🔍 Exemple de fonctionnement

Pour l'activité dans `input.json` (course à 11:54:41) :

1. **Analyse** : Le système récupère les données d'Intervals.icu
2. **Génération** :
   - Titre : "🏃 12.3K Lunch Run - Intervals"
   - Description : Décrit la structure (échauffement, intervalles, récup)
3. **Vérification** :
   - ⚠️ Activité à 11:54 = pendant les heures de travail (08:30-12:00)
   - ✅ Pas d'informations sensibles détectées
   - 🔒 **Recommandation : PRIVÉ**
4. **Traduction** *(si activée)* :
   - Traduit le titre et la description vers la langue cible
   - Préserve les emojis et le formatage
   - Adapte la terminologie sportive

### Workflow complet

```text
Strava Activity Created
  ↓
Step 1: Generate Description (Description Agent)
  → Fetch data from Intervals.icu
  → Analyze workout structure
  → Generate title + description
  ↓
Step 2: Privacy Check (Privacy Agent)
  → Detect sensitive information
  → Check work hours compliance
  → Sanitize if needed
  ↓
Step 3: Translation (Translation Agent) [Optional]
  → Translate title to target language
  → Translate description to target language
  → Preserve emojis and formatting
  ↓
Final Output → Update Strava
```

## 🐛 Dépannage

### Le serveur MCP ne répond pas
 
```bash
# Vérifier la connectivité
curl "https://mcp.emottet.com/metamcp/stravaDescriptionAgent/mcp?api_key=..."
```

### Erreur de parsing JSON

- Vérifier que l'input est un JSON valide
- S'assurer que `object_data` est présent

### Activité toujours en privé

- Vérifier les horaires dans `.env`
- Vérifier le fuseau horaire de `start_date_local`

### LLM ne répond pas

```bash
# Tester l'endpoint local
curl http://192.168.0.141:8181/v1/models

# Vérifier les logs
python crew.py < input.json 2> logs.txt
```

### Erreur "LLM Provider NOT provided"

Cette erreur survient lorsque le modèle LLM n'est pas correctement configuré. **Solution** :

1. Vérifiez que `OPENAI_API_BASE` et `OPENAI_MODEL_NAME` sont définis dans `.env`
2. Le modèle doit correspondre exactement au nom exposé par votre serveur local
3. CrewAI utilise LiteLLM qui nécessite le préfixe `openai/` pour les endpoints compatibles OpenAI
4. Testez votre endpoint :

```bash
curl -X POST http://192.168.0.141:8181/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy-key" \
  -d '{
    "model": "gpt-5-mini",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Si cela fonctionne, le problème est résolu dans la version actuelle du code.

## 📚 Documentation CrewAI

- [CrewAI Documentation](https://docs.crewai.com/)
- [Tools Creation Guide](https://docs.crewai.com/core-concepts/Tools/)
- [Agents & Tasks](https://docs.crewai.com/core-concepts/Agents/)

## 🤝 Support

Pour toute question ou problème, vérifiez :

1. Les logs stderr pour les détails d'exécution
2. La connectivité au serveur MCP
3. Les credentials Intervals.icu dans le serveur MCP
4. L'endpoint LLM local

## 📝 Licence

Usage privé uniquement.
