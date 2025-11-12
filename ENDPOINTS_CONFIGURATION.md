# Configuration des Endpoints - Résumé

## 📊 Modèles Disponibles

### Endpoint `/copilot/v1`
**Support**: ✅ Function Calling | ✅ System Prompts

Modèles principaux:
- `gpt-5` - GPT-5 (400K context, 128K output)
- `gpt-5-mini` - GPT-5 mini (264K context, 64K output)
- `gpt-5-codex` - GPT-5 Codex Preview (400K context, 128K output)
- `gpt-4.1` - GPT-4.1 (128K context, 16K output)
- `gpt-4o` - GPT-4o (128K context, 16K output)
- `claude-sonnet-4.5` - Claude Sonnet 4.5 (144K context, 16K output)
- `claude-haiku-4.5` - Claude Haiku 4.5 (144K context, 16K output)
- `claude-sonnet-4` - Claude Sonnet 4 (216K context, 16K output)
- `claude-3.5-sonnet` - Claude 3.5 Sonnet (90K context, 8K output)
- `gemini-2.5-pro` - Gemini 2.5 Pro (128K context, 64K output)
- `grok-code-fast-1` - Grok Code Fast 1 (128K context, 64K output)

### Endpoint `/claude/v1`
**Support**: ✅ Function Calling | ✅ System Prompts

Modèles Claude uniquement:
- `claude-sonnet-4-5-20250929` - Claude Sonnet 4.5
- `claude-haiku-4-5-20251001` - Claude Haiku 4.5
- `claude-opus-4-1-20250805` - Claude Opus 4.1
- `claude-opus-4-20250514` - Claude Opus 4
- `claude-sonnet-4-20250514` - Claude Sonnet 4
- `claude-3-7-sonnet-20250219` - Claude 3.7 Sonnet
- `claude-3-5-sonnet-20241022` - Claude 3.5 Sonnet
- `claude-3-5-haiku-20241022` - Claude 3.5 Haiku
- Plus anciens modèles Claude 3

### Endpoint `/codex/v1`
**Support**: ✅ Function Calling | ⚠️ System Prompts désactivés (mergés dans message user)

> **Note**: Le codex endpoint supporte bien les tools/MCP mais requiert que les system prompts
> soient retirés et fusionnés dans le message utilisateur. C'est une optimisation spécifique
> à cet endpoint pour la génération de code.

Tous les modèles de `/copilot/v1` sont disponibles.

## ✅ Compatibilité avec CrewAI/MCP

### Résumé des Tests
Tous les endpoints ont été testés avec `test_function_calling_endpoints.py`:

| Endpoint | Model | Function Calling | Result |
|----------|-------|------------------|--------|
| `/copilot/v1` | gpt-5-mini | ✅ | Tool call detected |
| `/copilot/v1` | claude-sonnet-4.5 | ✅ | Tool call detected |
| `/copilot/v1` | claude-haiku-4.5 | ✅ | Tool call detected |
| `/codex/v1` | gpt-5 | ✅ | Tool call detected |
| `/codex/v1` | claude-sonnet-4.5 | ✅ | Tool call detected |

### Configuration Agents CrewAI

**Agents AVEC outils MCP** (Description, Music):
```bash
# Recommandé: Copilot (plus de flexibilité)
OPENAI_DESCRIPTION_API_BASE=https://ccproxy.emottet.com/copilot/v1
OPENAI_DESCRIPTION_MODEL_NAME=claude-sonnet-4.5

OPENAI_MUSIC_API_BASE=https://ccproxy.emottet.com/copilot/v1
OPENAI_MUSIC_MODEL_NAME=claude-haiku-4.5

# Alternative: Codex (fonctionne aussi)
# OPENAI_MUSIC_API_BASE=https://ccproxy.emottet.com/codex/v1
# OPENAI_MUSIC_MODEL_NAME=gpt-5
```

**Agents SANS outils** (Privacy, Translation):
```bash
# Tous les endpoints fonctionnent
OPENAI_PRIVACY_API_BASE=https://ccproxy.emottet.com/copilot/v1
OPENAI_PRIVACY_MODEL_NAME=gpt-5-mini

OPENAI_TRANSLATION_API_BASE=https://ccproxy.emottet.com/copilot/v1
OPENAI_TRANSLATION_MODEL_NAME=gpt-5-mini
```

## 🔧 Changements Effectués

### 1. Suppression des restrictions incorrectes
**Fichier**: `llm_provider_rotation.py`

```python
# AVANT (❌ FAUX)
TOOL_FREE_ENDPOINT_HINTS = ("codex",)  # Bloquait codex pour tools
TOOL_FREE_MODEL_HINTS = ("gpt-5",)     # Bloquait gpt-5 pour tools

# APRÈS (✅ CORRECT)
TOOL_FREE_ENDPOINT_HINTS: tuple = ()   # Aucune restriction
TOOL_FREE_MODEL_HINTS: tuple = ()      # Tous les modèles supportent tools
```

### 2. Suppression de la validation dans crew.py
La validation qui rejetait codex pour les agents avec outils a été supprimée car
elle était basée sur une fausse hypothèse.

### 3. Conservation du mode sans system prompt pour codex
```python
# Codex nécessite toujours que les system prompts soient désactivés
PROMPTLESS_ENDPOINT_HINTS = ("codex",)
# Mais cela n'empêche PAS l'utilisation des tools !
```

## 📝 Tests Disponibles

```bash
# Tester la compatibilité function calling
python test_function_calling_endpoints.py

# Tester Music Agent avec codex
python test_codex_music_tools.py

# Tester la gestion des system prompts
python -c "from llm_provider_rotation import _requires_promptless_mode; 
print(_requires_promptless_mode('https://ccproxy.emottet.com/codex/v1', 'gpt-5'))"
```

## 🎯 Résultat Final

✅ **Tous vos endpoints supportent les tools/MCP pour CrewAI**  
✅ **Le Music Agent peut maintenant appeler les outils Spotify**  
✅ **Plus de flexibilité dans le choix des providers**  
✅ **Rotation de providers fonctionne correctement**

## ⚠️ Points d'Attention

1. **Rate Limits**: L'endpoint `/claude/v1` peut avoir des rate limits plus stricts
2. **System Prompts**: Codex les désactive automatiquement (pas un problème pour les tools)
3. **Coûts**: Vérifier les tarifs de chaque endpoint/modèle pour optimisation

## 🔗 Documentation

- `FUNCTION_CALLING_FIX.md` - Détails de la correction effectuée
- `test_function_calling_endpoints.py` - Tests de compatibilité endpoints
- `test_codex_music_tools.py` - Tests Music Agent + codex
