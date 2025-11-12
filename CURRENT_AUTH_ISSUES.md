# État actuel : Problèmes d'authentification persistants

## Date: 2025-11-12

## Problèmes identifiés

### 1. Erreur `_basic_auth_patched` persistante
```
litellm.APIError: APIError: OpenAIException - Completions.create() got an unexpected keyword argument '_basic_auth_patched'
```

**Cause**: L'attribut `_basic_auth_patched` ajouté aux classes OpenAI/OpenAIConfig est probablement lu par litellm et passé comme kwargs à `completion()`.

**Impact**: Tous les appels directs à `litellm.completion()` échouent pour codex endpoint.

### 2. Provider NOT provided pour Claude
```
litellm.BadRequestError: LLM Provider NOT provided. Pass in the LLM provider you are trying to call. You passed model=claude-sonnet-4.5
```

**Cause**: LiteLLM ne reconnaît pas automatiquement le provider depuis le model name.

**Solution**: Ajouter le préfixe provider explicite (`anthropic/` ou `openai/`).

### 3. 401 Authorization Required pour Claude endpoint
```
litellm.AuthenticationError: AnthropicException - <html>
<head><title>401 Authorization Required</title></head>
```

**Cause**: Le format Basic Auth n'est pas correctement transmis au endpoint Claude.

### 4. NO MCP tools were called
Malgré que les tools soient passés au LLM, **aucun tool n'est appelé**.

**Cause possible**: Le LLM ne reçoit pas les tools dans le bon format, ou le fallback à `LLM.call()` est trop rapide.

## Solution recommandée

### Option 1: Abandonner l'appel direct à litellm.completion()

**Revenir à CrewAI's flow normal** et laisser CrewAI gérer l'exécution des tools via `available_functions`.

**Avantages**:
- Pas de problèmes d'auth
- CrewAI gère l'exécution des tools
- Pas besoin de parser tool_calls

**Inconvénients**:
- Les tools MCP ne sont peut-être pas compatibles avec le format `available_functions`

### Option 2: Fix les problèmes d'auth un par un

1. **Retirer complètement `_basic_auth_patched` des classes**
   - Ne pas ajouter d'attributs aux classes OpenAI
   - Gérer l'auth uniquement via les headers HTTP

2. **Ajouter provider prefix pour tous les modèles**
   ```python
   # Pour Claude
   model = "anthropic/claude-sonnet-4-5-20250929"
   
   # Pour GPT
   model = "openai/gpt-5"
   ```

3. **Utiliser custom_llm_provider**
   ```python
   litellm.completion(
       model=provider.model,
       custom_llm_provider="openai",  # Force OpenAI-compatible
       messages=formatted_messages,
       tools=tools,
       api_base=provider.api_base,
       api_key=api_key_str,
   )
   ```

### Option 3: Utiliser CrewAI's MCP DSL (RECOMMANDÉ)

Selon la doc CrewAI, il y a un support natif pour MCP :

```python
agent = Agent(
    role="Full-Featured Agent",
    goal="Use all available tool types",
    backstory="Agent with comprehensive tool access",
    
    # MCP servers directement via DSL
    mcps=[
        "https://mcp.exa.ai/mcp?api_key=key",
        "crewai-amp:research-tools"
    ],
    
    verbose=True,
    max_iter=15
)
```

**TODO**: Vérifier si notre code utilise déjà cette approche ou s'il faut migrer.

## Actions immédiates recommandées

1. ✅ Vérifier si le code utilise le DSL CrewAI MCP (`mcps` field)
2. ⏳ Si non, migrer vers le DSL officiel
3. ⏳ Si oui, vérifier pourquoi les tools ne sont pas appelés
4. ⏳ Simplifier l'authentification (retirer patches inutiles)
5. ⏳ Tester avec un seul endpoint d'abord (copilot)

## Fichiers à modifier

- `crew.py`: Retirer code dupliqué d'auth, utiliser uniquement `llm_auth_init`
- `llm_provider_rotation.py`: Simplifier appel litellm ou revenir à CrewAI flow
- `llm_auth_init.py`: Retirer `setattr(..., "_basic_auth_patched", ...)` si problématique
- `agents/*.py`: Vérifier si `mcps` field est utilisé

## Questions à résoudre

1. Est-ce que notre implémentation MCP est compatible avec CrewAI's DSL ?
2. Pourquoi `_basic_auth_patched` est passé comme kwargs ?
3. Est-ce que `drop_params=True` devrait filtrer `_basic_auth_patched` ?
4. Faut-il vraiment bypasser `LLM.call()` ou peut-on utiliser le flow normal ?
