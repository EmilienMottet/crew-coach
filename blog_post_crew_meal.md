---
title: "Du MITM au Design Agentique : Automatiser sa nutrition sportive avec Claude, n8n et Hexis"
date: 2025-12-28T10:00:00+01:00
draft: false
tags: ["AI Agents", "Reverse Engineering", "MCP", "n8n", "Python", "Automation", "LLM Optimization"]
categories: ["Engineering"]
description: "Comment j'ai contourné une API fermée via une attaque Man-in-the-Middle pour construire une flotte d'agents IA autonomes, optimisés et sécurisés."
---

L'automatisation de la nutrition sportive est le Graal de beaucoup d'athlètes amateurs. On veut la précision d'un nutritionniste sans la charge mentale.

Mon objectif était simple : synchroniser mes entraînements (Intervals.icu) avec mes besoins nutritionnels (Hexis) et générer des plans de repas automatiques. Le problème ? **Hexis n'a pas d'API publique pour la création de repas.**

Voici comment j'ai combiné **Reverse Engineering**, **Man-in-the-Middle**, **Design Agentique Avancé**, **Meta-MCP** et **n8n** pour construire un système autonome et économique.

## 1. Le Hack : Reverse Engineering & Man-in-the-Middle

Puisque la porte d'entrée était fermée, je suis passé par la fenêtre. Pour automatiser l'enregistrement des repas dans Hexis, j'ai dû comprendre comment leur application mobile communiquait avec leur backend.

### L'approche MITM (Man-in-the-Middle)
J'ai configuré un proxy pour intercepter le trafic HTTPS entre mon téléphone et les serveurs d'Hexis. En observant les requêtes lors de la création d'un repas dans l'app, j'ai découvert une structure de données complexe.

### La découverte du `refCode`
L'API ne se contente pas d'un ID d'aliment. Elle utilise une base de données tierce (Passio) et exige un paramètre spécifique et non documenté : le `refCode`.

C'est une chaîne encodée en Base64 renvoyée lors de la recherche d'aliment, mais qui doit impérativement être renvoyée telle quelle lors de l'enregistrement du repas. Sans ce `refCode`, l'API renvoie une erreur 400 muette.

```json
// Ce que l'API attend réellement (découvert via MITM)
{
  "foodId": "12345",
  "quantity": 150,
  "unit": "g",
  "metaData": {
    "refCode": "Base64StringHiddenInSearchResult..." // Le sésame !
  }
}
```

J'ai encapsulé cette logique dans un **serveur MCP (Model Context Protocol)** personnalisé. Cet outil agit comme une couche d'abstraction : il gère la recherche, extrait le `refCode` silencieusement, et formate la requête pour Hexis, rendant l'opération transparente pour mes agents IA.

## 2. Architecture Agentique : Le Pattern Supervisor/Executor/Reviewer

Une fois l'accès aux données résolu, il fallait un "cerveau". J'utilise **CrewAI** avec des modèles LLM avancés (Claude Opus/Sonnet).

Cependant, j'ai rencontré un problème majeur : les modèles "Thinking" (qui raisonnent longuement) sont excellents pour la stratégie nutritionnelle, mais **hallucinent souvent les appels d'outils** (tool calling) ou échouent à respecter les formats JSON stricts.

La solution ? Une architecture en trois tiers :

### Tier 1 : Le Superviseur (Cerveau)
*   **Rôle** : Stratégie pure. Il conçoit le plan de repas idéal en fonction des macros cibles.
*   **Modèle** : Modèle "Thinking" complexe (ex: Claude Opus).
*   **Contrainte** : Aucun accès aux outils (pour éviter les hallucinations).

### Tier 2 : L'Exécuteur (Bras)
*   **Rôle** : Interaction API. Il prend le plan du superviseur et cherche les ingrédients réels dans la base de données via MCP.
*   **Modèle** : Modèle rapide et "bête" (ex: GPT-4o-mini ou Haiku).
*   **Spécificité** : `has_tools=True`. Il ne réfléchit pas, il exécute.

### Tier 3 : Le Réviseur (Contrôle Qualité)
*   **Rôle** : Vérification et Assemblage. Il recalcule les macros exactes avec du code Python (plus fiable que le calcul mental d'un LLM) et valide la cohérence.
*   **Modèle** : Intermédiaire.

Tout ce petit monde communique via des schémas **Pydantic** stricts, garantissant que la sortie de l'un correspond parfaitement à l'entrée de l'autre.

## 3. Infrastructure MCP : Sécurité et Modularité avec Meta-MCP

Gérer une flotte d'agents qui ont besoin d'accès variés (Strava, Intervals.icu, Hexis, Météo...) peut vite devenir un cauchemar de sécurité. Je ne voulais pas que chaque agent ait accès à tous les outils.

J'utilise **Meta-MCP**, une couche d'abstraction qui me permet de :

1.  **Regrouper les outils par domaine** : Un serveur MCP pour le sport, un pour la nutrition, un pour la météo.
2.  **Sécuriser les accès** : Chaque agent ne reçoit que le sous-ensemble d'outils dont il a besoin via une clé API unique.
3.  **Standardiser l'interface** : Peu importe le service sous-jacent, mes agents voient une interface d'outils unifiée.

C'est cette couche qui permet à mon "Exécuteur Hexis" d'accéder aux outils de nutrition sans risquer de supprimer une activité sur Strava par erreur.

## 4. Optimisation des Coûts : L'IA à prix malin

Lancer des agents autonomes 24/7 a un coût, surtout avec des modèles performants. Pour éviter une facture OpenAI/Anthropic astronomique, j'ai mis en place une stratégie agressive d'optimisation :

### Reverse Proxies & Modèles Alternatifs
Au lieu de taper directement chez les géants de la tech, j'utilise des **Reverse Proxies** compatibles OpenAI.
*   **Solutions testées** : J'ai commencé avec [copilot-api](https://github.com/ericc-ch/copilot-api), puis **ccproxy**, pour finir avec [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI).
*   **Modèles exotiques** : Mention honorable à **DeepSeek** et **GLM-4.6 (via z.ai)** qui offrent des performances proches de GPT-4 pour une fraction du prix.
*   **Astuces** : L'utilisation de modèles "Coder" (comme Qwen-Coder) pour des tâches de structure JSON est souvent plus efficace et moins chère que les modèles généralistes.

### Rotation Automatique
Mon système gère une **cascade de modèles**. Si le modèle "Premium" (Claude 3.5 Sonnet) atteint son quota ou rate-limit, le système bascule automatiquement sur un modèle "Eco" (GPT-4o-mini ou GLM-4) pour terminer la tâche. C'est transparent pour l'utilisateur et ça sauve la production.

## 5. L'Orchestration avec n8n

Avoir des agents intelligents ne suffit pas, il faut les faire vivre dans le temps. C'est là qu'intervient **n8n**.

J'ai mis en place un workflow (`meal-planning-weekly`) qui tourne chaque dimanche soir :

1.  **Récupération de la charge d'entraînement** : Le workflow interroge Intervals.icu pour connaître mes séances de la semaine à venir.
2.  **Optimisation du planning** : Un script JS réorganise mes créneaux (ex: "Si grosse sortie vélo le samedi -> Repas riche en glucides le vendredi soir").
3.  **Appel Asynchrone vers l'IA** : n8n déclenche mon script Python CrewAI via un webhook HTTP.
4.  **Pattern Callback** : Comme la génération de repas prend du temps (plusieurs minutes de réflexion pour les agents), n8n ne reste pas bloqué. Il attend un "ping" de retour (callback) une fois que les agents ont fini leur travail.
5.  **Distribution** : Le résultat final (liste de courses + menu) est envoyé directement sur Telegram.

## Conclusion

Ce projet montre que les limitations des API fermées ne sont que temporaires face à un peu de reverse engineering.

En combinant la puissance brute d'analyse des **LLMs "Thinking"**, la fiabilité d'exécution des **serveurs MCP**, et l'orchestration de **n8n**, on peut créer des systèmes véritablement autonomes qui ont un impact réel sur le quotidien.

Le code est disponible (partiellement) sur mon GitHub. La prochaine étape ? Automatiser la commande des courses via l'API d'un drive... mais ça, c'est une autre histoire de reverse engineering.

👉 [Lien vers le repository GitHub du projet](https://github.com/EmilienMottet/crew-coach)
