**Accroche :**
J'ai dû attaquer ma propre app de nutrition en "Man-in-the-Middle" pour que mon IA puisse me faire à manger. 🥗🤖

**Corps du post :**
Automatiser ma nutrition sportive était un défi technique passionnant. Je voulais que mes agents IA (Claude) lisent mes entraînements sur Strava/Intervals.icu et planifient mes repas sur Hexis.

Problème : Hexis n'a pas d'API publique pour créer des repas.

La solution a impliqué un mélange de cybersécurité et d'architecture agentique moderne.

👉 **Le Hack (Reverse Engineering)** :
J'ai intercepté le trafic de l'application mobile (MITM) pour comprendre comment elle "parlait" au serveur. J'y ai découvert un paramètre caché (`refCode` en Base64) indispensable pour valider chaque ingrédient. J'ai ensuite encapsulé cette logique dans un serveur MCP (Model Context Protocol) maison.

👉 **L'Architecture Agentique (Supervisor/Executor)** :
Pour éviter les hallucinations des modèles "Thinking" (qui sont brillants pour réfléchir mais mauvais pour utiliser des outils), j'ai séparé les responsabilités :
1️⃣ **Superviseur** (Claude Opus) : Conçoit la stratégie nutritionnelle (Pure réflexion).
2️⃣ **Exécuteur** (Modèle rapide) : Utilise les outils API pour trouver les aliments (Pure exécution).
3️⃣ **Réviseur** : Valide les macros via des calculs Python stricts.

👉 **L'Orchestration** :
Tout est piloté par un workflow **n8n** qui gère l'asynchronisme et notifie le plan final sur Telegram chaque dimanche soir.

J'ai détaillé toute l'architecture technique et le fonctionnement du hack dans mon dernier article de blog.

Lien en premier commentaire 👇

#AI #AgenticWorkflow #ReverseEngineering #n8n #Automation #ClaudeAI #Python #MCP
