# SERVDjango_IA_FINAL

# 📚 Projet IA IMPERIATEC

Application complète d’Intelligence Artificielle permettant :
- 📄 L’analyse automatique et le résumé de documents PDF.
- 💬 Une interface de chat IA conversationnelle basée sur les contenus traités.
- 🧠 La reformulation intelligente des réponses générées.

---

## 🧱 Structure du projet

```bash
backend/
├── ia_backend/
│   ├── ask_engine.py
│   ├── services/
│   │   ├── pdf_utils.py
│   │   ├── chat_memory.py
│   │   ├── ollama_gateway.py
│   │   ├── backup_service.py
│   ├── views.py
│   ├── tasks.py

Front-IA-IMPERIATEC/
├── app/
│   ├── chat.tsx
├── assets/
├── package.json
🚀 Démarrage rapide

🔧 Prérequis
Docker & Docker Compose
Node.js & npm
Python 3.10+
Ollama (local ou dans Docker)
🐳 Démarrage backend avec Docker

# 1. Construire les images
docker compose build

# 2. Lancer les services en arrière-plan
docker compose up -d

# 3. Voir les logs du worker Celery
docker compose logs -f worker

# 4. Voir les logs du backend Django
docker compose logs -f web

# 5. Stopper tous les services
docker compose down


💻 Lancer le frontend React Native

# Aller dans le dossier frontend
cd Front-IA-IMPERIATEC

# Installer les dépendances
npm install

# Lancer l'application (Expo)
npx expo start
Assure-toi que l’URL de l’API (API_BASE_URL) est bien définie dans le code React Native (ex. chat.tsx), comme :

const API_BASE_URL = "http://192.168.10.121:8000";
🧠 Fonctionnalités IA

Résumé de PDF par bloc ➡️ synthèse globale
Chat IA personnalisé avec mémoire de session
Reformulation automatique via prompt dédié
Détection de langue et traduction automatique (EN ➡️ FR)
📚 Technologies principales

Composant	Rôle
Django / DRF	Backend API
Celery + Redis	Traitement asynchrone
SQLite / JSON	Cache local rapide
SentenceTransformer	Embedding sémantique
CrossEncoder (MS MARCO)	Re-ranking des résultats
Ollama + LLaMA3	Génération des réponses IA
React Native + Expo	Interface utilisateur mobile
📦 Installation des modèles Ollama

Exemple :

ollama pull mistral
❓ À propos

Ce projet a été conçu pour intégrer l’IA dans une application mobile intuitive, tout en exploitant la puissance des LLMs localement avec une grande efficacité.