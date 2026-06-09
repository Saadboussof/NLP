# ============================================================
#  config.py
#  Partie 2 : Configuration du système RAG
# ============================================================
#  Ce fichier contient les paramètres de configuration
#  du système RAG (Retrieval-Augmented Generation).
# ============================================================

import os

# --- Chemins ---
# Répertoire de base du projet (partie2/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")

# --- Modèle d'embeddings ---
# all-MiniLM-L6-v2 : rapide, léger (~80 Mo), dimension 384
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# --- Paramètres de chunking ---
# Taille de chaque morceau de texte (en caractères)
CHUNK_SIZE = 500
# Chevauchement entre les morceaux (pour ne pas couper les idées)
CHUNK_OVERLAP = 50

# --- Paramètres de recherche ---
# Nombre de documents pertinents à récupérer
TOP_K = 3
# Seuil de similarité minimum (0.0 à 1.0)
SIMILARITY_THRESHOLD = 0.3

# --- Modèle de génération (LLM local) ---
# Flan-T5 : modèle de Google, bon pour le Q&A, fonctionne sur CPU
GENERATOR_MODEL = "google/flan-t5-base"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
