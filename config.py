# config.py
import os

class Config:
    # Clé secrète pour Flask.
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'default_fallback_secret_key_if_env_not_set'

    # Paramètres PostgreSQL
    DB_USER = os.environ.get('DB_USER')
    DB_PASSWORD = os.environ.get('DB_PASSWORD')
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_PORT = os.environ.get('DB_PORT', '5432')
    DB_NAME = os.environ.get('DB_NAME')

    SQLALCHEMY_DATABASE_URI = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- Configurations pour LM Studio UNIFIÉ (remplace Jan.ai et LM Studio Embeddings séparés) ---
    # LM Studio sert les deux APIs (chat et embeddings) sur le même BASE_URL
    LMSTUDIO_UNIFIED_API_BASE = "http://localhost:1234/v1"
    LMSTUDIO_API_KEY = os.environ.get('LMSTUDIO_API_KEY')

    # NOUVELLE LIGNE AJOUTÉE OU CORRIGÉE POUR LMSTUDIO_CHAT_MODEL
    # Assurez-vous que le nom du modèle correspond à celui que vous avez chargé dans LM Studio.
    # Si vous utilisez un modèle plus léger (ex: Phi-3-mini-4k-instruct-gguf), remplacez la valeur par défaut.
    LMSTUDIO_CHAT_MODEL = os.environ.get('LMSTUDIO_CHAT_MODEL', 'Llama-3.1-8B-UltraLong-4M-Instruct-Q4_K_M')

    # Chemins vers les dossiers RAG
    KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kb_documents')
    CODE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'codebase')
    CHROMA_PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_db')

    # NOUVELLES CONFIGURATIONS POUR RAG SERVICE (pour éviter les erreurs Pylance)
    # Chemins spécifiques pour les bases de données Chroma (à l'intérieur du dossier persist_directory)
    CHROMA_PATH_KB = os.path.join(CHROMA_PERSIST_DIRECTORY, 'kb')
    CHROMA_PATH_CODEBASE = os.path.join(CHROMA_PERSIST_DIRECTORY, 'codebase')
    
    # Chemin pour le cache de traitement (pour stocker les hash des fichiers)
    PROCESSING_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.rag_cache') 

    # Paramètres de chunking par défaut
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Paramètres de récupération (nombre de documents à récupérer)
    TOP_K_RETRIEVAL_KB = 5
    TOP_K_RETRIEVAL_CODEBASE = 7 # Un peu plus élevé pour le code pourrait être utile