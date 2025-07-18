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
    LMSTUDIO_UNIFIED_API_BASE = "http://localhost:1234/v1" # Base d'URL unique
    LMSTUDIO_API_KEY = os.environ.get('LMSTUDIO_API_KEY') # La clé API (la même pour les deux)

    # Chemins vers les dossiers RAG
    KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kb_documents')
    CHROMA_PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_db')