import os

class Config:
    # Récupère la clé secrète depuis les variables d'environnement (chargées par python-dotenv)
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'default_fallback_secret_key_if_env_not_set'

    # Construction de l'URI de la base de données PostgreSQL à partir des variables d'environnement
    DB_USER = os.environ.get('DB_USER')
    DB_PASSWORD = os.environ.get('DB_PASSWORD')
    DB_HOST = os.environ.get('DB_HOST', 'localhost') # Valeur par défaut 'localhost'
    DB_PORT = os.environ.get('DB_PORT', '5432')     # Valeur par défaut '5432'
    DB_NAME = os.environ.get('DB_NAME')

    SQLALCHEMY_DATABASE_URI = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Récupère la clé API de Jan.ai depuis les variables d'environnement
    JAN_AI_API_BASE = "http://localhost:1337/v1"
    JAN_AI_API_KEY = os.environ.get('JAN_AI_API_KEY')

    # Récupère la clé API de LM Studio depuis les variables d'environnement
    LMSTUDIO_EMBEDDING_API_BASE = "http://localhost:1234/v1"
    LMSTUDIO_API_KEY = os.environ.get('LMSTUDIO_API_KEY')

    # Chemins vers les dossiers RAG (non sensibles, peuvent rester ici ou aussi dans .env)
    KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kb_documents')
    CHROMA_PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_db')