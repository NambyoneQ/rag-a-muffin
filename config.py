import os

class Config:
    # Clé secrète pour Flask. À changer pour la production !
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'une_cle_secrete_tres_complexe_et_aleatoire_pour_dev'

    # Configuration de la base de données PostgreSQL
    SQLALCHEMY_DATABASE_URI = 'postgresql://nambadmin:qwan2025admin@localhost:5432/Projet_Test'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Configuration pour l'API locale du LLM de chat (DeepSeek Coder via Jan.ai)
    JAN_AI_API_BASE = "http://localhost:1337/v1"
    JAN_AI_API_KEY = "my-jan-secret-key"

    # Configuration pour l'API locale du modèle d'embeddings (Nomic Embed Text via LM Studio)
    # ASSUREZ-VOUS QUE LA 2ÈME INSTANCE LM STUDIO TOURNE SUR LE PORT 1234
    LMSTUDIO_EMBEDDING_API_BASE = "http://localhost:1234/v1"
    LMSTUDIO_API_KEY = "lm-studio" # <--- ASSUREZ-VOUS QUE CETTE LIGNE EST PRÉSENTE ET CORRECTE !

    # Chemin vers le dossier des documents pour le RAG
    KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kb_documents')

    # Nom du dossier pour le Vector Store (ChromaDB)
    CHROMA_PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_db')