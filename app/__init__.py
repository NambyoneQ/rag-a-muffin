# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import logging
import os
from config import Config # Importation de la classe Config

# Initialisation de l'instance Flask de l'application (Globale au module)
app = Flask(__name__)

# Configure l'application avec les paramètres de config.py
app.config.from_object(Config)
app.logger.info("Configuration de l'application chargée.")

# Initialisation de l'extension SQLAlchemy (Globale au module)
db = SQLAlchemy() # db est une instance globale de SQLAlchemy

# Lie l'instance globale 'db' à l'application 'app'
db.init_app(app) 
app.logger.info("Base de données SQLAlchemy liée à l'application.")

# Configure le logger de l'application Flask
app.logger.setLevel(logging.INFO)
if not app.logger.handlers:
    handler = logging.StreamHandler()
    app.logger.addHandler(handler)

# Dictionnaire pour stocker les instances des services initialisés, accessibles globalement via app.extensions
app.extensions = {
    "llm_service": None,
    "rag_service": None, # Va stocker l'instance de RAGService avec ses retrievers
    "conversation_service": None,
    "available_folder_names": [] # Pour stocker les noms de dossiers disponibles pour le RAG
}

# Fonction pour initialiser tous les services de l'application au démarrage
def initialize_services_on_startup():
    from app import models # Importation de models ici pour éviter les circularités
    try:
        # S'assurer d'un contexte applicatif pour db.create_all()
        with app.app_context(): 
            db.create_all()
        app.logger.info("Tables de la base de données créées ou déjà existantes.")
    except Exception as e:
        app.logger.error(f"Erreur lors de la création des tables de la base de données: {e}")
        print(f"PRINT ERROR: Erreur lors de la création des tables de la base de données: {e}")
        print("Vérifiez que le serveur PostgreSQL est en cours d'exécution et que l'utilisateur 'dev_user' a les droits 'Create databases' sur la base de données 'mon_premier_rag_db'.")

    # Scan des dossiers de base de connaissances et de code pour le filtrage dynamique
    available_folder_names = set()
    kb_dir = app.config['KNOWLEDGE_BASE_DIR']
    code_dir = app.config['CODE_BASE_DIR']

    if os.path.exists(kb_dir):
        for item in os.listdir(kb_dir):
            if os.path.isdir(os.path.join(kb_dir, item)):
                available_folder_names.add(item)
    
    if os.path.exists(code_dir):
        for item in os.listdir(code_dir):
            if os.path.isdir(os.path.join(code_dir, item)):
                available_folder_names.add(item) 

    app.extensions['available_folder_names'] = sorted(list(available_folder_names))
    app.logger.info(f"Available top-level folders for RAG filtering: {app.extensions['available_folder_names']}")

    # 2. Initialiser le LLM Service
    from app.services import llm_service as _llm_service_module
    _llm_service_module.initialize_llms()
    app.extensions["llm_service"] = {
        "chat_llm": _llm_service_module.get_chat_llm(),
        "embeddings_llm": _llm_service_module.get_embeddings_llm()
    }
    app.logger.info("LLMs principal et d'embeddings initialisés et attachés à l'application.")

    # 3. Initialiser le RAG Service (CORRECTION ICI)
    from app.services.rag_service import RAGService # Importe directement la classe RAGService
    try:
        # Créer l'instance du RAGService. Son __init__ va charger/créer les ChromaDBs
        rag_service_instance = RAGService()
        
        # Lancer la mise à jour des vector stores. Cette méthode gérera l'ingestion, la mise à jour et la suppression.
        # Elle est appelée DANS un app_context car elle interagit avec db.session pour DocumentStatus.
        with app.app_context(): 
            rag_service_instance.update_vector_store()

        # Stocker les instances ChromaDB brutes, pas les retrievers pré-configurés
        # Le retriever sera créé dynamiquement dans chat_routes.py
        app.extensions["rag_service"] = {
            "kb_db_instance": rag_service_instance.get_kb_db_instance(), 
            "codebase_db_instance": rag_service_instance.get_codebase_db_instance()
        }
        app.logger.info("RAG Service (ChromaDB) initialisé et attaché à l'application avec des instances DB.")
    except RuntimeError as e:
        app.logger.warning(f"Impossible d'initialiser le RAG service : {e}. Le RAG sera désactivé.")
        app.extensions["rag_service"] = {
            "kb_db_instance": None,
            "codebase_db_instance": None
        }
    except Exception as e:
        app.logger.error(f"Erreur inattendue lors de l'initialisation du RAG service : {e}. Le RAG sera désactivé.")
        print(f"PRINT ERROR: Erreur inattendue lors de l'initialisation du RAG service : {e}")
        app.extensions["rag_service"] = {
            "kb_db_instance": None,
            "codebase_db_instance": None
        }

    # 4. Initialiser le Conversation Service
    from app.services import conversation_service as _conv_service_module
    app.extensions["conversation_service"] = {
        "load_history": _conv_service_module.load_conversation_history,
        "save_message": _conv_service_module.save_message
    }
    app.logger.info("Conversation Service initialisé et attaché à l'application.")

    # 5. Initialiser les chaînes LangChain (dépendent des LLMs et RAG)
    from app.routes import chat_routes as _chat_routes_module
    _chat_routes_module.initialize_chains_with_app(app) 
    app.logger.info("Chaînes LangChain initialisées et attachées à l'application.")

    from app.routes.chat_routes import chat_bp 
    app.register_blueprint(chat_bp) 
    app.logger.info("Blueprint 'chat_bp' enregistré.")

    app.logger.info("Tous les services ont été initialisés avec succès.")