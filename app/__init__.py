# app/__init__.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import logging
import os

# Initialisation de l'instance Flask de l'application (Globale au module)
app = Flask(__name__)

# Initialisation de l'extension SQLAlchemy (Globale au module)
db = SQLAlchemy()

# Configure le logger de l'application Flask
app.logger.setLevel(logging.INFO)
if not app.logger.handlers:
    handler = logging.StreamHandler()
    app.logger.addHandler(handler)

# Dictionnaire pour stocker les instances des services initialisés, accessibles globalement via app.extensions
# C'est la solution pour les dépendances
app.extensions = {
    "llm_service": None,
    "rag_service": None,
    "conversation_service": None
}

# Fonction pour initialiser tous les services de l'application au démarrage
def initialize_services_on_startup():
    from config import Config # Importe la classe Config
    
    app.config.from_object(Config)
    app.logger.info("Configuration de l'application chargée.")
    
    db.init_app(app)
    app.logger.info("Base de données SQLAlchemy liée à l'application.")

    # 1. Créer les tables de la base de données EN PREMIER
    from app import models # Assure que les modèles sont connus de SQLAlchemy
    try:
        db.create_all()
        app.logger.info("Tables de la base de données créées ou déjà existantes.")
    except Exception as e:
        app.logger.error(f"Erreur lors de la création des tables de la base de données: {e}")
        print(f"PRINT ERROR: Erreur lors de la création des tables de la base de données: {e}")
        print("Vérifiez que le serveur PostgreSQL est en cours d'exécution et que l'utilisateur 'dev_user' a les droits 'Create databases' sur la base de données 'mon_premier_rag_db'.")

    # 2. Initialiser le LLM Service et stocker ses instances dans app.extensions
    from app.services import llm_service as _llm_service_module
    _llm_service_module.initialize_llms() # Passe l'instance 'app' pour le contexte et la config
    app.extensions["llm_service"] = {
        "chat_llm": _llm_service_module.get_chat_llm(),
        "embeddings_llm": _llm_service_module.get_embeddings_llm()
    }
    app.logger.info("LLMs principal et d'embeddings initialisés et attachés à l'application.")

    # 3. Initialiser le RAG Service et stocker ses instances
    from app.services import rag_service as _rag_service_module
    try:
        _rag_service_module.initialize_vectorstore() # Peut lever RuntimeError si embeddings sont manquants
        app.extensions["rag_service"] = {
            "vectorstore": _rag_service_module.get_vectorstore(),
            "retriever": _rag_service_module.get_retriever() # C'est ici que l'erreur se produit si le retriever est None
        }
        app.logger.info("Vector Store (RAG) initialisé et attaché à l'application.")
    except RuntimeError as e:
        app.logger.warning(f"Impossible d'initialiser le RAG service : {e}. Le RAG sera désactivé.")
        app.extensions["rag_service"] = {
            "vectorstore": None,
            "retriever": None
        }
    except Exception as e:
        app.logger.error(f"Erreur inattendue lors de l'initialisation du RAG service : {e}. Le RAG sera désactivé.")
        app.extensions["rag_service"] = {
            "vectorstore": None,
            "retriever": None
        }

    # 4. Initialiser le Conversation Service (s'il y a des init spécifiques, sinon juste pour les fonctions)
    from app.services import conversation_service as _conv_service_module
    # Pas d'initialisation spécifique pour l'instant, mais on pourrait le faire
    app.extensions["conversation_service"] = {
        "load_history": _conv_service_module.load_conversation_history,
        "save_message": _conv_service_module.save_message
    }
    app.logger.info("Conversation Service initialisé et attaché à l'application.")


    # 5. Initialiser les chaînes LangChain (dépendent des LLMs et RAG)
    from app.routes import chat_routes as _chat_routes_module
    _chat_routes_module.initialize_chains() # Passe l'instance 'app'
    app.logger.info("Chaînes LangChain initialisées et attachées à l'application.")

    app.logger.info("Tous les services ont été initialisés avec succès.")

# Import des routes (doit se faire au niveau global pour les décorateurs @app.route)
# Les modules de routes accéderont aux services via app.extensions
from app.routes import chat_routes