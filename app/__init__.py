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
    "rag_service": None,
    "conversation_service": None
}

# Fonction pour initialiser tous les services de l'application au démarrage
# Cette fonction ne s'occupe plus de la config de base ni de db.init_app
def initialize_services_on_startup():
    print("DEBUG: initialize_services_on_startup - Démarrage") # NOUVEAU DEBUG POINT
    # MODIFICATION CRUCIALE : Déplacer l'importation de models ici, APRÈS db.init_app(app)
    # Ceci est LA solution pour la circularité models <-> db.
    from app import models 
    try:
        print("DEBUG: initialize_services_on_startup - Avant db.create_all()") # NOUVEAU DEBUG POINT
        # S'assurer d'un contexte applicatif pour db.create_all()
        with app.app_context(): 
            db.create_all()
        app.logger.info("Tables de la base de données créées ou déjà existantes.")
        print("DEBUG: initialize_services_on_startup - Après db.create_all()") # NOUVEAU DEBUG POINT
    except Exception as e:
        app.logger.error(f"Erreur lors de la création des tables de la base de données: {e}")
        print(f"PRINT ERROR: Erreur lors de la création des tables de la base de données: {e}")
        print("Vérifiez que le serveur PostgreSQL est en cours d'exécution et que l'utilisateur 'dev_user' a les droits 'Create databases' sur la base de données 'mon_premier_rag_db'.")
        # IMPORTANT : Si cette erreur est critique, vous pourriez vouloir quitter ici
        # return # Décommenter si l'erreur de DB doit arrêter le démarrage complet

    # 2. Initialiser le LLM Service
    print("DEBUG: initialize_services_on_startup - Avant LLM Service") # NOUVEAU DEBUG POINT
    from app.services import llm_service as _llm_service_module
    _llm_service_module.initialize_llms()
    app.extensions["llm_service"] = {
        "chat_llm": _llm_service_module.get_chat_llm(),
        "embeddings_llm": _llm_service_module.get_embeddings_llm()
    }
    app.logger.info("LLMs principal et d'embeddings initialisés et attachés à l'application.")
    print("DEBUG: initialize_services_on_startup - Après LLM Service") # NOUVEAU DEBUG POINT

    # 3. Initialiser le RAG Service
    print("DEBUG: initialize_services_on_startup - Avant RAG Service") # NOUVEAU DEBUG POINT
    from app.services import rag_service as _rag_service_module
    try:
        with app.app_context(): # S'assurer d'un contexte applicatif
            _rag_service_module.initialize_vectorstore(app) 
        app.extensions["rag_service"] = {
            "vectorstore": _rag_service_module.get_vectorstore(),
            "retriever": _rag_service_module.get_retriever()
        }
        app.logger.info("Vector Store (RAG) initialisé et attaché à l'application.")
        print("DEBUG: initialize_services_on_startup - Après RAG Service (Succès)") # NOUVEAU DEBUG POINT
    except RuntimeError as e:
        app.logger.warning(f"Impossible d'initialiser le RAG service : {e}. Le RAG sera désactivé.")
        app.extensions["rag_service"] = {
            "vectorstore": None,
            "retriever": None
        }
        print("DEBUG: initialize_services_on_startup - Après RAG Service (RuntimeError)") # NOUVEAU DEBUG POINT
    except Exception as e:
        app.logger.error(f"Erreur inattendue lors de l'initialisation du RAG service : {e}. Le RAG sera désactivé.")
        app.extensions["rag_service"] = {
            "vectorstore": None,
            "retriever": None
        }
        print("DEBUG: initialize_services_on_startup - Après RAG Service (Exception)") # NOUVEAU DEBUG POINT

    # 4. Initialiser le Conversation Service
    print("DEBUG: initialize_services_on_startup - Avant Conversation Service") # NOUVEAU DEBUG POINT
    from app.services import conversation_service as _conv_service_module
    app.extensions["conversation_service"] = {
        "load_history": _conv_service_module.load_conversation_history,
        "save_message": _conv_service_module.save_message
    }
    app.logger.info("Conversation Service initialisé et attaché à l'application.")
    print("DEBUG: initialize_services_on_startup - Après Conversation Service") # NOUVEAU DEBUG POINT

    # 5. Initialiser les chaînes LangChain (dépendent des LLMs et RAG)
    print("DEBUG: initialize_services_on_startup - Avant LangChain Chains") # NOUVEAU DEBUG POINT
    from app.routes import chat_routes as _chat_routes_module
    _chat_routes_module.initialize_chains_with_app(app) # Passe l'instance 'app'
    app.logger.info("Chaînes LangChain initialisées et attachées à l'application.")
    print("DEBUG: initialize_services_on_startup - Après LangChain Chains") # NOUVEAU DEBUG POINT

    # NOUVEL EMPLACEMENT POUR L'IMPORTATION ET L'ENREGISTREMENT DU BLUEPRINT
    print("DEBUG: initialize_services_on_startup - Avant Blueprint Registration") # NOUVEAU DEBUG POINT
    from app.routes.chat_routes import chat_bp 
    app.register_blueprint(chat_bp) 
    app.logger.info("Blueprint 'chat_bp' enregistré.")
    print("DEBUG: initialize_services_on_startup - Après Blueprint Registration") # NOUVEAU DEBUG POINT

    app.logger.info("Tous les services ont été initialisés avec succès.")
    print("DEBUG: initialize_services_on_startup - Fin") # NOUVEAU DEBUG POINT