from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import logging
import os

app = Flask(__name__)
db = SQLAlchemy()
app.logger.setLevel(logging.INFO)
if not app.logger.handlers:
    handler = logging.StreamHandler()
    app.logger.addHandler(handler)

# Fonction pour initialiser tous les services de l'application
def initialize_services_on_startup():
    from config import Config
    
    app.config.from_object(Config)
    app.logger.info("Configuration de l'application chargée.")
    
    db.init_app(app)
    app.logger.info("Base de données SQLAlchemy liée à l'application.")

    # IMPORT LOCAL ET APPEL pour s'assurer que les LLMs sont initialisés en premier
    from app.services import llm_service
    llm_service.initialize_llms() # Initialise chat_llm et embeddings_llm dans llm_service
    app.logger.info("LLMs principal et d'embeddings initialisés.")

    # IMPORT LOCAL ET APPEL pour le Vector Store (dépend des embeddings_llm)
    from app.services import rag_service
    rag_service.initialize_vectorstore() # Initialise vectorstore et retriever
    app.logger.info("Vector Store (RAG) initialisé.")

    # Création des tables de la base de données
    from app import models # Assure que les modèles sont connus de SQLAlchemy
    try:
        db.create_all()
        app.logger.info("Tables de la base de données créées ou déjà existantes.")
    except Exception as e:
        app.logger.error(f"Erreur lors de la création des tables de la base de données: {e}")
        print(f"PRINT ERROR: Erreur lors de la création des tables de la base de données: {e}")
        print("Vérifiez que le serveur PostgreSQL est en cours d'exécution et que l'utilisateur 'dev_user' a les droits 'Create databases' sur la base de données 'mon_premier_rag_db'.")

    # IMPORT LOCAL ET APPEL pour les chaînes LangChain (dépendent des LLMs)
    from app.routes import chat_routes
    chat_routes.initialize_chains() # Initialise les chaînes qui utilisent chat_llm
    app.logger.info("Chaînes LangChain initialisées.")

    app.logger.info("Tous les services ont été initialisés avec succès.")

# Importe les modules de routes et de services pour les enregistrer
# Ces imports se font au niveau global du module, mais les modules eux-mêmes
# ne doivent pas tenter d'utiliser les LLMs/RAG/chaînes avant leur initialisation
from app.routes import chat_routes
from app.services import llm_service, rag_service, conversation_service