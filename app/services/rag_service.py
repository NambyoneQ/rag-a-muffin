import os
from flask import current_app # Importe current_app pour accéder à la config Flask
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.models import DocumentStatus # Importe le modèle DocumentStatus
from app import db # Importe l'instance db
from app.services.llm_service import get_embeddings_llm # Importe la fonction pour obtenir embeddings_llm

# Variables globales pour le vector store et le retriever (précédées d'un underscore pour indiquer qu'elles sont internes au module)
_vectorstore = None
_retriever = None

# Fonction pour initialiser le Vector Store et le Retriever (appelée une seule fois au démarrage)
def initialize_vectorstore():
    global _vectorstore, _retriever # Déclare qu'on modifie les variables globales
    current_app.logger.info(f"Vérification ou création du Vector Store ChromaDB dans '{current_app.config['CHROMA_PERSIST_DIRECTORY']}'...")

    embeddings_llm_instance = get_embeddings_llm()
    if embeddings_llm_instance is None:
        current_app.logger.error("Erreur: Le modèle d'embeddings n'est pas initialisé via get_embeddings_llm(). Impossible de créer le Vector Store.")
        raise RuntimeError("Embeddings LLM not initialized via getter.")

    if os.path.exists(current_app.config['CHROMA_PERSIST_DIRECTORY']) and len(os.listdir(current_app.config['CHROMA_PERSIST_DIRECTORY'])) > 0:
        try:
            current_app.logger.info(f"Chargement du Vector Store existant depuis '{current_app.config['CHROMA_PERSIST_DIRECTORY']}'")
            _vectorstore = Chroma( # Affecte à la variable globale interne
                persist_directory=current_app.config['CHROMA_PERSIST_DIRECTORY'],
                embedding_function=embeddings_llm_instance
            )
            count = _vectorstore._collection.count()
            if count > 0:
                current_app.logger.info(f"Vector Store chargé avec {count} documents.")
                _retriever = _vectorstore.as_retriever(search_kwargs={"k": 5}) # Affecte à la variable globale interne
            else:
                current_app.logger.info("Le Vector Store existant est vide. Procédure d'ingestion lancée.")
                _process_and_create_vectorstore()
        except Exception as e:
            current_app.logger.error(f"Erreur lors du chargement du Vector Store existant: {e}. Une nouvelle ingestion sera tentée.")
            _process_and_create_vectorstore()
    else:
        current_app.logger.info(f"Dossier ChromaDB non trouvé ou vide. Initialisation de l'ingestion des documents...")
        _process_and_create_vectorstore()

# Fonction interne pour le traitement des documents et la création du vector store
def _process_and_create_vectorstore():
    global _vectorstore, _retriever # Déclare qu'on modifie les variables globales
    raw_documents = []

    embeddings_llm_instance = get_embeddings_llm() # Obtient l'instance ici aussi
    if embeddings_llm_instance is None:
        current_app.logger.error("Erreur: Embeddings LLM non initialisé pour le traitement des documents. Impossible de créer le Vector Store.")
        raise RuntimeError("Embeddings LLM not initialized for document processing.")

    if not os.path.exists(current_app.config['KNOWLEDGE_BASE_DIR']):
        current_app.logger.info(f"ATTENTION: Le dossier de base de connaissances '{current_app.config['KNOWLEDGE_BASE_DIR']}' n'existe pas. Création...")
        os.makedirs(current_app.config['KNOWLEDGE_BASE_DIR'])
        current_app.logger.info("Veuillez y placer des documents (fichiers .txt ou .pdf) pour que le RAG fonctionne.")
        current_app.logger.info("Aucun document à traiter pour le moment (dossier vide après création).")
        try:
            if os.path.exists(current_app.config['CHROMA_PERSIST_DIRECTORY']) and len(os.listdir(current_app.config['CHROMA_PERSIST_DIRECTORY'])) > 0:
                _vectorstore = Chroma(persist_directory=current_app.config['CHROMA_PERSIST_DIRECTORY'], embedding_function=embeddings_llm_instance)
                _retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})
                current_app.logger.info("Vector Store existant chargé (vide ou initialisé).")
            else:
                _vectorstore = Chroma.from_documents([], embeddings_llm_instance, persist_directory=current_app.config['CHROMA_PERSIST_DIRECTORY'])
                _vectorstore.persist()
                _retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})
                current_app.logger.info("Vector Store vide créé et persisté.")
        except Exception as e:
            current_app.logger.error(f"Erreur lors de l'initialisation du Vector Store vide: {e}")
            _vectorstore = None
            _retriever = None
        return


    for root, _, files in os.walk(current_app.config['KNOWLEDGE_BASE_DIR']):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file_path.endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        raw_documents.append(doc)
                    current_app.logger.info(f"Document texte brut chargé : {file_path}")
                elif file_path.endswith('.pdf'):
                    current_app.logger.info(f"Document PDF brut chargé : {file_path}")
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        raw_documents.append(doc)
            except Exception as e:
                current_app.logger.error(f"Erreur lors du chargement brut de {file_path}: {e}")

    if not raw_documents:
        current_app.logger.info("Aucun document brut valide trouvé dans le dossier de la base de connaissances. Le RAG ne sera pas fonctionnel pour les questions basées sur vos documents.")
        _vectorstore = None
        _retriever = None
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(raw_documents)
    current_app.logger.info(f"Documents divisés en {len(chunks)} chunks.")

    try:
        _vectorstore = Chroma.from_documents( # Affecte à la variable globale interne
            chunks,
            embeddings_llm_instance, # Utilise l'instance obtenue
            persist_directory=current_app.config['CHROMA_PERSIST_DIRECTORY']
        )
        _vectorstore.persist()
        _retriever = _vectorstore.as_retriever(search_kwargs={"k": 5}) # Affecte à la variable globale interne
        current_app.logger.info("Vector Store ChromaDB créé et persisté. RAG prêt.")
    except Exception as e:
        current_app.logger.error(f"Erreur lors de la création ou persistance du Vector Store ChromaDB: {e}")
        current_app.logger.error("Vérifiez la disponibilité du service LM Studio pour les embeddings sur le port 1234.")
        _vectorstore = None
        _retriever = None

# --- Fonctions pour accéder aux instances du Vector Store et du Retriever après leur initialisation ---
def get_vectorstore():
    if _vectorstore is None:
        raise RuntimeError("Vector Store n'a pas été initialisé. Appelez initialize_vectorstore() au démarrage de l'application.")
    return _vectorstore

def get_retriever():
    if _retriever is None:
        raise RuntimeError("Retriever n'a pas été initialisé. Appelez initialize_vectorstore() au démarrage de l'application.")
    return _retriever