# app/services/rag_service.py

import os
from flask import current_app
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.models import DocumentStatus
from app import db
from app.services.llm_service import get_embeddings_llm
import datetime
import shutil

# Variables globales pour le vector store et le retriever (initialisées par initialize_vectorstore())
_vectorstore = None
_retriever = None

# Fonction pour initialiser le Vector Store et le Retriever
def initialize_vectorstore():
    global _vectorstore, _retriever
    current_app.logger.info(f"Vérification ou création du Vector Store ChromaDB dans '{current_app.config['CHROMA_PERSIST_DIRECTORY']}'...")

    embeddings_llm_instance = get_embeddings_llm()
    if embeddings_llm_instance is None:
        current_app.logger.error("Erreur: Le modèle d'embeddings n'est pas initialisé. Impossible de créer le Vector Store.")
        raise RuntimeError("Embeddings LLM not initialized.")

    # Ouvrir une transaction de base de données pour toute la durée de l'initialisation du vectorstore
    # afin de s'assurer que les opérations sur DocumentStatus sont atomiques.
    with db.session.begin(): # Utilisation de db.session.begin() pour la transaction principale
        # Charger le Vector Store existant ou le créer vide si non trouvé/corrompu
        try:
            if os.path.exists(current_app.config['CHROMA_PERSIST_DIRECTORY']) and os.listdir(current_app.config['CHROMA_PERSIST_DIRECTORY']):
                _vectorstore = Chroma(persist_directory=current_app.config['CHROMA_PERSIST_DIRECTORY'], embedding_function=embeddings_llm_instance)
                current_app.logger.info("Vector Store ChromaDB existant chargé.")
            else:
                # Si le dossier est vide ou n'existe pas, créer un nouveau Vector Store
                current_app.logger.info("Dossier ChromaDB non trouvé ou vide. Création d'un nouveau Vector Store.")
                _vectorstore = Chroma(embedding_function=embeddings_llm_instance, persist_directory=current_app.config['CHROMA_PERSIST_DIRECTORY'])
                # Effacer toutes les entrées de DocumentStatus car le VS est vide
                DocumentStatus.query.delete() # type: ignore [reportCallIssue]
                current_app.logger.info("Anciens statuts de documents effacés pour un nouveau Vector Store.")
        except Exception as e:
            current_app.logger.error(f"Erreur lors du chargement ou de la création du Vector Store: {e}. Tentative de ré-initialisation.")
            # En cas d'erreur de chargement (ex: corruption), tenter de supprimer et recréer
            if os.path.exists(current_app.config['CHROMA_PERSIST_DIRECTORY']):
                shutil.rmtree(current_app.config['CHROMA_PERSIST_DIRECTORY'])
            _vectorstore = Chroma(embedding_function=embeddings_llm_instance, persist_directory=current_app.config['CHROMA_PERSIST_DIRECTORY'])
            DocumentStatus.query.delete() # type: ignore [reportCallIssue]
            current_app.logger.info("Vector Store complètement recréé suite à une erreur.")

        # Lancer le processus de mise à jour incrémentale
        _update_vectorstore_from_disk()

    # Le commit final de la transaction principale est géré par 'with db.session.begin():'
    # Si une exception est levée dans ce bloc, il y aura un rollback.
    # Si tout se passe bien, il y aura un commit implicite à la fin du bloc 'with'.

    # Finaliser la configuration du retriever
    if _vectorstore is not None:
        if _vectorstore._collection.count() > 0:
            _retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})
            current_app.logger.info("Vector Store (RAG) initialisé et Retriever prêt.")
        else:
            current_app.logger.info("Vector Store vide après ingestion. RAG sera désactivé temporairement.")
            _retriever = None
    else:
        current_app.logger.error("Le Vector Store n'a pas pu être initialisé. Le RAG sera désactivé.")
        _retriever = None

# Fonction interne pour le traitement incrémental des documents
def _update_vectorstore_from_disk():
    global _vectorstore, _retriever
    current_app.logger.info("Début du processus de mise à jour incrémentale des documents.")

    embeddings_llm_instance = get_embeddings_llm()
    if embeddings_llm_instance is None:
        current_app.logger.error("Erreur: Embeddings LLM non initialisé pour le traitement des documents.")
        raise RuntimeError("Embeddings LLM not initialized for document processing.")

    # Vérifier si _vectorstore est None ici (au cas où il y aurait eu une erreur grave plus tôt)
    if _vectorstore is None:
        current_app.logger.error("Vector Store n'est pas disponible pour la mise à jour incrémentale. Impossible de traiter les documents.")
        return

    # S'assurer que le dossier de la base de connaissances existe
    if not os.path.exists(current_app.config['KNOWLEDGE_BASE_DIR']):
        current_app.logger.info(f"ATTENTION: Le dossier de base de connaissances '{current_app.config['KNOWLEDGE_BASE_DIR']}' n'existe pas. Création...")
        os.makedirs(current_app.config['KNOWLEDGE_BASE_DIR'])
        current_app.logger.info("Veuillez y placer des documents (fichiers .txt ou .pdf) pour que le RAG fonctionne.")
        return

    # 1. Charger l'état actuel des documents sur le disque
    current_files_on_disk = {}
    for root, _, files in os.walk(current_app.config['KNOWLEDGE_BASE_DIR']):
        for file in files:
            file_path = os.path.join(root, file)
            current_files_on_disk[file_path] = os.path.getmtime(file_path)

    # 2. Charger l'état des documents indexés dans la base de données (DocumentStatus)
    indexed_documents_status = {doc.file_path: doc for doc in DocumentStatus.query.all()} # type: ignore [reportCallIssue]

    documents_to_add = [] # Liste des chemins de fichiers à ajouter/ré-ingérer
    chunks_to_delete_from_chroma_sources = [] # Sources (file_path) des chunks à supprimer de ChromaDB

    # Détection des documents supprimés ou modifiés
    for indexed_path, status_entry in indexed_documents_status.items():
        if indexed_path not in current_files_on_disk:
            current_app.logger.info(f"Document supprimé du disque: {indexed_path}. Suppression de ChromaDB.")
            chunks_to_delete_from_chroma_sources.append(indexed_path)
            db.session.delete(status_entry) # Suppression directe dans la session actuelle
        elif current_files_on_disk[indexed_path] > status_entry.last_modified.timestamp():
            current_app.logger.info(f"Document modifié: {indexed_path}. Ré-ingestion.")
            chunks_to_delete_from_chroma_sources.append(indexed_path)
            documents_to_add.append(indexed_path)
            # Au lieu de supprimer l'ancienne entrée, nous allons simplement la mettre à jour plus tard.
            # Marquer l'entrée existante pour modification dans la session.
            # Pas besoin de db.session.delete(status_entry) ici car on va UPDATE.
            pass # Cette ligne est intentionnelle pour montrer le changement de logique.

    # Exécuter les suppressions dans ChromaDB (peut être en dehors du contexte db.session.begin() car c'est ChromaDB)
    if chunks_to_delete_from_chroma_sources:
        try:
            for path_source in chunks_to_delete_from_chroma_sources:
                _vectorstore.delete(where={"source": path_source})
            _vectorstore.persist()
            current_app.logger.info(f"Suppression de {len(chunks_to_delete_from_chroma_sources)} documents obsolètes de ChromaDB et BDD de suivi.")
        except Exception as e:
            current_app.logger.error(f"Erreur lors de la suppression de documents de ChromaDB: {e}")
            current_app.logger.error("ChromaDB peut être corrompu ou les métadonnées de source ne correspondent pas.")
            db.session.rollback() # type: ignore [reportCallIssue] # Force un rollback en cas d'erreur ChromaDB
            raise RuntimeError(f"Échec critique de la suppression ChromaDB: {e}. Ré-ingestion complète nécessaire.")


    # Détection des nouveaux documents et préparation pour l'ingestion
    for disk_path, disk_mtime in current_files_on_disk.items():
        if disk_path not in indexed_documents_status:
            current_app.logger.info(f"Nouveau document détecté: {disk_path}. Ingestion.")
            documents_to_add.append(disk_path)
        # Pour les documents modifiés, ils sont déjà dans documents_to_add si détectés ci-dessus.
        # Nous n'avons plus besoin de les "supprimer" explicitement de la BDD ici, car nous allons les Mettre à jour.

    # Ingestion des documents nouveaux et modifiés
    if documents_to_add:
        chunks_to_process = []
        for file_path in documents_to_add:
            try:
                loaded_docs = []
                if file_path.endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                    loaded_docs = loader.load()
                    current_app.logger.info(f"Chargé: {file_path}")
                elif file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load()
                    current_app.logger.info(f"Chargé: {file_path}")

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                temp_chunks = text_splitter.split_documents(loaded_docs)
                for chunk in temp_chunks:
                    chunk.metadata['source'] = file_path # Ajoute le chemin du fichier comme metadata
                    chunks_to_process.append(chunk)

                mtime = os.path.getmtime(file_path)
                timestamp_mtime = datetime.datetime.fromtimestamp(mtime)

                # --- Nouvelle logique pour la mise à jour de DocumentStatus ---
                # Si le document existe déjà dans indexed_documents_status, mettez-le à jour
                if file_path in indexed_documents_status:
                    status_entry = indexed_documents_status[file_path]
                    status_entry.last_modified = timestamp_mtime
                    status_entry.indexed_at = datetime.datetime.now() # Met à jour la date d'indexation
                    # Pas besoin de db.session.add(), SQLAlchemy gère l'update si l'objet est déjà dans la session
                else:
                    # Sinon, créez une nouvelle entrée pour un nouveau document
                    status_entry = DocumentStatus(file_path=file_path, last_modified=timestamp_mtime) # type: ignore [reportCallIssue]
                    db.session.add(status_entry) # type: ignore [reportCallIssue]
                # --- Fin de la nouvelle logique ---

            except Exception as e:
                current_app.logger.error(f"Erreur lors de la lecture/traitement de {file_path}: {e}. Ce document sera ignoré.")
                db.session.rollback() # type: ignore [reportCallIssue] # Force un rollback pour cette transaction
                raise RuntimeError(f"Échec critique du traitement de document: {e}. Vérifiez les permissions ou le format du fichier.")


        if chunks_to_process:
            current_app.logger.info(f"Ajout de {len(chunks_to_process)} nouveaux chunks à ChromaDB.")
            try:
                _vectorstore.add_documents(chunks_to_process)
                _vectorstore.persist()
                current_app.logger.info("Nouveaux chunks ajoutés et statuts de documents mis à jour.")
            except Exception as e:
                current_app.logger.error(f"Erreur lors de l'ajout de chunks à ChromaDB: {e}")
                current_app.logger.error("La base de connaissances pourrait être incomplète. Annulation des statuts.")
                db.session.rollback() # type: ignore [reportCallIssue] # Force un rollback en cas d'erreur ChromaDB ici
                raise RuntimeError(f"Échec critique de l'ajout à ChromaDB: {e}. Ré-ingestion complète nécessaire.")
        elif not chunks_to_process:
            current_app.logger.info("Aucun nouveau chunk à ajouter après traitement des documents détectés.")
    else:
       current_app.logger.info("Aucun nouveau document ou modification détectée.")


# --- Fonctions pour accéder aux instances du Vector Store et du Retriever après leur initialisation ---
def get_vectorstore():
    if _vectorstore is None:
        raise RuntimeError("Vector Store n'a pas été initialisé. Appelez initialize_vectorstore() au démarrage de l'application.")
    return _vectorstore

def get_retriever():
    if _retriever is None:
        raise RuntimeError("Retriever n'a pas été initialisé. Appelez initialize_vectorstore() au démarrage de l'application.")
    return _retriever