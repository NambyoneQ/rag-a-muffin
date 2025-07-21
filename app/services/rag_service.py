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

# FONCTION UTILITAIRE : Détermine le type de document et le charge
def _load_document(file_path):
    """Charge un document en fonction de son extension."""
    if file_path.endswith('.txt'):
        return TextLoader(file_path, encoding='utf-8').load()
    elif file_path.endswith('.pdf'):
        return PyPDFLoader(file_path).load()
    # NOUVEAU: Ajoutez d'autres extensions de code ici
    elif file_path.endswith(('.py', '.js', '.java', '.cpp', '.c', '.php', '.html', '.css', '.sql', '.json', '.xml', '.yml', '.md')):
        # Pour les fichiers de code, utilisez TextLoader
        return TextLoader(file_path, encoding='utf-8').load()
    else:
        current_app.logger.warning(f"Type de fichier non supporté, ignoré: {file_path}")
        return []

# Fonction pour initialiser le Vector Store et le Retriever
def initialize_vectorstore():
    global _vectorstore, _retriever
    current_app.logger.info(f"Vérification ou création du Vector Store ChromaDB dans '{current_app.config['CHROMA_PERSIST_DIRECTORY']}'...")

    embeddings_llm_instance = get_embeddings_llm()
    if embeddings_llm_instance is None:
        current_app.logger.error("Erreur: Le modèle d'embeddings n'est pas initialisé. Impossible de créer le Vector Store.")
        raise RuntimeError("Embeddings LLM not initialized.")

    with db.session.begin():
        try:
            if os.path.exists(current_app.config['CHROMA_PERSIST_DIRECTORY']) and os.listdir(current_app.config['CHROMA_PERSIST_DIRECTORY']):
                _vectorstore = Chroma(persist_directory=current_app.config['CHROMA_PERSIST_DIRECTORY'], embedding_function=embeddings_llm_instance)
                current_app.logger.info("Vector Store ChromaDB existant chargé.")
            else:
                current_app.logger.info("Dossier ChromaDB non trouvé ou vide. Création d'un nouveau Vector Store.")
                _vectorstore = Chroma(embedding_function=embeddings_llm_instance, persist_directory=current_app.config['CHROMA_PERSIST_DIRECTORY'])
                DocumentStatus.query.delete() # type: ignore [reportCallIssue]
                current_app.logger.info("Anciens statuts de documents effacés pour un nouveau Vector Store.")
        except Exception as e:
            current_app.logger.error(f"Erreur lors du chargement ou de la création du Vector Store: {e}. Tentative de ré-initialisation.")
            if os.path.exists(current_app.config['CHROMA_PERSIST_DIRECTORY']):
                shutil.rmtree(current_app.config['CHROMA_PERSIST_DIRECTORY'])
            _vectorstore = Chroma(embedding_function=embeddings_llm_instance, persist_directory=current_app.config['CHROMA_PERSIST_DIRECTORY'])
            DocumentStatus.query.delete() # type: ignore [reportCallIssue]
            current_app.logger.info("Vector Store complètement recréé suite à une erreur.")

        # Lancer le processus de mise à jour incrémentale
        try:
            _update_vectorstore_from_disk()
        except RuntimeError as e: # Capture l'erreur spécifique de _update_vectorstore_from_disk
            current_app.logger.error(f"Erreur critique lors de la mise à jour incrémentale du Vector Store: {e}")
            # Ne relève pas l'erreur ici pour permettre à l'application de démarrer, mais RAG sera désactivé.
            # Le rollback est déjà effectué à l'intérieur de _update_vectorstore_from_disk()
            _vectorstore = None # S'assure que le vectorstore est None pour désactiver le RAG
            _retriever = None # S'assure que le retriever est None
            return # Sort de la fonction, car le RAG est désactivé


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

    if _vectorstore is None:
        current_app.logger.error("Vector Store n'est pas disponible pour la mise à jour incrémentale. Impossible de traiter les documents.")
        return

    kb_dir = current_app.config['KNOWLEDGE_BASE_DIR']
    code_dir = current_app.config['CODE_BASE_DIR']

    # S'assurer que les dossiers existent
    if not os.path.exists(kb_dir):
        current_app.logger.info(f"ATTENTION: Le dossier de base de connaissances '{kb_dir}' n'existe pas. Création...")
        os.makedirs(kb_dir)
        current_app.logger.info("Veuillez y placer des documents (fichiers .txt ou .pdf) pour que le RAG fonctionne.")

    if not os.path.exists(code_dir):
        current_app.logger.info(f"ATTENTION: Le dossier de code '{code_dir}' n'existe pas. Création...")
        os.makedirs(code_dir)
        current_app.logger.info("Veuillez y placer vos bases de code organisées par projet.")


    # 1. Charger l'état actuel des documents et des codes sur le disque
    current_files_on_disk = {}
    
    for root, _, files in os.walk(kb_dir):
        for file in files:
            file_path = os.path.join(root, file)
            current_files_on_disk[file_path] = {'mtime': os.path.getmtime(file_path), 'type': 'kb'}

    for root, dirs, files in os.walk(code_dir):
        dirs[:] = [d for d in dirs if d != 'chroma_db']
        for file in files:
            file_path = os.path.join(root, file)
            current_files_on_disk[file_path] = {'mtime': os.path.getmtime(file_path), 'type': 'code'}


    # 2. Charger l'état des documents indexés dans la base de données (DocumentStatus)
    indexed_documents_status = {doc.file_path: doc for doc in DocumentStatus.query.all()} # type: ignore [reportCallIssue]

    documents_to_add = []
    chunks_to_delete_from_chroma_sources = []

    # Détection des documents supprimés ou modifiés
    for indexed_path, status_entry in indexed_documents_status.items():
        if indexed_path not in current_files_on_disk:
            current_app.logger.info(f"Document supprimé du disque: {indexed_path}. Suppression de ChromaDB et BDD.")
            chunks_to_delete_from_chroma_sources.append(indexed_path)
            db.session.delete(status_entry)
        elif current_files_on_disk[indexed_path]['mtime'] > status_entry.last_modified.timestamp():
            current_app.logger.info(f"Document modifié: {indexed_path}. Ré-ingestion.")
            chunks_to_delete_from_chroma_sources.append(indexed_path)
            documents_to_add.append(indexed_path)
            
    # Exécuter les suppressions dans ChromaDB
    if chunks_to_delete_from_chroma_sources:
        try:
            # CORRECTION ICI: Itérer et supprimer un par un
            for path_source in chunks_to_delete_from_chroma_sources:
                current_app.logger.info(f"Tentative de suppression de la source: {path_source} de ChromaDB.")
                _vectorstore.delete(where={"source": path_source})
            _vectorstore.persist()
            current_app.logger.info(f"Suppression de {len(chunks_to_delete_from_chroma_sources)} documents obsolètes de ChromaDB et BDD de suivi.")
        except Exception as e:
            current_app.logger.error(f"Erreur lors de la suppression de documents de ChromaDB: {e}")
            current_app.logger.error("ChromaDB peut être corrompu ou les métadonnées de source ne correspondent pas.")
            db.session.rollback()
            raise RuntimeError(f"Échec critique de la suppression ChromaDB: {e}. Ré-ingestion complète nécessaire.")


    # Détection des nouveaux documents
    for disk_path, disk_info in current_files_on_disk.items():
        if disk_path not in indexed_documents_status:
            current_app.logger.info(f"Nouveau document détecté: {disk_path}. Ingestion.")
            documents_to_add.append(disk_path)

    # Ingestion des documents nouveaux et modifiés
    if documents_to_add:
        chunks_to_process = []
        for file_path in documents_to_add:
            try:
                loaded_docs = _load_document(file_path)
                if not loaded_docs:
                    continue

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                temp_chunks = text_splitter.split_documents(loaded_docs)
                
                file_type = current_files_on_disk[file_path]['type']
                project_name = None
                if file_type == 'code':
                    relative_path = os.path.relpath(file_path, current_app.config['CODE_BASE_DIR'])
                    # Assurez-vous que le chemin relatif a au moins un composant de répertoire après la base de code
                    path_components = relative_path.split(os.sep)
                    if len(path_components) > 1: # S'il y a un sous-dossier (le nom du projet)
                        project_name = path_components[0]
                    else:
                        current_app.logger.warning(f"Fichier de code sans sous-dossier de projet direct dans codebase: {file_path}. Il ne sera pas associé à un projet spécifique.")
                        file_type = 'kb' # Traiter comme KB si pas de projet

                for chunk in temp_chunks:
                    chunk.metadata['source'] = file_path
                    chunk.metadata['file_type'] = file_type # 'kb' ou 'code'
                    if project_name:
                        chunk.metadata['project_name'] = project_name
                    chunks_to_process.append(chunk)

                mtime = os.path.getmtime(file_path)
                timestamp_mtime = datetime.datetime.fromtimestamp(mtime)

                if file_path in indexed_documents_status:
                    status_entry = indexed_documents_status[file_path]
                    status_entry.last_modified = timestamp_mtime
                    status_entry.indexed_at = datetime.datetime.now()
                else:
                    status_entry = DocumentStatus(file_path=file_path, last_modified=timestamp_mtime) # type: ignore [reportCallIssue]
                    db.session.add(status_entry)

            except Exception as e:
                current_app.logger.error(f"Erreur lors de la lecture/traitement de {file_path}: {e}. Ce document sera ignoré.")
                # Important: Si une erreur se produit ici (lecture/traitement),
                # on ne veut PAS que ça fasse rollback la transaction entière
                # et désactive le RAG. On veut juste ignorer CE document.
                # L'erreur est relancée comme RuntimeError, ce qui est attrapé plus haut.
                # L'objectif est de ne pas désactiver le RAG pour une seule erreur de fichier.
                # Cependant, le 'raise RuntimeError' interrompra la boucle et désactivera le RAG.
                # Pour éviter la désactivation complète, on devrait attraper l'exception ici
                # et juste loguer, sans relancer.
                # Retirons le raise RuntimeError ici pour ce bloc.
                # db.session.rollback() # Le rollback n'est plus nécessaire ici si on ne relance pas l'erreur
                pass # Laisse l'erreur être gérée par le logger, ne lève plus de RuntimeError ici

        if chunks_to_process:
            current_app.logger.info(f"Ajout de {len(chunks_to_process)} nouveaux chunks à ChromaDB.")
            try:
                _vectorstore.add_documents(chunks_to_process)
                _vectorstore.persist()
                current_app.logger.info("Nouveaux chunks ajoutés et statuts de documents mis à jour.")
            except Exception as e:
                current_app.logger.error(f"Erreur lors de l'ajout de chunks à ChromaDB: {e}")
                current_app.logger.error("La base de connaissances pourrait être incomplète. Annulation des statuts.")
                db.session.rollback()
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