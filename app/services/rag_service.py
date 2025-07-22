# app/services/rag_service.py

import os
from flask import current_app
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredODTLoader # NOUVEAUX IMPORTS
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
    """
    Charge un document en fonction de son extension ou tente de le charger comme texte brut.
    Retourne une liste de documents (LangChain Document) ou une liste vide si le chargement échoue.
    """
    current_app.logger.info(f"Tentative de chargement du fichier : {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower() # Obtient l'extension et la met en minuscules

    try:
        if file_extension == '.pdf':
            docs = PyPDFLoader(file_path).load()
            current_app.logger.info(f"Chargé comme PDF : {file_path}")
            return docs
        elif file_extension == '.docx':
            docs = UnstructuredWordDocumentLoader(file_path).load()
            current_app.logger.info(f"Chargé comme DOCX : {file_path}")
            return docs
        elif file_extension == '.odt':
            docs = UnstructuredODTLoader(file_path).load()
            current_app.logger.info(f"Chargé comme ODT : {file_path}")
            return docs
        else: # Pour tous les autres fichiers (texte, code, etc.)
            loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True) 
            docs = loader.load()
            current_app.logger.info(f"Chargé comme texte brut : {file_path}")
            return docs
    except Exception as e:
        current_app.logger.warning(f"Impossible de charger '{file_path}' (extension '{file_extension}') : {e}. Fichier ignoré.")
        current_app.logger.warning("Cela peut être dû à un format non pris en charge, un fichier corrompu ou un problème d'encodage/dépendance (ex: LibreOffice pour .odt).")
        return [] # Retourne vide si le fichier n'est pas lisible ou s'il y a une erreur de chargement

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

        try:
            _update_vectorstore_from_disk()
        except RuntimeError as e:
            current_app.logger.error(f"Erreur critique lors de la mise à jour incrémentale du Vector Store: {e}")
            _vectorstore = None
            _retriever = None
            return


    if _vectorstore is not None:
        if _vectorstore._collection.count() > 0:
            _retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})
            current_app.logger.info("Vector Store (RAG) initialisé et Retriever prêt.")
        else:
            current_app.logger.info("Vector Store vide après ingestion. RAG sera désactivé temporairement.")
            _retriever = None
    else:
        current_app.logger.error("Le Vector Store n'a pas put être initialisé. Le RAG sera désactivé.")
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
    
    # NOUVEAU: Définir les chemins de l'application elle-même pour EXCLUSION STRICTE
    APP_EXCLUSIONS_RELATIVE_TO_ROOT = [
        'app',                 # Le dossier 'app' de votre chatbot
        'run.py',              # Le fichier run.py à la racine
        'config.py',           # Le fichier config.py
        '.env',                # Fichiers d'environnement
        '.env.example',
        '.gitignore',
        'requirements.txt',
        'chroma_db',           # Le dossier de ChromaDB
        'venv',                # Environnement virtuel (si à la racine)
        '.git'                 # Dossier Git
    ]
    EXCLUDED_ABS_PATHS_NORMALIZED = [
        os.path.normpath(os.path.join(current_app.root_path, p)) 
        for p in APP_EXCLUSIONS_RELATIVE_TO_ROOT
    ]
    
    current_app.logger.info(f"Fichiers/Dossiers de l'application à exclure de l'indexation : {EXCLUDED_ABS_PATHS_NORMALIZED}")


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
    
    # Parcourir les documents de la base de connaissances (kb_documents)
    for root, _, files in os.walk(kb_dir):
        for file in files:
            file_path = os.path.join(root, file)
            normalized_file_path = os.path.normpath(file_path)
            
            if any(normalized_file_path.startswith(ep) for ep in EXCLUDED_ABS_PATHS_NORMALIZED):
                current_app.logger.info(f"Exclusion (KB - code application) : {file_path}")
                continue
            
            current_files_on_disk[file_path] = {'mtime': os.path.getmtime(file_path), 'type': 'kb'}

    # Parcourir les documents de la base de code (codebase)
    for root, dirs, files in os.walk(code_dir):
        # Exclure les sous-dossiers spécifiques à Git, venv, cache, etc. du parcours DANS LE CODEBASE
        dirs[:] = [d for d in dirs if d not in ['.git', 'venv', '__pycache__', 'chroma_db']] 
        
        for file in files:
            file_path = os.path.join(root, file)
            normalized_file_path = os.path.normpath(file_path)
            
            if any(normalized_file_path.startswith(ep) for ep in EXCLUDED_ABS_PATHS_NORMALIZED):
                current_app.logger.info(f"Exclusion (Codebase - code application) : {file_path}")
                continue

            current_files_on_disk[file_path] = {'mtime': os.path.getmtime(file_path), 'type': 'code'}


    # 2. Charger l'état des documents indexés dans la base de données (DocumentStatus)
    indexed_documents_status = {doc.file_path: doc for doc in DocumentStatus.query.all()} # type: ignore [reportCallIssue]

    documents_to_add = []
    chunks_to_delete_from_chroma_sources = []

    # Détection des documents supprimés ou modifiés
    for indexed_path, status_entry in indexed_documents_status.items():
        file_is_on_disk = indexed_path in current_files_on_disk
        file_is_excluded_now = any(os.path.normpath(indexed_path).startswith(ep) for ep in EXCLUDED_ABS_PATHS_NORMALIZED)

        if not file_is_on_disk or file_is_excluded_now:
            current_app.logger.info(f"Document supprimé du disque ou maintenant exclu: {indexed_path}. Suppression de ChromaDB et BDD.")
            chunks_to_delete_from_chroma_sources.append(indexed_path)
            db.session.delete(status_entry)
        elif current_files_on_disk[indexed_path]['mtime'] > status_entry.last_modified.timestamp():
            current_app.logger.info(f"Document modifié: {indexed_path}. Ré-ingestion.")
            chunks_to_delete_from_chroma_sources.append(indexed_path)
            documents_to_add.append(indexed_path)
            
    # Exécuter les suppressions dans ChromaDB
    if chunks_to_delete_from_chroma_sources:
        try:
            for path_source in chunks_to_delete_from_chroma_sources:
                current_app.logger.info(f"Tentative de suppression de la source: {path_source} de ChromaDB.")
                _vectorstore.delete(where={"source": path_source})
            _vectorstore.persist()
            current_app.logger.info(f"Suppression de {len(chunks_to_delete_from_chroma_sources)} documents obsolètes de ChromaDB et BDD de suivi.")
        except Exception as e:
            current_app.logger.error(f"Erreur lors de la suppression de documents de ChromaDB: {e}")
            current_app.logger.error("ChromaDB peut être corrompu ou les métadonnées de source ne correspondent pas.")
            db.session.rollback()
            raise RuntimeError(f"Échec critique de l'ajout à ChromaDB: {e}. Ré-ingestion complète nécessaire.")


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
                    continue # Passe au fichier suivant si non supporté ou erreur de chargement

                file_size_bytes = os.path.getsize(file_path)
                # Décision de chunking basée sur la taille du fichier (environ 5000 caractères = 5KB)
                if file_size_bytes <= 5000: # Pour les petits documents (< environ 5KB)
                    current_app.logger.info(f"Fichier petit (<5KB): {file_path}. Traitement comme un seul chunk.")
                    # Chaque document (chargé par _load_document) est déjà un objet Document.
                    # Pour un seul chunk par fichier, pas besoin de splitter, juste d'ajouter les métadonnées.
                    for doc in loaded_docs:
                        # Assurez-vous que les métadonnées existantes du loader sont conservées
                        # et ajoutez les nôtres.
                        doc.metadata = doc.metadata or {} # S'assurer que metadata est un dict
                        _add_hierarchical_metadata(doc, file_path, current_files_on_disk[file_path]['type'])
                        chunks_to_process.append(doc)
                else: # Pour les documents plus grands (> environ 5KB)
                    current_app.logger.info(f"Fichier grand (>=5KB): {file_path}. Chunking par 1000 caractères.")
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # chunk_size 1000, overlap 200
                    temp_chunks = text_splitter.split_documents(loaded_docs)
                    
                    for chunk in temp_chunks:
                        chunk.metadata = chunk.metadata or {} # S'assurer que metadata est un dict
                        _add_hierarchical_metadata(chunk, file_path, current_files_on_disk[file_path]['type'])
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
                pass 

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


# --- NOUVELLE FONCTION : AJOUTE LES MÉTA-DONNÉES HIÉRARCHIQUES ---
def _add_hierarchical_metadata(doc, file_path, file_type):
    """Ajoute les métadonnées de dossier et de nom de fichier/titre au chunk."""
    # Nettoie le chemin de base pour le calcul du chemin relatif
    base_dir = current_app.config['KNOWLEDGE_BASE_DIR'] if file_type == 'kb' else current_app.config['CODE_BASE_DIR']
    
    relative_path = os.path.relpath(file_path, base_dir)
    path_components = relative_path.split(os.sep) # Sépare le chemin en ses composants

    # Ajoute les noms de dossier comme metadata_level_X
    # Limitez le nombre de niveaux pour ne pas surcharger inutilement
    MAX_FOLDER_LEVELS = 3 # Maximum 3 niveaux de dossier (Client/Projet/TypeDoc)

    for i, component in enumerate(path_components[:-1]): # Exclut le nom du fichier lui-même
        if i < MAX_FOLDER_LEVELS:
            doc.metadata[f'folder_level_{i+1}'] = component
        if i == len(path_components) - 2: # C'est le dernier dossier
            doc.metadata['last_folder_name'] = component

    doc.metadata['file_name'] = os.path.basename(file_path) # Le nom du fichier
    doc.metadata['document_path_relative'] = relative_path # Chemin relatif complet
    doc.metadata['file_type'] = file_type # Conserve le type 'kb' ou 'code'
    
    # Pour le 'document_title', si le loader Unstructured a déjà trouvé un titre, le conserver.
    # Sinon, utiliser le nom du fichier (sans extension) comme titre par défaut.
    if 'title' not in doc.metadata or not doc.metadata['title']: # 'title' est une clé standard de LangChain/Unstructured
        doc.metadata['document_title'] = os.path.splitext(os.path.basename(file_path))[0]
    else:
        doc.metadata['document_title'] = doc.metadata['title'] # Utilise le titre extrait par Unstructured

# --- Fonctions pour accéder aux instances du Vector Store et du Retriever après leur initialisation ---
def get_vectorstore():
    if _vectorstore is None:
        raise RuntimeError("Vector Store n'a pas été initialisé. Appelez initialize_vectorstore() au démarrage de l'application.")
    return _vectorstore

def get_retriever():
    if _retriever is None:
        raise RuntimeError("Retriever n'a pas été initialisé. Appelez initialize_vectorstore() au démarrage de l'application.")
    return _retriever