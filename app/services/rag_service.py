# app/services/rag_service.py

import os
import datetime
import shutil
import traceback 

from flask import current_app 
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredODTLoader, UnstructuredExcelLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 

from app.models import DocumentStatus 

# Variables globales pour le vector store et le retriever (initialisées par initialize_vectorstore())
_vectorstore = None
_retriever = None

# Liste des extensions de fichiers à ignorer explicitement (médias, binaires, etc.)
EXCLUDED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', # Images
    '.mp3', '.wav', '.ogg', '.flac', '.aac', # Audio
    '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', # Vidéo
    '.zip', '.tar', '.gz', '.rar', '.7z', # Archives
    '.exe', '.dll', '.bin', '.dat', '.so', '.dylib', # Exécutables et bibliothèques
    '.ico', '.db', '.sqlite', '.log', '.bak', '.tmp', # Divers
    '.psd', '.ai', '.eps', # Fichiers Adobe
    '.svg', # Images vectorielles (peuvent contenir du code, mais souvent visuelles)
    '.json', '.xml', '.yml', '.yaml', '.csv', '.tsv', # Fichiers de données structurées (souvent texte, mais à double tranchant)
}

# FONCTION UTILITAIRE : Détermine le type de document et le charge
def _load_document(file_path: str): # Ajout de l'annotation de type str
    current_app.logger.info(f"Tentative de chargement du fichier : {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower() 

    if file_extension in EXCLUDED_EXTENSIONS:
        current_app.logger.info(f"Fichier ignoré (extension non-texte/code explicite) : {file_path}")
        return [] 

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
        elif file_extension == '.xlsx':
            docs = UnstructuredExcelLoader(file_path).load()
            current_app.logger.info(f"Chargé comme XLSX : {file_path}")
            return docs
        else: 
            loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
            docs = loader.load()
            current_app.logger.info(f"Chargé comme texte brut (code/texte) : {file_path}")
            return docs
    except Exception as e:
        current_app.logger.warning(f"Impossible de charger '{file_path}' (extension '{file_extension}') : {e}. Fichier ignoré.")
        current_app.logger.warning("Cela peut être dû à un format non pris en charge, un fichier corrompu ou un problème d'encodage/dépendance (ex: LibreOffice pour .odt).")
        return [] 

# Fonction pour initialiser le Vector Store et le Retriever
def initialize_vectorstore(app_instance): 
    global _vectorstore, _retriever
    from app import db # Importation locale de db

    chroma_dir = app_instance.config['CHROMA_PERSIST_DIRECTORY']
    app_instance.logger.info(f"Vérification ou création du Vector Store ChromaDB dans '{chroma_dir}'...")

    embeddings_llm_instance = app_instance.extensions["llm_service"]["embeddings_llm"]
    if embeddings_llm_instance is None:
        app_instance.logger.error("Erreur: Le modèle d'embeddings n'est pas disponible via app.extensions. Impossible de créer le Vector Store.")
        raise RuntimeError("Embeddings LLM not initialized or not accessible via app.extensions.")

    # Détermine si le répertoire ChromaDB existe et contient des fichiers AVANT de tenter de charger/créer
    # Cette variable est utilisée pour savoir si nous devons effacer la table DocumentStatus plus tard
    chroma_dir_existed_before_init_attempt = os.path.exists(chroma_dir) and os.listdir(chroma_dir)

    with db.session.begin(): # Cette transaction gère le commit/rollback pour tout ce qui est en dessous
        try:
            # CHARGEMENT/RECÉATION INCONDITIONNELLE : C'est le changement clé.
            # Toujours essayer de charger depuis persist_directory. Si le dossier est vide/nouveau, Chroma l'initialisera.
            _vectorstore = Chroma(
                persist_directory=chroma_dir,
                embedding_function=embeddings_llm_instance
            )
            app_instance.logger.info(f"Vector Store ChromaDB chargé/rechargé depuis '{chroma_dir}'.")

            # Effacer la table DocumentStatus UNIQUEMENT si ChromaDB était complètement nouveau/vide
            if not chroma_dir_existed_before_init_attempt:
                db.session.query(DocumentStatus).delete()
                app_instance.logger.info("Anciens statuts de documents effacés car nouveau Vector Store.")

        except Exception as e:
            app_instance.logger.error(f"Erreur lors du chargement ou de la création du Vector Store: {e}. Tentative de ré-initialisation complète.")
            # Si une erreur s'est produite, supprimer agressivement et recréer ChromaDB
            if os.path.exists(chroma_dir):
                shutil.rmtree(chroma_dir)
            _vectorstore = Chroma(
                embedding_function=embeddings_llm_instance,
                persist_directory=chroma_dir
            )
            db.session.query(DocumentStatus).delete()
            app_instance.logger.info("Vector Store complètement recréé suite à une erreur.")

        try:
            # Appeler _update_vectorstore_from_disk pour traiter les fichiers et ajouter/supprimer des chunks
            _update_vectorstore_from_disk(app_instance)
        except RuntimeError as e:
            app_instance.logger.error(f"Erreur critique lors de la mise à jour incrémentale du Vector Store: {e}")
            _vectorstore = None
            _retriever = None
            return

    # VÉRIFICATION FINALE ET INITIALISATION DU RETRIEVER :
    # Après _update_vectorstore_from_disk, l'instance _vectorstore devrait être à jour avec les nouveaux chunks.
    if _vectorstore is not None:
        current_chunk_count = _vectorstore._collection.count() # Obtenez le nombre actuel de chunks
        app_instance.logger.info(f"Nombre de chunks actuel dans le Vector Store après mise à jour: {current_chunk_count}")

        if current_chunk_count > 0:
            _retriever = _vectorstore.as_retriever(search_kwargs={"k": 5}) 
            app_instance.logger.info("Vector Store (RAG) initialisé et Retriever prêt.")
        else:
            app_instance.logger.info("Vector Store vide. RAG sera désactivé temporairement.")
            _retriever = None
    else:
        app_instance.logger.error("Le Vector Store n'a pas put être initialisé. Le RAG sera désactivé.")
        _retriever = None

# Fonction interne pour le traitement incrémental des documents
def _update_vectorstore_from_disk(app_instance): 
    global _vectorstore, _retriever
    from app import db # Importation locale de db

    app_instance.logger.info("Début du processus de mise à jour incrémentale des documents.")

    embeddings_llm_instance = app_instance.extensions["llm_service"]["embeddings_llm"]
    if embeddings_llm_instance is None:
        app_instance.logger.error("Erreur: Le modèle d'embeddings n'est pas disponible via app.extensions. Impossible de traiter les documents.")
        raise RuntimeError("Embeddings LLM not initialized or not accessible via app.extensions for document processing.")

    if _vectorstore is None:
        app_instance.logger.error("Vector Store n'est pas disponible pour la mise à jour incrémentale. Impossible de traiter les documents.")
        return

    kb_dir = app_instance.config['KNOWLEDGE_BASE_DIR']
    code_dir = app_instance.config['CODE_BASE_DIR']
    
    APP_EXCLUSIONS_RELATIVE_TO_ROOT = [
        'app',                 
        'run.py',              
        'config.py',           
        '.env',                
        '.env.example',
        '.gitignore',
        'requirements.txt',
        'chroma_db',           
        'venv',                
        '.git'                 
    ]
    EXCLUDED_ABS_PATHS_NORMALIZED = [
        os.path.normpath(os.path.join(current_app.root_path, str(p))) # Ajout de str() pour clarifier le type pour Pylance
        for p in APP_EXCLUSIONS_RELATIVE_TO_ROOT
    ]
    
    app_instance.logger.info(f"Fichiers/Dossiers de l'application à exclure de l'indexation : {EXCLUDED_ABS_PATHS_NORMALIZED}")


    current_files_on_disk = {}
    
    # Gérer les répertoires de base de connaissances
    if not os.path.exists(kb_dir):
        app_instance.logger.info(f"ATTENTION: Le dossier de base de connaissances '{kb_dir}' n'existe pas. Création...")
        os.makedirs(kb_dir)
        app_instance.logger.info("Veuillez y placer des documents (fichiers .txt ou .pdf) pour que le RAG fonctionne.")
    for root, _, files in os.walk(kb_dir):
        for file in files:
            file_path = os.path.join(root, file)
            normalized_file_path = os.path.normpath(file_path) # Pylance devrait être OK ici

            # Utiliser str() pour s'assurer que 'indexed_path' est bien un string pour 'startswith'
            if any(str(normalized_file_path).startswith(ep) for ep in EXCLUDED_ABS_PATHS_NORMALIZED): # Ajout de str()
                app_instance.logger.info(f"Exclusion (KB - code application) : {file_path}")
                continue
            
            current_files_on_disk[file_path] = {'mtime': os.path.getmtime(file_path), 'type': 'kb'}

    # Gérer les répertoires de code
    if not os.path.exists(code_dir):
        app_instance.logger.info(f"ATTENTION: Le dossier de code '{code_dir}' n'existe pas. Création...")
        os.makedirs(code_dir)
        app_instance.logger.info("Veuillez y placer vos bases de code organisées par projet.")
    for root, dirs, files in os.walk(code_dir):
        dirs[:] = [d for d in dirs if d not in ['.git', 'venv', '__pycache__', 'chroma_db']] 
        
        for file in files:
            file_path = os.path.join(root, file)
            normalized_file_path = os.path.normpath(file_path) # Pylance devrait être OK ici
            
            # Utiliser str() pour s'assurer que 'indexed_path' est bien un string pour 'startswith'
            if any(str(normalized_file_path).startswith(ep) for ep in EXCLUDED_ABS_PATHS_NORMALIZED): # Ajout de str()
                app_instance.logger.info(f"Exclusion (Codebase - code application) : {file_path}")
                continue

            current_files_on_disk[file_path] = {'mtime': os.path.getmtime(file_path), 'type': 'code'}


    # 2. Charger l'état des documents indexés dans la base de données (DocumentStatus)
    indexed_documents_status = {doc.file_path: doc for doc in db.session.query(DocumentStatus).all()} 

    documents_to_add = []
    chunks_to_delete_from_chroma_sources = []

    for indexed_path, status_entry in indexed_documents_status.items():
        file_is_on_disk = indexed_path in current_files_on_disk
        file_is_excluded_now = any(os.path.normpath(str(indexed_path)).startswith(ep) for ep in EXCLUDED_ABS_PATHS_NORMALIZED) # Ajout de str() pour indexed_path

        if not file_is_on_disk or file_is_excluded_now:
            app_instance.logger.info(f"Document supprimé du disque ou maintenant exclu: {indexed_path}. Suppression de ChromaDB et BDD.")
            chunks_to_delete_from_chroma_sources.append(indexed_path)
            db.session.delete(status_entry)
        elif db.session.query(DocumentStatus).filter_by(file_path=indexed_path).first() and current_files_on_disk[indexed_path]['mtime'] > status_entry.last_modified.timestamp():
            app_instance.logger.info(f"Document modifié: {indexed_path}. Ré-ingestion.")
            chunks_to_delete_from_chroma_sources.append(indexed_path)
            documents_to_add.append(indexed_path)
            
    if chunks_to_delete_from_chroma_sources:
        for path_source in chunks_to_delete_from_chroma_sources:
            app_instance.logger.info(f"Tentative de suppression de la source: {path_source} de ChromaDB.")
            _vectorstore.delete(where={"source": path_source})
        app_instance.logger.info(f"Suppression de {len(chunks_to_delete_from_chroma_sources)} documents obsolètes de ChromaDB et BDD de suivi.")

    for disk_path, disk_info in current_files_on_disk.items():
        if disk_path not in indexed_documents_status:
            app_instance.logger.info(f"Nouveau document détecté: {disk_path}. Ingestion.")
            documents_to_add.append(disk_path)

    if documents_to_add:
        chunks_to_process = []
        for file_path in documents_to_add:
            try:
                loaded_docs = _load_document(file_path)
                if not loaded_docs:
                    continue 

                file_size_bytes = os.path.getsize(file_path)
                
                for doc_element in loaded_docs:
                    app_instance.logger.info(f"DEBUG RAG: Processing doc_element from {file_path}. Metadata category: {doc_element.metadata.get('category')}")
                    
                    file_extension = os.path.splitext(file_path)[1].lower() 
                    is_table = (doc_element.metadata.get('category') == 'Table' or 
                                file_extension in ['.xlsx', '.ods']) or \
                               (doc_element.metadata.get('text_as_markdown') is not None and doc_element.metadata['text_as_markdown'].strip().startswith('|')) or \
                               (doc_element.page_content.strip().startswith('|') and '\n|---' in doc_element.page_content)
                    
                    if is_table:
                        app_instance.logger.info(f"Détection de tableau (confirmée) dans : {file_path}. Traitement spécifique.")
                        table_text = doc_element.page_content
                        if doc_element.metadata.get('text_as_markdown'):
                            table_text = doc_element.metadata['text_as_markdown'] 

                        table_lines = table_text.split('\n')
                        
                        header_content = ""
                        body_lines = []

                        header_end_idx = -1
                        for i, line in enumerate(table_lines):
                            if line.strip().startswith('|---'):
                                header_end_idx = i
                                break
                        
                        if header_end_idx != -1:
                            header_lines = table_lines[:header_end_idx]
                            body_lines = table_lines[header_end_idx + 1:]
                            header_content = "\n".join(header_lines).strip() + "\n" 
                        else:
                            if len(table_lines) > 1 and '|' in table_lines[0]: 
                                header_content = table_lines[0].strip() + "\n" + table_lines[1].strip() + "\n" 
                                body_lines = table_lines[2:]
                            else:
                                body_lines = table_lines
                                header_content = "" 

                        if len(body_lines) > 0: 
                            table_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=250, 
                                chunk_overlap=50, 
                                separators=["\n\n", "\n", " ", ""] 
                            )
                            
                            current_table_chunks = table_splitter.split_text("\n".join(body_lines))
                            
                            for chunk_text in current_table_chunks:
                                new_content = header_content + chunk_text
                                new_doc = Document(page_content=new_content, metadata=doc_element.metadata.copy())
                                new_doc.metadata['is_table_chunk'] = True 
                                _add_hierarchical_metadata(new_doc, file_path, current_files_on_disk[file_path]['type'])
                                chunks_to_process.append(new_doc)
                        else: 
                            new_doc = Document(page_content=table_text, metadata=doc_element.metadata.copy())
                            new_doc.metadata['is_table_chunk'] = True
                            _add_hierarchical_metadata(new_doc, file_path, current_files_on_disk[file_path]['type'])
                            chunks_to_process.append(new_doc)

                    else: 
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        temp_chunks = text_splitter.split_documents([doc_element])
                        for chunk in temp_chunks:
                            _add_hierarchical_metadata(chunk, file_path, current_files_on_disk[file_path]['type'])
                            chunks_to_process.append(chunk)

                mtime = os.path.getmtime(file_path)
                timestamp_mtime = datetime.datetime.fromtimestamp(mtime)

                if file_path in indexed_documents_status:
                    status_entry = indexed_documents_status[file_path]
                    # Ces affectations sont correctes, Pylance devrait être silencieux avec les types Mapped
                    status_entry.last_modified = timestamp_mtime 
                    status_entry.indexed_at = datetime.datetime.now() 
                else:
                    status_entry = DocumentStatus(
                        file_path=file_path, 
                        last_modified=timestamp_mtime, 
                        indexed_at=datetime.datetime.now()
                    ) 
                    db.session.add(status_entry) 

            except Exception as e:
                app_instance.logger.error(f"Erreur lors de la lecture/traitement de {file_path}: {e}. Ce document sera ignoré.")
                app_instance.logger.error(f"TRACEBACK TABLE PROCESSING ERROR: \n{traceback.format_exc()}")
                pass 

        if chunks_to_process:
            app_instance.logger.info(f"Ajout de {len(chunks_to_process)} nouveaux chunks à ChromaDB.")
            _vectorstore.add_documents(chunks_to_process) # Ajout explicite des documents à l'instance _vectorstore
            _vectorstore.persist() # S'assurer que les changements sont persistés sur disque
            app_instance.logger.info("Nouveaux chunks ajoutés et statuts de documents mis à jour.")


# --- NOUVELLE FONCTION : AJOUTE LES MÉTA-DONNÉES HIÉRARCHIQUES ---
def _add_hierarchical_metadata(doc: Document, file_path: str, file_type: str): # Ajout des annotations de type
    """Ajoute les métadonnées de dossier et de nom de fichier/titre au chunk."""
    base_dir = current_app.config['KNOWLEDGE_BASE_DIR'] if file_type == 'kb' else current_app.config['CODE_BASE_DIR']
    
    relative_path = os.path.relpath(file_path, base_dir)
    path_components = relative_path.split(os.sep) 

    MAX_FOLDER_LEVELS = 3 

    for i, component in enumerate(path_components[:-1]): 
        if i < MAX_FOLDER_LEVELS:
            doc.metadata[f'folder_level_{i+1}'] = component
        if i == len(path_components) - 2: 
            doc.metadata['last_folder_name'] = component

    doc.metadata['file_name'] = os.path.basename(file_path) 
    doc.metadata['document_path_relative'] = relative_path 
    doc.metadata['file_type'] = file_type 
    
    if file_type == 'code' and len(path_components) > 0: 
        doc.metadata['project_name'] = path_components[0]
        current_app.logger.info(f"Ajout du metadata 'project_name': {path_components[0]} pour {file_path}")

    if 'title' not in doc.metadata or not doc.metadata['title']: 
        doc.metadata['document_title'] = os.path.splitext(os.path.basename(file_path))[0]
    else:
        doc.metadata['document_title'] = doc.metadata['title'] 

# --- Fonctions pour accéder aux instances du Vector Store et du Retriever après leur initialisation ---
def get_vectorstore():
    if _vectorstore is None:
        raise RuntimeError("Vector Store n'a pas été initialisé. Appelez initialize_vectorstore() au démarrage de l'application.")
    return _vectorstore

def get_retriever():
    if _retriever is None:
        raise RuntimeError("Retriever n'a pas été initialisé. Appelez initialize_vectorstore() au démarrage de l'application.")
    return _retriever