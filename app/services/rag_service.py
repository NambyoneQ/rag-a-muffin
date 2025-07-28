# app/services/rag_service.py

import os
import datetime
import shutil
import traceback 
from typing import List, Dict, Any, Optional 

from flask import current_app 
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredODTLoader # UnstructuredExcelLoader sera retiré pour .xlsx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 

import openpyxl # NOUVEAU : Importation de openpyxl

from app.models import DocumentStatus 

# Variables globales pour le vector store et le retriever (initialisées par initialize_vectorstore())
_vectorstore: Optional[Chroma] = None 
_retriever: Optional[Any] = None 

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
def _load_document(file_path: str) -> List[Document]: 
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
            # NOUVELLE LOGIQUE : Utilisation de openpyxl pour les fichiers XLSX
            current_app.logger.info(f"Chargement XLSX avec openpyxl : {file_path}")
            workbook = openpyxl.load_workbook(file_path, data_only=True) # data_only=True pour obtenir les valeurs calculées
            
            loaded_excel_docs: List[Document] = []
            for sheet_name_in_wb in workbook.sheetnames:
                sheet = workbook[sheet_name_in_wb]
                
                sheet_rows_data: List[List[str]] = []
                for row in sheet.iter_rows():
                    row_values = [str(cell.value if cell.value is not None else '').strip() for cell in row]
                    if any(val for val in row_values): 
                        sheet_rows_data.append(row_values)
                
                if sheet_rows_data:
                    # Concaténer les lignes de la feuille en un seul texte pour le Document
                    sheet_text_content = "\n".join(["\t".join(row_vals) for row_vals in sheet_rows_data])
                    
                    # Créer un Document LangChain pour chaque feuille
                    doc_metadata = {'sheet_name': sheet_name_in_wb, 'source': file_path, 'file_type': 'kb'}
                    # Assurez-vous que les métadonnées spécifiques à la feuille sont passées
                    if hasattr(sheet, 'title'): # openpyxl sheet has a title attribute
                        doc_metadata['sheet_title'] = sheet.title

                    loaded_excel_docs.append(Document(page_content=sheet_text_content, metadata=doc_metadata))
                else:
                    current_app.logger.info(f"  Feuille '{sheet_name_in_wb}' est vide. Ignorée.")
            
            return loaded_excel_docs

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
    from app import db 

    chroma_dir = app_instance.config['CHROMA_PERSIST_DIRECTORY']
    app_instance.logger.info(f"Vérification ou création du Vector Store ChromaDB dans '{chroma_dir}'...")

    embeddings_llm_instance = app_instance.extensions["llm_service"]["embeddings_llm"]
    if embeddings_llm_instance is None:
        app_instance.logger.error("Erreur: Le modèle d'embeddings n'est pas disponible via app.extensions. Impossible de créer le Vector Store.")
        raise RuntimeError("Embeddings LLM not initialized or not accessible via app.extensions.")

    chroma_dir_existed_before_init_attempt = os.path.exists(chroma_dir) and os.listdir(chroma_dir)

    with db.session.begin(): 
        try:
            _vectorstore = Chroma(
                persist_directory=chroma_dir,
                embedding_function=embeddings_llm_instance
            )
            app_instance.logger.info(f"Vector Store ChromaDB chargé/rechargé depuis '{chroma_dir}'.")

            if not chroma_dir_existed_before_init_attempt:
                db.session.query(DocumentStatus).delete()
                app_instance.logger.info("Anciens statuts de documents effacés car nouveau Vector Store.")

        except Exception as e:
            app_instance.logger.error(f"Erreur lors du chargement ou de la création du Vector Store: {e}. Tentative de ré-initialisation complète.")
            if os.path.exists(chroma_dir):
                shutil.rmtree(chroma_dir)
            _vectorstore = Chroma(
                embedding_function=embeddings_llm_instance,
                persist_directory=chroma_dir
            )
            db.session.query(DocumentStatus).delete()
            app_instance.logger.info("Vector Store complètement recréé suite à une erreur.")

        try:
            _update_vectorstore_from_disk(app_instance)
        except RuntimeError as e:
            app_instance.logger.error(f"Erreur critique lors de la mise à jour incrémentale du Vector Store: {e}")
            _vectorstore = None
            _retriever = None
            return

    if _vectorstore is not None:
        current_chunk_count = _vectorstore._collection.count() 
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
    from app import db 

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
        os.path.normpath(os.path.join(current_app.root_path, str(p))) 
        for p in APP_EXCLUSIONS_RELATIVE_TO_ROOT
    ]
    
    app_instance.logger.info(f"Fichiers/Dossiers de l'application à exclure de l'indexation : {EXCLUDED_ABS_PATHS_NORMALIZED}")


    current_files_on_disk: Dict[str, Dict[str, Any]] = {} 
    
    # Gérer les répertoires de base de connaissances
    if not os.path.exists(kb_dir):
        app_instance.logger.info(f"ATTENTION: Le dossier de base de connaissances '{kb_dir}' n'existe pas. Création...")
        os.makedirs(kb_dir)
        app_instance.logger.info("Veuillez y placer des documents (fichiers .txt ou .pdf) pour que le RAG fonctionne.")
    for root, _, files in os.walk(kb_dir):
        for file in files:
            file_path = os.path.join(root, file)
            normalized_file_path = os.path.normpath(file_path) 

            if any(str(normalized_file_path).startswith(ep) for ep in EXCLUDED_ABS_PATHS_NORMALIZED): 
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
            normalized_file_path = os.path.normpath(file_path) 
            
            if any(str(normalized_file_path).startswith(ep) for ep in EXCLUDED_ABS_PATHS_NORMALIZED): 
                app_instance.logger.info(f"Exclusion (Codebase - code application) : {file_path}")
                continue

            current_files_on_disk[file_path] = {'mtime': os.path.getmtime(file_path), 'type': 'code'}


    # 2. Charger l'état des documents indexés dans la base de données (DocumentStatus)
    indexed_documents_status = {doc.file_path: doc for doc in db.session.query(DocumentStatus).all()} 

    documents_to_add: List[str] = [] 
    chunks_to_delete_from_chroma_sources: List[str] = [] 

    for indexed_path, status_entry in indexed_documents_status.items():
        file_is_on_disk = indexed_path in current_files_on_disk
        file_is_excluded_now = any(os.path.normpath(str(indexed_path)).startswith(ep) for ep in EXCLUDED_ABS_PATHS_NORMALIZED) 

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
        all_chunks_to_add_in_this_run: List[Document] = [] 

        for file_path in documents_to_add:
            try:
                # _load_document retourne maintenant des Documents LangChain (un par feuille Excel)
                loaded_docs = _load_document(file_path)
                if not loaded_docs:
                    current_app.logger.warning(f"Aucun document chargé pour {file_path}. Skipping.")
                    continue 

                file_size_bytes = os.path.getsize(file_path)
                file_extension = os.path.splitext(file_path)[1].lower()

                # --- TRAITEMENT DES DOCUMENTS PAR TYPE DE FICHIER ---
                # Si c'est un fichier Excel, loaded_docs contient un Document par feuille
                if file_extension == '.xlsx':
                    app_instance.logger.info(f"Traitement des feuilles XLSX individuelles pour {file_path}")
                    
                    for doc_element in loaded_docs: # Chaque doc_element est maintenant une feuille Excel
                        sheet_name = doc_element.metadata.get('sheet_name')
                        # Fallback pour le nom de la feuille si non détecté (utilise le nom de fichier si une seule feuille)
                        if sheet_name is None: 
                            sheet_name = os.path.splitext(os.path.basename(file_path))[0]
                        doc_element.metadata['tab'] = sheet_name # Ajout de la métadonnée 'tab'

                        app_instance.logger.info(f"DEBUG RAG: Traitement de la feuille '{sheet_name}' de {file_path}")

                        consolidated_content_from_sheet = doc_element.page_content # Le contenu de la feuille est déjà consolidé
                        lines = [line.strip() for line in consolidated_content_from_sheet.split('\n') if line.strip()]
                        
                        infos_index = -1
                        for i, line in enumerate(lines):
                            if "infos" == line.lower(): # Recherche exacte "infos" (insensible à la casse) sur la ligne
                                infos_index = i
                                break

                        pre_infos_content = ""
                        table_headers: List[str] = []
                        table_data_lines: List[str] = []

                        if infos_index != -1:
                            # Contenu AVANT "Infos" (exclure la ligne "Infos" elle-même)
                            pre_infos_content = "\n".join(lines[:infos_index]).strip()
                            if pre_infos_content:
                                pre_table_doc = Document(page_content=pre_infos_content, metadata=doc_element.metadata.copy())
                                pre_table_doc.metadata['chunk_type'] = "pre_table_context"
                                pre_table_doc.metadata['tab'] = sheet_name 
                                pre_table_doc.metadata['is_table_chunk'] = True 
                                _add_hierarchical_metadata(pre_table_doc, file_path, current_files_on_disk[file_path]['type'])
                                all_chunks_to_add_in_this_run.append(pre_table_doc)
                                app_instance.logger.info(f"Créé chunk 'pre_table_context' pour '{sheet_name}' de {file_path}")

                            # Contenu APRÈS "Infos"
                            post_infos_lines = lines[infos_index + 1:]
                            
                            if post_infos_lines:
                                potential_header_line = post_infos_lines[0]
                                
                                # Tenter de parser comme des en-têtes séparés par '|'
                                if '|' in potential_header_line:
                                    table_headers = [h.strip() for h in potential_header_line.split('|') if h.strip()]
                                    table_data_lines = post_infos_lines[1:] # Les données commencent après la ligne d'en-tête
                                    app_instance.logger.info(f"DEBUG RAG: En-têtes détectés par '|': {table_headers}")
                                else:
                                    # Si pas de '|', et ce n'est pas vide, alors c'est la première ligne de données
                                    # ou un en-tête non formaté. On utilise un en-tête générique.
                                    table_headers = ['Donnée'] 
                                    table_data_lines = post_infos_lines # Toutes les lignes restantes sont des données
                                    app_instance.logger.info(f"DEBUG RAG: Pas de '|' détecté dans l'en-tête potentiel. En-tête générique utilisé.")

                            table_data_lines = [line for line in table_data_lines if "infos" not in line.lower()]

                        else: # Pas de "Infos" trouvé dans cette feuille
                            # Traiter toute la feuille comme un chunk de contexte si pas de "Infos"
                            full_sheet_text_without_infos = "\n".join(lines).strip()
                            if full_sheet_text_without_infos:
                                doc = Document(page_content=full_sheet_text_without_infos, metadata=doc_element.metadata.copy())
                                doc.metadata['chunk_type'] = "full_sheet_context" 
                                doc.metadata['tab'] = sheet_name 
                                doc.metadata['is_table_chunk'] = True 
                                _add_hierarchical_metadata(doc, file_path, current_files_on_disk[file_path]['type'])
                                all_chunks_to_add_in_this_run.append(doc)
                                app_instance.logger.info(f"Créé chunk 'full_sheet_context' pour '{sheet_name}' de {file_path} (pas d'Infos)")
                            else:
                                app_instance.logger.info(f"Feuille '{sheet_name}' est vide après nettoyage. Ignorée.")
                            table_data_lines = [] # S'assurer qu'il n'y a pas de données de tableau à traiter
                            table_headers = []

                        # Formater les données de tableau structurées en chunks Markdown
                        if table_headers and table_data_lines:
                            markdown_table_header_line = "| " + " | ".join(table_headers) + " |\n"
                            markdown_table_separator_line = "|-" + "-|-".join(['-' * len(h) for h in table_headers]) + "-|\n"
                            
                            current_markdown_rows: List[str] = []
                            MAX_CHUNK_CHARS = 4000 * 4 

                            initial_chunk_content_header = f"Données tabulaires extraites (feuille '{sheet_name or 'N/A'}'):\n" + \
                                                         markdown_table_header_line + markdown_table_separator_line
                            
                            current_chunk_chars = len(initial_chunk_content_header) 

                            for row_line in table_data_lines:
                                row_line_stripped = row_line.strip()
                                if not row_line_stripped: continue 

                                # Les lignes de données sont déjà des chaînes textuelles dans table_data_lines
                                formatted_row_for_markdown = "| " + " | ".join(row_line_stripped.split('\t')) + " |" # Split par tab car on a joint par tab

                                if current_chunk_chars + len(formatted_row_for_markdown) + 1 > MAX_CHUNK_CHARS and current_markdown_rows:
                                    structured_content = initial_chunk_content_header + "\n".join(current_markdown_rows)
                                    
                                    structured_doc = Document(page_content=structured_content, metadata=doc_element.metadata.copy())
                                    structured_doc.metadata['is_table_chunk'] = True
                                    structured_doc.metadata['chunk_type'] = "structured_table_data"
                                    structured_doc.metadata['tab'] = sheet_name 
                                    _add_hierarchical_metadata(structured_doc, file_path, current_files_on_disk[file_path]['type'])
                                    all_chunks_to_add_in_this_run.append(structured_doc)
                                    app_instance.logger.info(f"Créé chunk 'structured_table_data' pour '{sheet_name}' de {file_path} (partie)")

                                    current_markdown_rows = []
                                    current_chunk_chars = len(initial_chunk_content_header) 

                                current_markdown_rows.append(formatted_row_for_markdown)
                                current_chunk_chars += len(formatted_row_for_markdown) + 1 

                            if current_markdown_rows:
                                structured_content = initial_chunk_content_header + "\n".join(current_markdown_rows)
                                structured_doc = Document(page_content=structured_content, metadata=doc_element.metadata.copy())
                                structured_doc.metadata['is_table_chunk'] = True
                                structured_doc.metadata['chunk_type'] = "structured_table_data"
                                structured_doc.metadata['tab'] = sheet_name 
                                _add_hierarchical_metadata(structured_doc, file_path, current_files_on_disk[file_path]['type'])
                                all_chunks_to_add_in_this_run.append(structured_doc)
                                app_instance.logger.info(f"Créé dernier chunk 'structured_table_data' pour '{sheet_name}' de {file_path}")
                        else:
                            app_instance.logger.warning(f"Aucune donnée de tableau structurable trouvée (après 'Infos' ou pas d'en-têtes/données valides) pour la feuille '{sheet_name}' de {file_path}.")

                else: # Traitement pour les documents non-Excel (texte, PDF, etc.)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    temp_chunks = text_splitter.split_documents(loaded_docs) 
                    for chunk in temp_chunks:
                        _add_hierarchical_metadata(chunk, file_path, current_files_on_disk[file_path]['type'])
                        all_chunks_to_add_in_this_run.append(chunk)

                mtime = os.path.getmtime(file_path)
                timestamp_mtime = datetime.datetime.fromtimestamp(mtime)

                if file_path in indexed_documents_status:
                    status_entry = indexed_documents_status[file_path]
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

        if all_chunks_to_add_in_this_run: 
            app_instance.logger.info(f"Ajout de {len(all_chunks_to_add_in_this_run)} nouveaux chunks à ChromaDB.")
            _vectorstore.add_documents(all_chunks_to_add_in_this_run) 
            _vectorstore.persist() 
            app_instance.logger.info("Nouveaux chunks ajoutés et statuts de documents mis à jour.")


# --- NOUVELLE FONCTION : AJOUTE LES MÉTA-DONNÉES HIÉRARCHIQUES ---
def _add_hierarchical_metadata(doc: Document, file_path: str, file_type: str): 
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
def get_vectorstore() -> Chroma: 
    if _vectorstore is None:
        raise RuntimeError("Vector Store n'a pas été initialisé. Appelez initialize_vectorstore() au démarrage de l'application.")
    return _vectorstore

def get_retriever() -> Any: 
    if _retriever is None:
        raise RuntimeError("Retriever n'a pas été initialisé. Appelez initialize_vectorstore() au démarrage de l'application.")
    return _retriever