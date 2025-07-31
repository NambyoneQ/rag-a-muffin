# app/routes/chat_routes.py

import os
import uuid
import re
import logging
from flask import render_template, request, jsonify, Blueprint, current_app

# Assurez-vous d'importer les types pour les annotations
from typing import List, Dict, Any, Optional

# (Ces imports sont basés sur votre code initial, ajustez si votre structure a changé)
# Assurez-vous que ces modules existent et sont accessibles dans votre projet
from app.services.conversation_service import load_conversation_history, save_message
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

# Créez une instance de Blueprint
chat_bp = Blueprint('chat_bp', __name__, template_folder='../templates', static_folder='../static')

# Variables globales pour les chaînes LangChain et prompts
rag_strict_prompt: Optional[ChatPromptTemplate] = None
rag_fallback_prompt: Optional[ChatPromptTemplate] = None
code_analysis_prompt: Optional[ChatPromptTemplate] = None
general_llm_chain: Optional[Any] = None
general_llm_prompt_template: Optional[ChatPromptTemplate] = None

_llm_instances_by_session_id: Dict[str, ChatOpenAI] = {}


def initialize_chains_with_app(app_instance):
    """
    Initialise les chaînes LangChain avec l'instance de l'application Flask.
    Cette fonction devrait être appelée une fois au démarrage de l'application.
    """
    global rag_strict_prompt, rag_fallback_prompt, code_analysis_prompt, general_llm_chain, general_llm_prompt_template

    chat_llm_instance_global = app_instance.extensions["llm_service"]["chat_llm"]
    if chat_llm_instance_global is None:
        app_instance.logger.error("Erreur: Le chat LLM n'est pas initialisé via app.extensions.")
        raise RuntimeError("Chat LLM not initialized for chain creation.")

    # Prompts pour le RAG
    rag_strict_prompt = ChatPromptTemplate.from_messages([
        ("system", "En te basant STRICTEMENT et UNIQUEMENT sur le CONTEXTE fourni, réponds à la question de l'utilisateur de manière concise. Si la réponse n'est PAS dans le CONTEXTE, dis CLAIREMENT 'Je ne trouve pas cette information dans les documents fournis.' Ne fabrique pas de réponses."),
        ("system", "Contexte: {context}"),
        ("system", "Historique de la conversation: {chat_history}"),
        ("user", "{input}")
    ])

    rag_fallback_prompt = ChatPromptTemplate.from_messages([
        ("system", "En te basant sur le CONTEXTE fourni et tes connaissances générales, réponds à la question de l'utilisateur de manière précise. Si le contexte ne contient pas l'information principale, utilise tes connaissances générales. Ne fabrique pas de réponses qui ne seraient basées sur rien."),
        ("system", "Contexte: {context}"),
        ("system", "Historique de la conversation: {chat_history}"),
        ("user", "{input}")
    ])

    # Prompt pour l'analyse de code (si applicable à votre projet)
    code_analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un expert en analyse de code. Basé STRICTEMENT et UNIQUEMENT sur le CONTEXTE de code fourni pour le projet '{project_name}', réponds à la question. Fournis des explications claires et concises, ainsi que des exemples de code si pertinent. Si l'information n'est PAS dans le CONTEXTE ou ne concerne PAS le projet spécifié, dis CLAIREMENT 'Je ne trouve pas cette information dans le code source du projet '{project_name}'.' Ne fabrique pas de réponses."),
        ("system", "Contexte du code: {context}"),
        ("system", "Historique de la conversation: {chat_history}"),
        ("user", "{input}")
    ])

    # Chaîne de documents générale (utilisée par create_retrieval_chain)
    document_chain = create_stuff_documents_chain(chat_llm_instance_global, rag_fallback_prompt)

    # Prompt et chaîne pour le LLM général (sans RAG)
    general_llm_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Vous êtes un assistant IA expert en développement, en codage, et en configurations Linux. Répondez aux questions de manière précise et concise, fournissez des exemples de code si nécessaire. Si vous ne trouvez pas la réponse, dites simplement que vous ne savez pas."),
        ("system", "{chat_history}"),
        ("user", "{input}")
    ])
    general_llm_chain = general_llm_prompt_template | chat_llm_instance_global

    app_instance.logger.info("Chaînes LangChain (prompts RAG et general_llm_chain) initialisées.")


# Définition des routes du Blueprint

@chat_bp.route('/')
def index():
    from app import models # Import local pour éviter les références circulaires au démarrage
    code_base_dir = current_app.config.get('CODE_BASE_DIR')
    project_names = []
    if code_base_dir and os.path.exists(code_base_dir):
        # Lister les sous-répertoires dans CODE_BASE_DIR comme noms de projets
        project_names = [d for d in os.listdir(code_base_dir) if os.path.isdir(os.path.join(code_base_dir, d))]

    conversations = models.Conversation.query.order_by(models.Conversation.timestamp.desc()).all()
    current_chat_model = current_app.config.get('LMSTUDIO_CHAT_MODEL', 'Modèle non défini')
    return render_template('index.html', conversations=conversations, project_names=project_names, current_chat_model=current_chat_model)


@chat_bp.route('/conversations', methods=['GET'])
def get_all_conversations():
    from app import models # Import local
    conversations = models.Conversation.query.order_by(models.Conversation.timestamp.desc()).all()
    return jsonify([{'id': conv.id, 'name': conv.name} for conv in conversations])


@chat_bp.route('/conversations', methods=['POST'])
def create_conversation():
    from app import db, models # Import local
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        current_app.logger.error(f"Requête JSON manquante ou malformée pour créer conversation. Data reçue: {data}")
        return jsonify({'error': 'Requête invalide: JSON manquant ou malformé.'}), 400

    name = data.get('name')
    if not name:
        current_app.logger.error("Nom de conversation manquant dans la requête JSON.")
        return jsonify({'error': 'Le nom de la conversation est requis.'}), 400

    existing_conversations_count = models.Conversation.query.count()
    if existing_conversations_count >= 3: # Limite le nombre de conversations persistantes
        return jsonify({'error': 'Vous ne pouvez pas créer plus de 3 conversations persistantes.'}), 400

    new_conv = models.Conversation(name=name)
    try:
        db.session.add(new_conv)
        db.session.commit()
        return jsonify({'id': new_conv.id, 'name': new_conv.name}), 201
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Erreur lors de la création de la conversation: {e}")
        return jsonify({'error': f'Erreur lors de la création de la conversation: {e}'}), 500


@chat_bp.route('/chat', methods=['POST'])
def chat():
    request_data = request.get_json(silent=True)
    if not isinstance(request_data, dict):
        current_app.logger.error(f"Requête JSON manquante ou malformée pour chat. Data reçue: {request_data}")
        return jsonify({'response': "Erreur: La requête n'est pas au format JSON valide."}), 400

    # Extraction des données de la requête
    user_message = request_data.get('message', '')
    conv_id_from_request = request_data.get('conversation_id')
    ephemeral_history_from_frontend = request_data.get('ephemeral_history', []) # Historique non persistant du frontend
    rag_mode = request_data.get('rag_mode', 'fallback_rag') # Mode RAG (kb_rag, code_rag, general)
    selected_project = request_data.get('selected_project', None) # Projet sélectionné pour l'analyse de code
    strict_mode = request_data.get('strict_mode', False) # Mode strict pour le RAG
    selected_llm_model_name = request_data.get('llm_model_name') # Modèle LLM sélectionné par l'utilisateur

    chat_llm_instance_for_current_request: Optional[ChatOpenAI] = None
    final_chat_history_for_llm: List[BaseMessage] = []

    session_key: Optional[str] = None # Clé de session unique pour gérer les instances LLM et l'historique

    # Initialisation des variables pour éviter les avertissements "possibly unbound"
    detected_tab_names: List[str] = []
    matched_project_filters: List[Dict[str, Any]] = [] # Nouvelle liste pour stocker les filtres de projets/marques

    actual_llm_model_to_use = selected_llm_model_name if selected_llm_model_name else current_app.config['LMSTUDIO_CHAT_MODEL']

    # Logique pour déterminer la clé de session et charger l'historique
    if conv_id_from_request == "new_ephemeral_session_request" or conv_id_from_request is None:
        session_key = str(uuid.uuid4()) # Générer un nouvel UUID pour la session éphémère
        current_app.logger.info(f"Mode 'Nouvelle Conversation Éphémère' ou Première requête: Nouvelle session_key générée: {session_key}")
        final_chat_history_for_llm = []
    elif isinstance(conv_id_from_request, int):
        session_key = str(conv_id_from_request)
        current_app.logger.info(f"Conversation persistante ID: {session_key}. Chargement de l'historique.")
        final_chat_history_for_llm = load_conversation_history(conv_id_from_request)
    else:
        # Fallback pour les ID de conversation inattendus (ex: chaîne vide, etc.)
        session_key = str(uuid.uuid4())
        current_app.logger.warning(f"ID de conversation inattendu reçu: {conv_id_from_request}. Création d'une nouvelle session éphémère avec ID: {session_key}")
        final_chat_history_for_llm = []

    # Gestion de l'instance LLM par clé de session (pour conserver la mémoire et le modèle choisi)
    if session_key is not None:
        if session_key not in _llm_instances_by_session_id:
            chat_llm_instance_for_current_request = ChatOpenAI(
                base_url=current_app.config['LMSTUDIO_UNIFIED_API_BASE'],
                api_key=current_app.config['LMSTUDIO_API_KEY'],
                model=actual_llm_model_to_use,
                temperature=0.2, # Température pour des réponses plus stables et factuelles
                model_kwargs={"user": session_key} # ID utilisateur pour LM Studio (si supporté)
            )
            _llm_instances_by_session_id[session_key] = chat_llm_instance_for_current_request
        else:
            chat_llm_instance_for_current_request = _llm_instances_by_session_id[session_key]
            # Si le modèle sélectionné change, mettre à jour l'instance LLM
            if chat_llm_instance_for_current_request.model_name != actual_llm_model_to_use:
                chat_llm_instance_for_current_request = ChatOpenAI(
                    base_url=current_app.config['LMSTUDIO_UNIFIED_API_BASE'],
                    api_key=current_app.config['LMSTUDIO_API_KEY'],
                    model=actual_llm_model_to_use,
                    temperature=0.2,
                    model_kwargs={"user": session_key}
                )
                _llm_instances_by_session_id[session_key] = chat_llm_instance_for_current_request
    else:
        current_app.logger.error("Session key non définie après la logique d'identification de session. Impossible de créer l'instance LLM.")
        return jsonify({'response': "Erreur interne: Clé de session LLM non définie."}), 500

    response_content = "Désolé, une erreur inattendue est survenue lors de la génération de la réponse."

    try:
        # Initialisation de la mémoire de conversation
        temp_memory = ConversationSummaryBufferMemory(
            llm=chat_llm_instance_for_current_request,
            max_token_limit=2000, # Limite de tokens pour l'historique de conversation
            memory_key="chat_history",
            return_messages=True
        )
        temp_memory.chat_memory.add_messages(final_chat_history_for_llm)

        # Récupération du vector store et du retriever depuis les extensions de l'application
        current_vectorstore = current_app.extensions["rag_service"]["vectorstore"]
        current_retriever = current_app.extensions["rag_service"]["retriever"]

        use_rag_processing = False
        retrieved_docs: List[Document] = []
        retriever_to_use: Optional[Any] = None # Peut être un retriever filtré ou le retriever global
        filter_applied_log = "Aucun filtre de métadonnées." # Pour le débogage

        # Initialisation du dictionnaire de filtre pour ChromaDB.
        # Utilise List[Dict[str, Any]] pour les valeurs de $and, afin de gérer différentes structures de filtre.
        final_metadata_filter: Dict[str, List[Dict[str, Any]]] = {'$and': []}

        # Initialisation de selected_prompt ici pour s'assurer qu'il est toujours défini
        selected_prompt: Optional[ChatPromptTemplate] = None

        user_message_lower = user_message.lower() # Convertir le message utilisateur en minuscules une seule fois

        # --- LOGIQUE PRINCIPALE BASÉE SUR LE MODE RAG ---
        if rag_mode == 'kb_rag' or rag_mode == 'strict_rag':
            current_app.logger.info(f"DEBUG RAG: Question utilisateur: '{user_message}' (Mode: {rag_mode}, Projet: {selected_project}, Strict: {strict_mode}, Session: {session_key})")
            print(f"PRINT DEBUG RAG: Question utilisateur: '{user_message}' (Mode: {rag_mode}, Projet: {selected_project}, Strict: {strict_mode}, Session: {session_key})")

            selected_prompt = rag_strict_prompt if strict_mode else rag_fallback_prompt
            final_metadata_filter['$and'].append({"file_type": {"$eq": "kb"}})
            filter_applied_log = "Filtre KB: file_type=kb"

            # --- LOGIQUE DE RECONNAISSANCE DE DATE FLEXIBLE ET STANDARDISATION ---
            month_year_pattern = re.compile(
                r"(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)[-\s_]?(\d{4}|\d{2})"
            )
            matches = month_year_pattern.findall(user_message_lower)

            month_mapping = {
                "janvier": "Janvier", "février": "Février", "mars": "Mars", "avril": "Avril",
                "mai": "Mai", "juin": "Juin", "juillet": "Juillet", "août": "Août",
                "septembre": "Septembre", "octobre": "Octobre", "novembre": "Novembre", "décembre": "Décembre"
            }

            for month_word, year_str in matches:
                if month_word in month_mapping:
                    if len(year_str) == 2:
                        current_century_prefix = "20"
                        year_full = current_century_prefix + year_str
                    else:
                        year_full = year_str

                    tab_name = f"{month_mapping[month_word]} {year_full}"
                    if tab_name not in detected_tab_names:
                        detected_tab_names.append(tab_name)

            if detected_tab_names:
                if len(detected_tab_names) == 1:
                    final_metadata_filter['$and'].append({'tab': {'$eq': detected_tab_names[0]}})
                    filter_applied_log += f" + Filtre Tab: {detected_tab_names[0]}"
                else:
                    tab_filters = [{'tab': {'$eq': tab}} for tab in detected_tab_names]
                    final_metadata_filter['$and'].append({'$or': tab_filters})
                    filter_applied_log += f" + Filtre Tab: {', '.join(detected_tab_names)}"
            # --- FIN LOGIQUE DE RECONNAISSANCE DE DATE FLEXIBLE ET STANDARDISATION ---

            # --- LOGIQUE DE FILTRAGE PAR PROJET/MARQUE (Ex: "Sporebio", "Sportlogiq") ---
            project_brand_names_to_match = ["sporebio", "sportlogiq", "fournit", "qwanteos"]
            matched_project_filters.clear()

            for project_name_candidate in project_brand_names_to_match:
                if project_name_candidate.lower() in user_message_lower:
                    # Assurez-vous que le champ de métadonnée 'last_folder_name' ou 'project_name'
                    # correspond à votre indexation dans rag_service.py
                    matched_project_filters.append({"last_folder_name": {"$eq": project_name_candidate.capitalize()}})

            if matched_project_filters:
                if len(matched_project_filters) == 1:
                    # Si un seul projet/marque est détecté, ajoutez directement ce filtre
                    final_metadata_filter['$and'].append(matched_project_filters[0])
                    log_project_names = [matched_project_filters[0].get('last_folder_name', {}).get('$eq', '')] # Pour le log
                    filter_applied_log += f" + Filtre Projet/Marque: {', '.join(log_project_names)}"
                else:
                    # Si plusieurs projets/marques, utilisez l'opérateur $or
                    final_metadata_filter['$and'].append({"$or": matched_project_filters})
                    log_project_names = [f.get('last_folder_name',{}).get('$eq', '') for f in matched_project_filters if isinstance(f, dict) and 'last_folder_name' in f]
                    filter_applied_log += f" + Filtre Projet/Marque: {', '.join(log_project_names)}"
            # --- FIN LOGIQUE DE FILTRAGE PAR PROJET/MARQUE ---


            # --- LOGIQUE DE FILTRAGE POUR LES DONNÉES TABULAIRES (`is_table_chunk`) ---
            table_keywords = ["tableau", "données", "statistiques", "feuille de calcul", "excel", "ods"]
            table_keyword_in_query = any(keyword in user_message_lower for keyword in table_keywords)

            if table_keyword_in_query:
                if {"is_table_chunk": {"$eq": True}} not in final_metadata_filter['$and']:
                    final_metadata_filter['$and'].append({"is_table_chunk": {"$eq": True}})
                    filter_applied_log += " + Filtre Tableau"
            # --- FIN LOGIQUE DE FILTRAGE POUR LES DONNÉES TABULAIRES ---

        elif rag_mode == 'code_rag':
            current_app.logger.info(f"DEBUG RAG: Question utilisateur: '{user_message}' (Mode: {rag_mode}, Projet: {selected_project}, Strict: {strict_mode}, Session: {session_key})")
            print(f"PRINT DEBUG RAG: Question utilisateur: '{user_message}' (Mode: {rag_mode}, Projet: {selected_project}, Strict: {strict_mode}, Session: {session_key})")

            if selected_project:
                selected_prompt = code_analysis_prompt
                final_metadata_filter['$and'].append({"file_type": {"$eq": "code"}})
                final_metadata_filter['$and'].append({"project_name": {"$eq": selected_project}})
                filter_applied_log = f"Filtre Code: {final_metadata_filter}"
            else:
                response_content = "Veuillez sélectionner un projet pour le mode d'analyse de code."
                current_app.logger.warning("Mode analyse de code sélectionné sans projet.")
                print("PRINT DEBUG: Mode analyse de code sélectionné sans projet.")
                return jsonify({'response': response_content})

        elif rag_mode == 'general':
            # Pas de traitement RAG spécifique, utilisera directement le LLM général plus tard
            pass
        else:
            current_app.logger.warning(f"Mode RAG inconnu ou non géré: {rag_mode}. Basculement sur LLM général.")
            pass


        # DÉTERMINATION DU FILTRE FINAL ET INVOCATION DU RETRIEVER
        # Cette section ne s'exécute que si un mode RAG (kb_rag ou code_rag) est actif
        if rag_mode != 'general' and current_vectorstore and current_retriever and selected_prompt is not None:
            # Déterminer la structure finale du filtre à passer à ChromaDB
            if not final_metadata_filter['$and']:
                metadata_filter_dict = {} # Si aucune condition n'a été ajoutée, pas de filtre
            elif len(final_metadata_filter['$and']) == 1:
                metadata_filter_dict = final_metadata_filter['$and'][0] # Déballer le seul filtre
            else:
                metadata_filter_dict = final_metadata_filter # Utiliser le $and avec plusieurs filtres

            current_app.logger.info(f"DEBUG RAG: Préparation du retriever avec filtres: {metadata_filter_dict}")
            print(f"PRINT DEBUG RAG: Préparation du retriever avec filtres: {metadata_filter_dict}")

            try:
                k_value_for_retriever = 5 # Valeur par défaut pour le nombre de documents à récupérer
                if detected_tab_names:
                    k_value_for_retriever = min(len(detected_tab_names) * 3, 10)
                    k_value_for_retriever = max(2, k_value_for_retriever)
                elif matched_project_filters:
                     k_value_for_retriever = 5 # Peut être ajusté pour les requêtes de projet

                if metadata_filter_dict:
                    retriever_to_use = current_vectorstore.as_retriever(search_kwargs={"k": k_value_for_retriever, "filter": metadata_filter_dict})
                    current_app.logger.info(f"DEBUG RAG: Utilisation du retriever filtré avec filtre: {filter_applied_log} (k={k_value_for_retriever})")
                else:
                    retriever_to_use = current_retriever
                    current_app.logger.warning("DEBUG RAG: Mode RAG sélectionné mais pas de filtre de métadonnées appliqué. Utilisation du retriever global.")

                if retriever_to_use is not None:
                    retrieved_docs = retriever_to_use.invoke(user_message)
                    if len(retrieved_docs) > 0:
                        use_rag_processing = True # Des documents ont été récupérés, le RAG sera utilisé
                    current_app.logger.info(f"DEBUG RAG: Documents récupérés via retriever: {len(retrieved_docs)}")
                    print(f"PRINT DEBUG RAG: Documents récupérés via retriever: {len(retrieved_docs)}")
                    for i, doc in enumerate(retrieved_docs):
                        source_info = doc.metadata.get('source', 'N/A') if isinstance(doc.metadata, dict) else 'N/A'
                        tab_info = doc.metadata.get('tab', 'N/A') if isinstance(doc.metadata, dict) else 'N/A'
                        current_app.logger.info(f"  Doc {i+1} (Source: {source_info}, Tab: {tab_info}): '{doc.page_content[:100]}...'")
                        print(f"PRINT DEBUG RAG:   Doc {i+1} (Source: {source_info}, Tab: {tab_info}): '{doc.page_content[:100]}...'")
                else:
                    current_app.logger.error("Retriever_to_use est None. Impossible d'invoquer.")
                    response_content = "Désolé, le retriever n'a pas pu être préparé."

            except Exception as invoke_e:
                current_app.logger.error(f"Erreur lors de l'invocation du retriever: {invoke_e}")
                import traceback
                current_app.logger.error(f"TRACEBACK RETRIEVER INVOKE ERROR: \n{traceback.format_exc()}")
                response_content = "Désolé, une erreur est survenue lors de la recherche de documents pertinents."
                current_app.logger.info("DEBUG RAG: Fallback suite erreur invocation retriever.")
        # --- FIN DÉTERMINATION DU FILTRE FINAL ET INVOCATION DU RETRIEVER ---


        # Logique finale pour la génération de réponse (RAG ou LLM général)
        if rag_mode != 'general' and use_rag_processing and selected_prompt is not None and retriever_to_use is not None: # <-- Ligne modifiée
            print("PRINT DEBUG RAG: Documents pertinents trouvés. Utilisation du RAG.")
            current_app.logger.info("DEBUG RAG: Utilisation du RAG.")

            chat_llm_instance_for_chain = chat_llm_instance_for_current_request

            dynamic_document_chain = create_stuff_documents_chain(chat_llm_instance_for_chain, selected_prompt)
            rag_chain = create_retrieval_chain(retriever_to_use, dynamic_document_chain)

            try:
                response_langchain = rag_chain.invoke(
                    {"input": user_message, "chat_history": temp_memory.load_memory_variables({})["chat_history"]}
                )
                response_content = response_langchain["answer"]
                current_app.logger.info(f"DEBUG LLM Response (RAG): {response_content[:200]}...")
            except Exception as llm_e:
                current_app.logger.error(f"Erreur lors de l'invocation de la chaîne LLM (RAG): {llm_e}")
                import traceback
                current_app.logger.error(f"TRACEBACK LLM CHAIN ERROR (RAG): \n{traceback.format_exc()}")
                response_content = "Désolé, le modèle de langage a rencontré une erreur lors de la génération de la réponse RAG."

        else:
            # Si pas de mode RAG, ou pas de documents trouvés, basculer vers le LLM général
            print("PRINT DEBUG: Basculement sur le LLM général.")
            current_app.logger.info("DEBUG: Basculement sur le LLM général.")

            _general_llm_chain = general_llm_chain
            if _general_llm_chain is not None:
                try:
                    response_langchain = _general_llm_chain.invoke({
                        "input": user_message,
                        "chat_history": temp_memory.load_memory_variables({})["chat_history"]
                    })
                    response_content = response_langchain.content
                    current_app.logger.info(f"DEBUG LLM Response (General/Fallback): {response_content[:200]}...")
                except Exception as llm_e:
                    current_app.logger.error(f"Erreur lors de l'invocation de la chaîne LLM générale (Fallback): {llm_e}")
                    import traceback
                    current_app.logger.error(f"TRACEBACK GENERAL LLM CHAIN ERROR: \n{traceback.format_exc()}")
                    response_content = "Désolé, le modèle de langage général a rencontré une erreur."
            else:
                response_content = "Désolé, le service LLM général n'est pas initialisé."
                current_app.logger.error("LLM général non initialisé, impossible de répondre.")

        # Vérification finale du type de la réponse
        if not isinstance(response_content, str):
            current_app.logger.error(f"response_content est de type inattendu: {type(response_content)}. Valeur: {response_content}. Forçage à une erreur générique.")
            response_content = "Désolé, une erreur interne est survenue lors de la génération de la réponse (format inattendu)."

        # Sauvegarde de l'historique si la conversation est persistante
        if isinstance(conv_id_from_request, int):
            save_message(conv_id_from_request, "user", user_message)
            save_message(conv_id_from_request, "bot", response_content)
        elif session_key and len(session_key) == 36 and session_key.count('-') == 4:
            current_app.logger.info(f"Conversation éphémère (ID: {session_key}), messages non sauvegardés en base de données.")
        else:
            current_app.logger.info(f"Conversation éphémère (ID: {conv_id_from_request}), messages non sauvegardées en base de données.")


        return jsonify({'response': response_content, 'conversation_id': session_key})

    except Exception as e:
        current_app.logger.error(f"Erreur fatale lors du traitement du chat: {e}")
        import traceback
        current_app.logger.error(f"TRACEBACK COMPLET: \n{traceback.format_exc()}")
        print(f"PRINT ERROR: Erreur fatale lors du traitement du chat: {e}")
        print(f"PRINT ERROR: TRACEBACK COMPLET: \n{traceback.format_exc()}")
        return jsonify({'response': f"Désolé, une erreur inattendue et fatale est survenue. Détails : {str(e)}."}), 500

@chat_bp.route('/conversations/<int:conv_id>', methods=['GET'])
def get_conversation_messages(conv_id):
    from app import models # Import local
    messages = models.Message.query.filter_by(conversation_id=conv_id).order_by(models.Message.timestamp).all()
    return jsonify([{'sender': msg.sender, 'content': msg.content} for msg in messages])

@chat_bp.route('/conversations/<int:conv_id>', methods=['DELETE'])
def delete_conversation(conv_id):
    from app import models # Import local
    conv = models.Conversation.query.get(conv_id)
    if not conv:
        return jsonify({'error': 'Conversation non trouvée.'}), 404
    try:
        from app import db # Import local
        db.session.delete(conv)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Conversation supprimée.'}), 200
    except Exception as e:
        from app import db # Import local
        db.session.rollback()
        return jsonify({'error': f'Erreur lors de la suppression de la conversation: {e}'}), 500