# app/routes/chat_routes.py

import os
import uuid
import re 
from flask import render_template, request, jsonify, Blueprint, current_app 
from app.services.conversation_service import load_conversation_history, save_message 
from langchain_openai import ChatOpenAI 

from langchain.memory import ConversationSummaryBufferMemory 
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List, Dict, Any, Optional 
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
    global rag_strict_prompt, rag_fallback_prompt, code_analysis_prompt, general_llm_chain, general_llm_prompt_template

    chat_llm_instance_global = app_instance.extensions["llm_service"]["chat_llm"]
    if chat_llm_instance_global is None:
        app_instance.logger.error("Erreur: Le chat LLM n'est pas initialisé via app.extensions.")
        raise RuntimeError("Chat LLM not initialized for chain creation.")

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

    code_analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un expert en analyse de code. Basé STRICTEMENT et UNIQUEMENT sur le CONTEXTE de code fourni pour le projet '{project_name}', réponds à la question. Fournis des explications claires et concises, ainsi que des exemples de code si pertinent. Si l'information n'est PAS dans le CONTEXTE ou ne concerne PAS le projet spécifié, dis CLAIREMENT 'Je ne trouve pas cette information dans le code source du projet '{project_name}'.' Ne fabrique pas de réponses."),
        ("system", "Contexte du code: {context}"),
        ("system", "Historique de la conversation: {chat_history}"),
        ("user", "{input}")
    ])

    document_chain = create_stuff_documents_chain(chat_llm_instance_global, rag_fallback_prompt) 
    
    general_llm_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Vous êtes un assistant IA expert en développement, en codage, et en configurations Linux. Répondez aux questions de manière précise et concise, fournissez des exemples de code si nécessaire. Si vous ne trouvez pas la réponse, dites simplement que vous ne savez pas."),
        ("system", "{chat_history}"),
        ("user", "{input}")
    ])
    general_llm_chain = general_llm_prompt_template | chat_llm_instance_global

    app_instance.logger.info("Chaînes LangChain (prompts RAG et général_llm_chain) initialisées.")


# Définition des routes du Blueprint
@chat_bp.route('/')
def index():
    from app import models 
    code_base_dir = current_app.config.get('CODE_BASE_DIR') 
    project_names = []
    if code_base_dir and os.path.exists(code_base_dir): 
        project_names = [d for d in os.listdir(code_base_dir) if os.path.isdir(os.path.join(code_base_dir, d))]
    
    conversations = models.Conversation.query.order_by(models.Conversation.timestamp.desc()).all()
    current_chat_model = current_app.config.get('LMSTUDIO_CHAT_MODEL', 'Modèle non défini')
    return render_template('index.html', conversations=conversations, project_names=project_names, current_chat_model=current_chat_model)


@chat_bp.route('/conversations', methods=['GET'])
def get_all_conversations():
    from app import models 
    conversations = models.Conversation.query.order_by(models.Conversation.timestamp.desc()).all()
    return jsonify([{'id': conv.id, 'name': conv.name} for conv in conversations])


@chat_bp.route('/conversations', methods=['POST'])
def create_conversation():
    from app import db, models 
    data = request.get_json(silent=True) 
    if not isinstance(data, dict):
        current_app.logger.error(f"Requête JSON manquante ou malformée pour créer conversation. Data reçue: {data}")
        return jsonify({'error': 'Requête invalide: JSON manquant ou malformé.'}), 400

    name = data.get('name') 
    if not name:
        current_app.logger.error("Nom de conversation manquant dans la requête JSON.")
        return jsonify({'error': 'Le nom de la conversation est requis.'}), 400

    existing_conversations_count = models.Conversation.query.count()
    if existing_conversations_count >= 3:
        return jsonify({'error': 'Vous ne pouvez pas créer plus de 3 conversations persistantes.'}), 400

    new_conv = models.Conversation(name=name) 
    try:
        db.session.add(new_conv)
        db.session.commit()
        return jsonify({'id': new_conv.id, 'name': new_conv.name}), 201
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Erreur lors de la création de la conversation: {e}")
        return jsonify({'error': f'Erreur lors de la création de lahela conversation: {e}'}), 500


@chat_bp.route('/chat', methods=['POST'])
def chat():
    request_data = request.get_json(silent=True)
    if not isinstance(request_data, dict):
        current_app.logger.error(f"Requête JSON manquante ou malformée pour chat. Data reçue: {request_data}")
        return jsonify({'response': "Erreur: La requête n'est pas au format JSON valide."}), 400

    user_message = request_data.get('message', '')
    conv_id_from_request = request_data.get('conversation_id') 
    ephemeral_history_from_frontend = request_data.get('ephemeral_history', [])
    rag_mode = request_data.get('rag_mode', 'fallback_rag')
    selected_project = request_data.get('selected_project', None)
    strict_mode = request_data.get('strict_mode', False)
    selected_llm_model_name = request_data.get('llm_model_name') 

    chat_llm_instance_for_current_request: Optional[ChatOpenAI] = None 
    final_chat_history_for_llm: List[BaseMessage] = []
    
    session_key: Optional[str] = None # Initialisation explicite de session_key

    # Correction ici: Initialisation des variables pour éviter les avertissements "possibly unbound"
    detected_tab_names: List[str] = [] 
    matched_folder_filters: List[Dict[str, Any]] = [] # Initialisation
    folder_matched_in_query: bool = False # Initialisation

    actual_llm_model_to_use = selected_llm_model_name if selected_llm_model_name else current_app.config['LMSTUDIO_CHAT_MODEL']

    # Logique pour déterminer session_key
    if conv_id_from_request == "new_ephemeral_session_request" or conv_id_from_request is None:
        session_key = str(uuid.uuid4()) # Générer un nouvel UUID pour la session éphémère
        current_app.logger.info(f"Mode 'Nouvelle Conversation Éphémère' ou Première requête: Nouvelle session_key générée: {session_key}")
        final_chat_history_for_llm = [] 
    elif isinstance(conv_id_from_request, int):
        session_key = str(conv_id_from_request)
        current_app.logger.info(f"Conversation persistante ID: {session_key}. Chargement de l'historique.")
        final_chat_history_for_llm = load_conversation_history(conv_id_from_request) 
    else:
        session_key = str(uuid.uuid4()) # Fallback: créer un nouvel UUID
        current_app.logger.warning(f"ID de conversation inattendu reçu: {conv_id_from_request}. Création d'une nouvelle session éphémère avec ID: {session_key}")
        final_chat_history_for_llm = []

    # Gestion de l'instance LLM par session_key
    if session_key is not None: 
        if session_key not in _llm_instances_by_session_id:
            chat_llm_instance_for_current_request = ChatOpenAI(
                base_url=current_app.config['LMSTUDIO_UNIFIED_API_BASE'], 
                api_key=current_app.config['LMSTUDIO_API_KEY'], 
                model=actual_llm_model_to_use, 
                temperature=0.2,
                model_kwargs={"user": session_key} 
            )
            _llm_instances_by_session_id[session_key] = chat_llm_instance_for_current_request
        else:
            chat_llm_instance_for_current_request = _llm_instances_by_session_id[session_key]
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
        temp_memory = ConversationSummaryBufferMemory(
            llm=chat_llm_instance_for_current_request, 
            max_token_limit=2000, 
            memory_key="chat_history",
            return_messages=True 
        )
        temp_memory.chat_memory.add_messages(final_chat_history_for_llm)

        current_vectorstore = current_app.extensions["rag_service"]["vectorstore"]
        current_retriever = current_app.extensions["rag_service"]["retriever"]

        use_rag_processing = False
        retrieved_docs: List[Document] = [] 
        selected_prompt: Optional[ChatPromptTemplate] = None
        
        retriever_to_use: Optional[Any] = None 
        filter_applied_log = "Aucun filtre de métadonnées." 

        if current_vectorstore and current_retriever: 
            current_app.logger.info(f"DEBUG RAG: Question utilisateur: '{user_message}' (Mode: {rag_mode}, Projet: {selected_project}, Strict: {strict_mode}, Session: {session_key})")
            print(f"PRINT DEBUG RAG: Question utilisateur: '{user_message}' (Mode: {rag_mode}, Projet: {selected_project}, Strict: {strict_mode}, Session: {session_key})")

            metadata_filter_dict: Dict[str, Any] = {} 

            if rag_mode == 'kb_rag' or rag_mode == 'strict_rag':
                all_filters: List[Dict[str, Any]] = [{"file_type": {"$eq": "kb"}}] 
                selected_prompt = rag_strict_prompt if strict_mode else rag_fallback_prompt
                filter_applied_log = "Filtre KB: file_type=kb"

                user_message_lower = user_message.lower()

                available_folder_names = current_app.extensions.get('available_folder_names', []) 
                
                # matched_folder_filters et folder_matched_in_query sont initialisés en haut
                for folder_name in available_folder_names:
                    if folder_name.lower() in user_message_lower:
                        folder_matched_in_query = True
                        matched_folder_filters.append({"folder_level_1": {"$eq": folder_name}})
                        matched_folder_filters.append({"folder_level_2": {"$eq": folder_name}})
                        matched_folder_filters.append({"folder_level_3": {"$eq": folder_name}})
                        matched_folder_filters.append({"last_folder_name": {"$eq": folder_name}})
                
                if matched_folder_filters:
                    all_filters.append({"$or": matched_folder_filters})
                    filter_applied_log += f" + Filtre Dossier Dynamique"

                # detected_tab_names est initialisé en haut
                month_year_pattern = re.compile(r"(mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|janvier|février)\s*(\d{4})")
                matches = month_year_pattern.findall(user_message_lower)
                
                for month_word, year_str in matches:
                    month_mapping = {
                        "janvier": "Janvier", "février": "Février", "mars": "Mars", "avril": "Avril",
                        "mai": "Mai", "juin": "Juin", "juillet": "Juillet", "août": "Août",
                        "septembre": "Septembre", "octobre": "Octobre", "novembre": "Novembre", "décembre": "Décembre"
                    }
                    if month_word in month_mapping:
                        tab_name = f"statistiques {month_mapping[month_word]} {year_str}" 
                        detected_tab_names.append(tab_name)
                        
                        tab_name_alt = f"{month_mapping[month_word]}_{year_str}"
                        detected_tab_names.append(tab_name_alt)

                if detected_tab_names:
                    tab_filters = [{"tab": {"$eq": tab_name}} for tab_name in detected_tab_names]
                    all_filters.append({"$or": tab_filters})
                    filter_applied_log += f" + Filtre Tab: {', '.join(detected_tab_names)}"

                table_keywords = ["tableau", "données", "statistiques", "feuille de calcul", "excel", "ods"]
                table_keyword_in_query = any(keyword in user_message_lower for keyword in table_keywords)

                if table_keyword_in_query or (folder_matched_in_query and any(kw in user_message_lower for kw in ["sporebio", "sportlogiq"])): 
                    all_filters.append({"is_table_chunk": {"$eq": True}})
                    filter_applied_log += " + Filtre Tableau"
                    if not table_keyword_in_query: 
                        filter_applied_log += " (Auto-détection)"

                if len(all_filters) == 1:
                    metadata_filter_dict = all_filters[0]
                elif len(all_filters) > 1:
                    metadata_filter_dict = {"$and": all_filters}
                else:
                    metadata_filter_dict = {}


                current_app.logger.info(f"DEBUG RAG: Filtres finaux appliqués: {metadata_filter_dict}")
                print(f"PRINT DEBUG RAG: Filtres finaux appliqués: {metadata_filter_dict}")
            
            elif rag_mode == 'code_rag':
                if selected_project:
                    metadata_filter_dict = {
                        "$and": [
                            {"file_type": {"$eq": "code"}},
                            {"project_name": {"$eq": selected_project}}
                        ]
                    }
                    selected_prompt = code_analysis_prompt
                    filter_applied_log = f"Filtre Code: {metadata_filter_dict}"
                else:
                    response_content = "Veuillez sélectionner un projet pour le mode d'analyse de code."
                    current_app.logger.warning("Mode analyse de code sélectionné sans projet.")
                    print("PRINT DEBUG: Mode analyse de code sélectionné sans projet.")
                    return jsonify({'response': response_content})
            
            if rag_mode != 'general':
                print(f"PRINT DEBUG RAG: Value of metadata_filter_dict before final check: {metadata_filter_dict}")
                current_app.logger.info(f"DEBUG RAG: Value of metadata_filter_dict before final check: {metadata_filter_dict}")
                
                try:
                    k_value_for_retriever = 5 
                    if detected_tab_names: 
                        k_value_for_retriever = min(len(detected_tab_names) * 3, 10) 
                        k_value_for_retriever = max(2, k_value_for_retriever) 
                    elif folder_matched_in_query: 
                         k_value_for_retriever = 5
                    
                    if metadata_filter_dict: 
                        retriever_to_use = current_vectorstore.as_retriever(search_kwargs={"k": k_value_for_retriever, "filter": metadata_filter_dict})
                        current_app.logger.info(f"DEBUG RAG: Utilisation du retriever filtré avec filtre: {filter_applied_log} (k={k_value_for_retriever})")
                    else: 
                        retriever_to_use = current_retriever
                        current_app.logger.warning("DEBUG RAG: Mode RAG sélectionné mais pas de filtre de métadonnées appliqué (possible erreur logique ou données). Utilisation du retriever global.")
                    
                    if retriever_to_use is not None:
                        retrieved_docs = retriever_to_use.invoke(user_message)
                        if len(retrieved_docs) > 0:
                            use_rag_processing = True
                        current_app.logger.info(f"DEBUG RAG: Documents récupérés via retriever: {len(retrieved_docs)}")
                        print(f"PRINT DEBUG RAG: Documents récupérés via retriever: {len(retrieved_docs)}")
                        for i, doc in enumerate(retrieved_docs):
                            source_info = doc.metadata.get('source', 'N/A') if isinstance(doc.metadata, dict) else 'N/A'
                            tab_info = doc.metadata.get('tab', 'N/A') if isinstance(doc.metadata, dict) else 'N/A'
                            current_app.logger.info(f"  Doc {i+1} (Source: {source_info}, Tab: {tab_info}): '{doc.page_content[:100]}...'")
                            print(f"PRINT DEBUG RAG:   Doc {i+1} (Source: {source_info}, Tab: {tab_info}): '{doc.page_content[:100]}...'")
                    else:
                        current_app.logger.error("Retriever_to_use est None après la logique de sélection. Impossible d'invoquer.")
                        response_content = "Désolé, le retriever n'a pas pu être préparé."

                except Exception as invoke_e:
                    current_app.logger.error(f"Erreur lors de l'invocation du retriever: {invoke_e}")
                    import traceback
                    current_app.logger.error(f"TRACEBACK RETRIEVER INVOKE ERROR: \n{traceback.format_exc()}")
                    response_content = "Désolé, une erreur est survenue lors de la recherche de documents pertinents."
                    current_app.logger.info("DEBUG RAG: Fallback suite erreur invocation retriever.")
            else: 
                current_app.logger.info("DEBUG RAG: Mode Général ou RAG non initialisé. Pas de recherche de documents via retriever.")
                print("PRINT DEBUG RAG: Mode Général ou RAG non initialisé. Pas de recherche de documents via retriever.")
        else: 
            current_app.logger.warning("DEBUG RAG: Vector Store ou Retriever non initialisé. Impossible d'utiliser le RAG.")
            print("PRINT DEBUG RAG: Vector Store ou Retriever non initialisé. Impossible d'utiliser le RAG.")


        if rag_mode != 'general' and use_rag_processing and selected_prompt is not None and retriever_to_use is not None:
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


        if not isinstance(response_content, str):
            current_app.logger.error(f"response_content est de type inattendu: {type(response_content)}. Valeur: {response_content}. Forçage à une erreur générique.")
            response_content = "Désolé, une erreur interne est survenue lors de la génération de la réponse (format inattendu)."

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
    from app import models 
    messages = models.Message.query.filter_by(conversation_id=conv_id).order_by(models.Message.timestamp).all() 
    return jsonify([{'sender': msg.sender, 'content': msg.content} for msg in messages])

@chat_bp.route('/conversations/<int:conv_id>', methods=['DELETE'])
def delete_conversation(conv_id):
    from app import models 
    conv = models.Conversation.query.get(conv_id)
    if not conv:
        return jsonify({'error': 'Conversation non trouvée.'}), 404
    try:
        from app import db 
        db.session.delete(conv)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Conversation supprimée.'}), 200
    except Exception as e:
        from app import db 
        db.session.rollback()
        return jsonify({'error': f'Erreur lors de la suppression de la conversation: {e}'}), 500