# app/routes/chat_routes.py

import os
import uuid
import re
import logging
import json 
from flask import render_template, request, jsonify, Blueprint, current_app

from typing import List, Dict, Any, Optional

from app.services.conversation_service import load_conversation_history, save_message
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever # Import BaseRetriever for type hinting

chat_bp = Blueprint('chat_bp', __name__, template_folder='../templates', static_folder='../static')

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

    general_llm_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Vous êtes un assistant IA expert en développement, en codage, et en configurations Linux. Répondez aux questions de manière précise et concise, fournissez des exemples de code si nécessaire. Si vous ne trouvez pas la réponse, dites simplement que vous ne savez pas."),
        ("system", "{chat_history}"),
        ("user", "{input}")
    ])
    general_llm_chain = general_llm_prompt_template | chat_llm_instance_global

    app_instance.logger.info("Chaînes LangChain (prompts RAG et general_llm_chain) initialisées.")


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
        return jsonify({'error': f'Erreur lors de la création de la conversation: {e}'}), 500


@chat_bp.route('/chat', methods=['POST'])
def chat():
    request_data = request.get_json(silent=True)
    if not isinstance(request_data, dict):
        current_app.logger.error(f"Requête JSON manquante ou malformée pour chat. Data reçue: {request_data}")
        return jsonify({'response': "Erreur: La requête n'est pas au format JSON valide."}), 400

    user_message = request_data.get('message', '')
    conv_id_from_request = request_data.get('conversation_id')
    ephemeral_history_from_frontend = request_data.get('ephemeral_history', []) 
    rag_mode = request_data.get('rag_mode', 'general') 
    selected_project = request_data.get('selected_project', None) 
    strict_mode = request_data.get('strict_mode', False) 
    selected_llm_model_name = request_data.get('llm_model_name') 

    chat_llm_instance_for_current_request: Optional[ChatOpenAI] = None
    final_chat_history_for_llm: List[BaseMessage] = []
    session_key: Optional[str] = None 
    retrieved_sources: List[Dict[str, Any]] = [] 

    actual_llm_model_to_use = selected_llm_model_name if selected_llm_model_name else current_app.config['LMSTUDIO_CHAT_MODEL']

    if conv_id_from_request == "new_ephemeral_session_request" or conv_id_from_request is None:
        session_key = str(uuid.uuid4()) 
        current_app.logger.info(f"Mode 'Nouvelle Conversation Éphémère' ou Première requête: Nouvelle session_key générée: {session_key}")
        final_chat_history_for_llm = []
        for msg_data in ephemeral_history_from_frontend:
            if msg_data['sender'] == 'user':
                final_chat_history_for_llm.append(HumanMessage(content=msg_data['content']))
            else:
                final_chat_history_for_llm.append(AIMessage(content=msg_data['content']))
    elif isinstance(conv_id_from_request, int):
        session_key = str(conv_id_from_request)
        current_app.logger.info(f"Conversation persistante ID: {session_key}. Chargement de l'historique.")
        final_chat_history_for_llm = load_conversation_history(conv_id_from_request)
    else:
        session_key = str(uuid.uuid4())
        current_app.logger.warning(f"ID de conversation inattendu reçu: {conv_id_from_request}. Création d'une nouvelle session éphémère avec ID: {session_key}")
        final_chat_history_for_llm = []

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

        kb_retriever = current_app.extensions["rag_service"]["kb_retriever"]
        codebase_retriever = current_app.extensions["rag_service"]["codebase_retriever"]

        use_rag_processing = False
        retrieved_docs: List[Document] = []
        retriever_to_use: Optional[BaseRetriever] = None # Use BaseRetriever for type hinting
        filter_applied_log = "Aucun filtre de métadonnées." 

        selected_prompt: Optional[ChatPromptTemplate] = None
        user_message_lower = user_message.lower() 

        if rag_mode == 'kb_rag':
            current_app.logger.info(f"DEBUG RAG: Question utilisateur: '{user_message}' (Mode: {rag_mode}, Strict: {strict_mode}, Session: {session_key})")
            print(f"PRINT DEBUG RAG: Question utilisateur: '{user_message}' (Mode: {rag_mode}, Strict: {strict_mode}, Session: {session_key})")

            selected_prompt = rag_strict_prompt if strict_mode else rag_fallback_prompt
            
            metadata_filters: List[Dict[str, Any]] = [{"file_type": {"$eq": "kb"}}]
            filter_applied_log = "Filtre KB: file_type=kb"

            month_year_pattern = re.compile(
                r"(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)[-\s_]?(\d{4}|\d{2})"
            )
            matches = month_year_pattern.findall(user_message_lower)

            month_mapping = {
                "janvier": "Janvier", "février": "Février", "mars": "Mars", "avril": "Avril",
                "mai": "Mai", "juin": "Juin", "juillet": "Juillet", "août": "Août",
                "septembre": "Septembre", "octobre": "Octobre", "novembre": "Novembre", "décembre": "Décembre"
            }
            detected_tab_names: List[str] = []
            for month_word, year_str in matches:
                if month_word in month_mapping:
                    year_full = year_str if len(year_str) == 4 else f"20{year_str}"
                    tab_name = f"{month_mapping[month_word]} {year_full}"
                    if tab_name not in detected_tab_names:
                        detected_tab_names.append(tab_name)

            if detected_tab_names:
                if len(detected_tab_names) == 1:
                    metadata_filters.append({'tab': {'$eq': detected_tab_names[0]}})
                    filter_applied_log += f" + Filtre Tab: {detected_tab_names[0]}"
                else:
                    tab_or_filters = [{'tab': {'$eq': tab}} for tab in detected_tab_names]
                    metadata_filters.append({'$or': tab_or_filters})
                    filter_applied_log += f" + Filtre Tab: {', '.join(detected_tab_names)}"

            project_brand_names_to_match = ["sporebio", "sportlogiq", "qwanteos"] 
            matched_project_filters: List[Dict[str, Any]] = []

            for project_name_candidate in project_brand_names_to_match:
                if project_name_candidate.lower() in user_message_lower:
                    matched_project_filters.append({"last_folder_name": {"$eq": project_name_candidate.capitalize()}}) 

            if matched_project_filters:
                if len(matched_project_filters) == 1:
                    metadata_filters.append(matched_project_filters[0])
                    log_project_names = [matched_project_filters[0].get('last_folder_name', {}).get('$eq', '')]
                    filter_applied_log += f" + Filtre Projet/Marque: {', '.join(log_project_names)}"
                else:
                    metadata_filters.append({"$or": matched_project_filters})
                    log_project_names = [f.get('last_folder_name',{}).get('$eq', '') for f in matched_project_filters]
                    filter_applied_log += f" + Filtre Projet/Marque: {', '.join(log_project_names)}"

            table_keywords = ["tableau", "données", "statistiques", "feuille de calcul", "excel", "ods"]
            table_keyword_in_query = any(keyword in user_message_lower for keyword in table_keywords)

            if table_keyword_in_query:
                metadata_filters.append({"is_table_chunk": {"$eq": True}})
                filter_applied_log += " + Filtre Tableau"

            final_chromadb_filter: Dict[str, Any] = {}
            if len(metadata_filters) == 1:
                final_chromadb_filter = metadata_filters[0]
            elif len(metadata_filters) > 1:
                final_chromadb_filter = {"$and": metadata_filters}
            
            # --- CORRECTION ICI : Configurer les search_kwargs directement sur le retriever ---
            retriever_to_use = kb_retriever # Get the base retriever
            # Create a new dictionary for search_kwargs to avoid modifying the original global retriever's settings
            search_kwargs_for_this_query = kb_retriever.search_kwargs.copy() 
            if final_chromadb_filter:
                search_kwargs_for_this_query['filter'] = final_chromadb_filter
            retriever_to_use.search_kwargs = search_kwargs_for_this_query # Assign the new search_kwargs

            current_app.logger.info(f"DEBUG RAG: KB Retriever configuré avec filtres: {search_kwargs_for_this_query.get('filter')} (k={search_kwargs_for_this_query.get('k')})")


        elif rag_mode == 'code_rag':
            current_app.logger.info(f"DEBUG RAG: Question utilisateur: '{user_message}' (Mode: {rag_mode}, Projet: {selected_project}, Strict: {strict_mode}, Session: {session_key})")
            print(f"PRINT DEBUG RAG: Question utilisateur: '{user_message}' (Mode: {rag_mode}, Projet: {selected_project}, Strict: {strict_mode}, Session: {session_key})")

            if not codebase_retriever:
                response_content = "Désolé, le service d'indexation de codebase n'est pas disponible."
                current_app.logger.warning("Codebase retriever non initialisé.")
                return jsonify({'response': response_content})

            if selected_project:
                selected_prompt = code_analysis_prompt
                
                code_filters: List[Dict[str, Any]] = [{"file_type": {"$eq": "code"}}]
                code_filters.append({"project_name": {"$eq": selected_project}})
                filter_applied_log = f"Filtre Code: project_name={selected_project}, file_type=code"

                # --- CORRECTION ICI : Configurer les search_kwargs directement sur le retriever ---
                retriever_to_use = codebase_retriever # Get the base retriever
                search_kwargs_for_this_query = codebase_retriever.search_kwargs.copy()
                search_kwargs_for_this_query['filter'] = {"$and": code_filters}
                retriever_to_use.search_kwargs = search_kwargs_for_this_query # Assign the new search_kwargs

                current_app.logger.info(f"DEBUG RAG: Codebase Retriever configuré avec filtres: {search_kwargs_for_this_query.get('filter')} (k={search_kwargs_for_this_query.get('k')})")

            else:
                response_content = "Veuillez sélectionner un projet pour le mode d'analyse de code."
                current_app.logger.warning("Mode analyse de code sélectionné sans projet.")
                print("PRINT DEBUG: Mode analyse de code sélectionné sans projet.")
                return jsonify({'response': response_content})

        elif rag_mode == 'general':
            pass
        else:
            current_app.logger.warning(f"Mode RAG inconnu ou non géré: {rag_mode}. Basculement sur LLM général.")
            pass


        if rag_mode != 'general' and retriever_to_use is not None and selected_prompt is not None:
            current_app.logger.info(f"DEBUG RAG: Invocation du retriever pour {rag_mode}.")
            print(f"PRINT DEBUG RAG: Invocation du retriever pour {rag_mode}.")

            try:
                retrieved_docs = retriever_to_use.invoke(user_message)
                if len(retrieved_docs) > 0:
                    use_rag_processing = True 
                    current_app.logger.info(f"DEBUG RAG: Documents récupérés via retriever: {len(retrieved_docs)}")
                    print(f"PRINT DEBUG RAG: Documents récupérés via retriever: {len(retrieved_docs)}")
                    
                    for doc in retrieved_docs:
                        source_info = {
                            "source": doc.metadata.get('source', 'N/A'),
                            "file_name": doc.metadata.get('file_name', 'N/A'),
                            "type": doc.metadata.get('file_type', 'N/A'),
                            "entity_type": doc.metadata.get('entity_type', 'N/A'), 
                            "entity_name": doc.metadata.get('entity_name', 'N/A'), 
                            "project_name": doc.metadata.get('project_name', 'N/A'), 
                            "sheet_name": doc.metadata.get('sheet_name', 'N/A'), 
                            "start_line": doc.metadata.get('start_line', 'N/A') 
                        }
                        retrieved_sources.append(source_info)

                        source_log_info = f"Source: {doc.metadata.get('source', 'N/A')} "
                        if doc.metadata.get('tab'): source_log_info += f"| Tab: {doc.metadata.get('tab')} "
                        if doc.metadata.get('entity_name'): source_log_info += f"| Entity: {doc.metadata.get('entity_name')} "
                        if doc.metadata.get('project_name'): source_log_info += f"| Project: {doc.metadata.get('project_name')} "
                        current_app.logger.info(f"  Doc: {source_log_info} | Content: '{doc.page_content[:100]}...'")
                        print(f"PRINT DEBUG RAG:   Doc: {source_log_info} | Content: '{doc.page_content[:100]}...'")
                else:
                    current_app.logger.info("DEBUG RAG: Aucun document pertinent trouvé pour la requête avec les filtres appliqués.")
                    print("PRINT DEBUG RAG: Aucun document pertinent trouvé pour la requête.")
                    if strict_mode:
                        response_content = "Je ne trouve pas cette information dans les documents fournis."
                        return jsonify({'response': response_content, 'conversation_id': session_key, 'sources': retrieved_sources})
                    else:
                        current_app.logger.info("DEBUG RAG: Basculement vers le LLM général (mode non strict et aucun document trouvé).")
                        rag_mode = 'general' 
                        use_rag_processing = False 
            except Exception as invoke_e:
                current_app.logger.error(f"Erreur lors de l'invocation du retriever: {invoke_e}")
                import traceback
                current_app.logger.error(f"TRACEBACK RETRIEVER INVOKE ERROR: \n{traceback.format_exc()}")
                response_content = "Désolé, une erreur est survenue lors de la recherche de documents pertinents."
                current_app.logger.info("DEBUG RAG: Fallback suite erreur invocation retriever.")
                rag_mode = 'general' 
                use_rag_processing = False

        if rag_mode != 'general' and use_rag_processing and selected_prompt is not None:
            print("PRINT DEBUG RAG: Documents pertinents trouvés. Utilisation du RAG.")
            current_app.logger.info("DEBUG RAG: Utilisation du RAG.")

            chat_llm_instance_for_chain = chat_llm_instance_for_current_request

            document_chain_for_rag = create_stuff_documents_chain(chat_llm_instance_for_chain, selected_prompt)
            
            try:
                response_langchain = document_chain_for_rag.invoke(
                    {
                        "context": retrieved_docs, 
                        "input": user_message, 
                        "chat_history": temp_memory.load_memory_variables({})["chat_history"]
                    }
                )
                response_content = response_langchain
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
            sources_json = json.dumps(retrieved_sources) if retrieved_sources else None
            save_message(conv_id_from_request, "bot", response_content, is_rag_response=use_rag_processing, source_documents=sources_json)
        elif session_key and len(session_key) == 36 and session_key.count('-') == 4:
            current_app.logger.info(f"Conversation éphémère (ID: {session_key}), messages non sauvegardés en base de données.")
        else:
            current_app.logger.info(f"Conversation éphémère (ID: {conv_id_from_request}), messages non sauvegardées en base de données.")


        return jsonify({'response': response_content, 'conversation_id': session_key, 'sources': retrieved_sources})

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
    return jsonify([
        {
            'sender': msg.sender, 
            'content': msg.content, 
            'is_rag_response': msg.is_rag_response,
            'source_documents': json.loads(msg.source_documents) if msg.source_documents else []
        } 
        for msg in messages
    ])

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