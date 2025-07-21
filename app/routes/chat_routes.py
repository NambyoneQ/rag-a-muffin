# app/routes/chat_routes.py

import os
import uuid 
from flask import render_template, request, jsonify, Blueprint, current_app 

from app import app, db 
from app.services.conversation_service import load_conversation_history, save_message 
from langchain_openai import ChatOpenAI 

from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Créez une instance de Blueprint
chat_bp = Blueprint('chat_bp', __name__, template_folder='../templates', static_folder='../static')

# Variables globales pour les chaînes LangChain et prompts
rag_strict_prompt = None
rag_fallback_prompt = None
code_analysis_prompt = None
general_llm_chain = None
general_llm_prompt_template = None # Déclarée ici pour être globalement accessible

_llm_instances_by_session_id = {} # Dictionnaire pour stocker les instances de LLM par ID de session


# Cette fonction sera appelée par __init__.py après que tous les services soient prêts
def initialize_chains_with_app(app_instance):
    global rag_strict_prompt, rag_fallback_prompt, code_analysis_prompt, general_llm_chain, general_llm_prompt_template # Déclare toutes les variables globales

    chat_llm_instance_global = app_instance.extensions["llm_service"]["chat_llm"] # L'instance LLM globale par défaut
    if chat_llm_instance_global is None:
        app_instance.logger.error("Erreur: Le chat LLM n'est pas initialisé via app.extensions.")
        raise RuntimeError("Chat LLM not initialized for chain creation.")

    _llm_instances_by_session_id['default'] = chat_llm_instance_global

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
    
    # DÉFINITION DE general_llm_prompt_template
    general_llm_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Vous êtes un assistant IA expert en développement, en codage, et en configurations Linux. Répondez aux questions de manière précise et concise, fournissez des exemples de code si nécessaire. Si vous ne trouvez pas la réponse, dites simplement que vous ne savez pas."),
        ("system", "{chat_history}"),
        ("user", "{input}")
    ])
    # INITIALISATION DE general_llm_chain (Ligne 65 dans l'erreur Pylance)
    general_llm_chain = general_llm_prompt_template | chat_llm_instance_global

    app_instance.logger.info("Chaînes LangChain (prompts RAG et général_llm_chain) initialisées.")


# Définition des routes du Blueprint
@chat_bp.route('/')
def index():
    from app.models import Conversation 
    conversations = Conversation.query.order_by(Conversation.timestamp.desc()).all()
    project_names = []
    # Accédez à la configuration via l'instance 'app' qui est importée globalement
    code_base_dir = app.config.get('CODE_BASE_DIR') 
    if code_base_dir and os.path.exists(code_base_dir):
        project_names = [d for d in os.listdir(code_base_dir) if os.path.isdir(os.path.join(code_base_dir, d))]
    
    return render_template('index.html', conversations=conversations, project_names=project_names)


# Route pour lister toutes les conversations (GET)
@chat_bp.route('/conversations', methods=['GET'])
def get_all_conversations():
    from app.models import Conversation
    conversations = Conversation.query.order_by(Conversation.timestamp.desc()).all()
    return jsonify([{'id': conv.id, 'name': conv.name} for conv in conversations])


# Route pour la création de conversation (POST)
# C'EST L'UNIQUE DÉFINITION DE create_conversation (Ligne 94 dans l'erreur Pylance)
@chat_bp.route('/conversations', methods=['POST'])
def create_conversation():
    from app.models import Conversation
    data = request.get_json(silent=True) 
    # Vérification robuste si data est None ou n'est pas un dictionnaire
    if not isinstance(data, dict):
        app.logger.error(f"Requête JSON manquante ou malformée pour créer conversation. Data reçue: {data}")
        return jsonify({'error': 'Requête invalide: JSON manquant ou malformé.'}), 400

    name = data.get('name') # Ligne 108 dans l'erreur Pylance si existante
    if not name:
        app.logger.error("Nom de conversation manquant dans la requête JSON.")
        return jsonify({'error': 'Le nom de la conversation est requis.'}), 400

    existing_conversations_count = Conversation.query.count()
    if existing_conversations_count >= 3:
        return jsonify({'error': 'Vous ne pouvez pas créer plus de 3 conversations persistantes.'}), 400

    new_conv = Conversation(name=name) # type: ignore [reportCallIssue]
    try:
        db.session.add(new_conv)
        db.session.commit()
        return jsonify({'id': new_conv.id, 'name': new_conv.name}), 201
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Erreur lors de la création de la conversation: {e}")
        return jsonify({'error': f'Erreur lors de la création de lahela conversation: {e}'}), 500


@chat_bp.route('/chat', methods=['POST'])
def chat():
    # current_conversation_id est géré localement dans la fonction.
    request_data = request.get_json(silent=True)
    # Vérification robuste si request_data est None ou n'est pas un dictionnaire
    if not isinstance(request_data, dict):
        app.logger.error(f"Requête JSON manquante ou malformée pour chat. Data reçue: {request_data}")
        return jsonify({'response': "Erreur: La requête n'est pas au format JSON valide."}), 400

    user_message = request_data.get('message', '')
    conv_id_from_request = request_data.get('conversation_id')
    ephemeral_history_from_frontend = request_data.get('ephemeral_history', [])
    rag_mode = request_data.get('rag_mode', 'fallback_rag')
    selected_project = request_data.get('selected_project', None)
    strict_mode = request_data.get('strict_mode', False)

    # Déterminez le chat_llm_instance à utiliser pour cette requête
    chat_llm_instance_for_current_request = None
    final_chat_history_for_llm = []
    
    # Déterminez la clé de session unique pour cette conversation (pour l'API du LLM)
    session_key = "default_session" # Clé par défaut, sera surchargée

    if conv_id_from_request == "new_ephemeral":
        # Générer un UUID unique pour la nouvelle conversation éphémère
        session_key = str(uuid.uuid4())
        app.logger.info(f"Mode 'Nouvelle Conversation Éphémère': Nouvelle session_key générée: {session_key}")
        final_chat_history_for_llm = [] 
        
        chat_llm_instance_for_current_request = ChatOpenAI(
            base_url=current_app.config['LMSTUDIO_UNIFIED_API_BASE'], 
            api_key=current_app.config['LMSTUDIO_API_KEY'], 
            model=current_app.config['LMSTUDIO_CHAT_MODEL'], 
            temperature=0.4,
            user=session_key
        )
        _llm_instances_by_session_id[session_key] = chat_llm_instance_for_current_request

    elif conv_id_from_request:
        try:
            conversation_id_int = int(conv_id_from_request)
            session_key = str(conversation_id_int) 
            app.logger.info(f"Conversation persistante ID: {conversation_id_int}. Chargement de l'historique.")
            final_chat_history_for_llm = load_conversation_history(conversation_id_int)
            
            if session_key not in _llm_instances_by_session_id:
                chat_llm_instance_for_current_request = ChatOpenAI(
                    base_url=current_app.config['LMSTUDIO_UNIFIED_API_BASE'], 
                    api_key=current_app.config['LMSTUDIO_API_KEY'], 
                    model=current_app.config['LMSTUDIO_CHAT_MODEL'], 
                    temperature=0.4,
                    user=session_key
                )
                _llm_instances_by_session_id[session_key] = chat_llm_instance_for_current_request
            else:
                chat_llm_instance_for_current_request = _llm_instances_by_session_id[session_key]
                
        except ValueError:
            app.logger.error(f"ID de conversation invalide reçu: {conv_id_from_request}. Traitement comme éphémère vide.")
            session_key = str(uuid.uuid4())
            final_chat_history_for_llm = []
            chat_llm_instance_for_current_request = ChatOpenAI(
                base_url=current_app.config['LMSTUDIO_UNIFIED_API_BASE'], 
                api_key=current_app.config['LMSTUDIO_API_KEY'], 
                model=current_app.config['LMSTUDIO_CHAT_MODEL'], 
                temperature=0.4,
                user=session_key
            )
            _llm_instances_by_session_id[session_key] = chat_llm_instance_for_current_request

    else: 
        session_key = 'default_ephemeral_session'
        app.logger.info("Conv_ID manquant ou invalide. Traitement comme éphémère avec historique frontend.")
        
        if session_key not in _llm_instances_by_session_id:
             chat_llm_instance_for_current_request = ChatOpenAI(
                base_url=current_app.config['LMSTUDIO_UNIFIED_API_BASE'], 
                api_key=current_app.config['LMSTUDIO_API_KEY'], 
                model=current_app.config['LMSTUDIO_CHAT_MODEL'], 
                temperature=0.4,
                user=session_key
            )
             _llm_instances_by_session_id[session_key] = chat_llm_instance_for_current_request
        else:
            chat_llm_instance_for_current_request = _llm_instances_by_session_id[session_key]

        for msg in (ephemeral_history_from_frontend if isinstance(ephemeral_history_from_frontend, list) else []):
            if isinstance(msg, dict):
                sender = msg.get('sender')
                content = msg.get('content')
                if sender == 'user' and content is not None:
                    final_chat_history_for_llm.append(HumanMessage(content=content))
                elif sender == 'bot' and content is not None:
                    final_chat_history_for_llm.append(AIMessage(content=content))
            else:
                app.logger.warning(f"Élément inattendu dans ephemeral_history_from_frontend: {msg}. Ignoré.")


    if chat_llm_instance_for_current_request is None:
        app.logger.error("Chat LLM non disponible après initialisation conditionnelle.")
        return jsonify({'response': "Erreur interne: Chat LLM non disponible."}), 500

    response_content = "Désolé, une erreur inattendue est survenue lors de la génération de la réponse."

    try:
        temp_memory = ConversationBufferMemory(
            llm=chat_llm_instance_for_current_request, 
            memory_key="chat_history",
            return_messages=True
        )
        temp_memory.chat_memory.messages = final_chat_history_for_llm 

        current_vectorstore = app.extensions["rag_service"]["vectorstore"]
        current_retriever = app.extensions["rag_service"]["retriever"]

        use_rag_processing = False
        retrieved_docs = []
        selected_prompt = None
        filtered_retriever = None

        if current_vectorstore and current_retriever:
            app.logger.info(f"DEBUG RAG: Question utilisateur: '{user_message}' (Mode: {rag_mode}, Projet: {selected_project}, Strict: {strict_mode}, Session: {session_key})")
            print(f"PRINT DEBUG RAG: Question utilisateur: '{user_message}' (Mode: {rag_mode}, Projet: {selected_project}, Strict: {strict_mode}, Session: {session_key})")

            metadata_filter = {}

            if rag_mode == 'kb_rag':
                metadata_filter = {"file_type": "kb"}
                selected_prompt = rag_strict_prompt if strict_mode else rag_fallback_prompt
                
            elif rag_mode == 'code_rag':
                if selected_project:
                    metadata_filter = {
                        "$and": [
                            {"file_type": {"$eq": "code"}},
                            {"project_name": {"$eq": selected_project}}
                        ]
                    }
                    selected_prompt = code_analysis_prompt
                else:
                    response_content = "Veuillez sélectionner un projet pour le mode d'analyse de code."
                    app.logger.warning("Mode analyse de code sélectionné sans projet.")
                    print("PRINT DEBUG: Mode analyse de code sélectionné sans projet.")
                    return jsonify({'response': response_content})
            
            retriever_to_use = None
            if metadata_filter and rag_mode != 'general':
                try:
                    filtered_retriever = current_vectorstore.as_retriever(search_kwargs={"k": 10, "filter": metadata_filter})
                    retriever_to_use = filtered_retriever
                    app.logger.info(f"DEBUG RAG: Utilisation du retriever filtré avec filtre: {metadata_filter}")
                except Exception as filter_e:
                    app.logger.error(f"Erreur lors de la création du retriever filtré: {filter_e}")
                    import traceback
                    app.logger.error(f"TRACEBACK RETRIEVER FILTER ERROR: \n{traceback.format_exc()}")
                    response_content = "Désolé, il y a eu un problème avec le filtre de documents du RAG. Réessayez."
                    app.logger.info("DEBUG RAG: Fallback suite erreur filtre.")
            elif rag_mode != 'general':
                retriever_to_use = current_retriever
                app.logger.info("DEBUG RAG: Utilisation du retriever global (non filtré).")

            if retriever_to_use:
                try:
                    retrieved_docs = retriever_to_use.invoke(user_message)
                    if len(retrieved_docs) > 0:
                        use_rag_processing = True
                    app.logger.info(f"DEBUG RAG: Documents récupérés via retriever: {len(retrieved_docs)}")
                    print(f"PRINT DEBUG RAG: Documents récupérés via retriever: {len(retrieved_docs)}")
                    for i, doc in enumerate(retrieved_docs):
                        source_info = doc.metadata.get('source', 'N/A') if isinstance(doc.metadata, dict) else 'N/A'
                        app.logger.info(f"  Doc {i+1} (Source: {source_info}): '{doc.page_content[:100]}...'")
                        print(f"PRINT DEBUG RAG:   Doc {i+1} (Source: {source_info}): '{doc.page_content[:100]}...'")
                except Exception as invoke_e:
                    app.logger.error(f"Erreur lors de l'invocation du retriever: {invoke_e}")
                    import traceback
                    app.logger.error(f"TRACEBACK RETRIEVER INVOKE ERROR: \n{traceback.format_exc()}")
                    response_content = "Désolé, une erreur est survenue lors de la recherche de documents pertinents."
                    app.logger.info("DEBUG RAG: Fallback suite erreur invocation retriever.")
            else:
                app.logger.info("DEBUG RAG: Pas de retriever à utiliser pour ce mode ou pas de documents trouvés.")
                print("PRINT DEBUG RAG: Pas de retriever à utiliser pour ce mode ou pas de documents trouvés.")
        
        if use_rag_processing and response_content == "Désolé, une erreur inattendue est survenue lors de la génération de la réponse.":
            print("PRINT DEBUG RAG: Documents pertinents trouvés. Utilisation du RAG.")
            app.logger.info("DEBUG RAG: Utilisation du RAG.")

            chat_llm_instance_for_chain = chat_llm_instance_for_current_request 
            
            if selected_prompt is None:
                selected_prompt = rag_fallback_prompt

            dynamic_document_chain = create_stuff_documents_chain(chat_llm_instance_for_chain, selected_prompt) # type: ignore [reportCallIssue]

            final_rag_chain_retriever = filtered_retriever if filtered_retriever else current_retriever

            chain_inputs = {
                "input": user_message,
                "chat_history": temp_memory.load_memory_variables({})["chat_history"]
            }
            if selected_prompt == code_analysis_prompt and selected_project:
                chain_inputs["project_name"] = selected_project

            rag_chain = create_retrieval_chain(final_rag_chain_retriever, dynamic_document_chain) # type: ignore [reportArgumentTypeIssue]
            
            try:
                response_langchain = rag_chain.invoke(chain_inputs)
                response_content = response_langchain["answer"]
                app.logger.info(f"DEBUG LLM Response (RAG): {response_content[:200]}...")
            except Exception as llm_e:
                app.logger.error(f"Erreur lors de l'invocation de la chaîne LLM (RAG): {llm_e}")
                import traceback
                app.logger.error(f"TRACEBACK LLM CHAIN ERROR (RAG): \n{traceback.format_exc()}")
                response_content = "Désolé, le modèle de langage a rencontré une erreur lors de la génération de la réponse RAG."
                
        elif rag_mode == 'general' and response_content == "Désolé, une erreur inattendue est survenue lors de la génération de la réponse.":
            print("PRINT DEBUG: Mode Général sélectionné. Utilisation du LLM général.")
            app.logger.info("DEBUG: Mode Général sélectionné. Utilisation du LLM général.")
            
            try:
                response_langchain = general_llm_chain.invoke({ # type: ignore [reportCallIssue, reportOptionalMemberAccess]
                    "input": user_message,
                    "chat_history": temp_memory.load_memory_variables({})["chat_history"]
                })
                response_content = response_langchain.content # type: ignore [reportAttributeAccessIssue]
                app.logger.info(f"DEBUG LLM Response (General): {response_content[:200]}...")
            except Exception as llm_e:
                app.logger.error(f"Erreur lors de l'invocation de la chaîne LLM générale: {llm_e}")
                import traceback
                app.logger.error(f"TRACEBACK GENERAL LLM CHAIN ERROR: \n{traceback.format_exc()}")
                response_content = "Désolé, le modèle de langage général a rencontré une erreur."


        elif response_content == "Désolé, une erreur inattendue est survenue lors de la génération de la réponse.":
            if rag_mode == 'kb_rag' and strict_mode:
                response_content = "Je ne trouve pas cette information dans les documents fournis."
                app.logger.info("DEBUG RAG: Mode RAG strict sur KB activé. Aucun document récupéré, réponse générique fournie.")
                print("PRINT DEBUG RAG: Mode RAG strict sur KB activé. Aucun document récupéré, réponse générique fournie.")
            elif rag_mode == 'code_rag' and selected_project:
                 response_content = f"Je ne trouve pas cette information dans le code source du projet '{selected_project}'."
                 app.logger.info(f"DEBUG RAG: Mode analyse de code activé. Aucun document récupéré pour le projet '{selected_project}', réponse générique fournie.")
                 print(f"PRINT DEBUG RAG: Mode analyse de code activé. Aucun document récupéré pour le projet '{selected_project}', réponse générique fournie.")
            else:
                print("PRINT DEBUG RAG: RAG inactif ou aucun document pertinent trouvé. Basculement sur les connaissances générales du LLM.")
                app.logger.info("DEBUG RAG: RAG inactif ou aucun document pertinent trouvé. Utilisation du LLM général par default.")

                try:
                    response_langchain = general_llm_chain.invoke({ # type: ignore [reportCallIssue, reportOptionalMemberAccess]
                        "input": user_message,
                        "chat_history": temp_memory.load_memory_variables({})["chat_history"]
                    })
                    response_content = response_langchain.content # type: ignore [reportAttributeAccessIssue]
                    app.logger.info(f"DEBUG LLM Response (Fallback): {response_content[:200]}...")
                except Exception as llm_e:
                    app.logger.error(f"Erreur lors de l'invocation de la chaîne LLM (Fallback): {llm_e}")
                    import traceback
                    app.logger.error(f"TRACEBACK FALLBACK LLM CHAIN ERROR: \n{traceback.format_exc()}")
                    response_content = "Désolé, le modèle de langage a rencontré une erreur lors du basculement."


        if not isinstance(response_content, str):
            app.logger.error(f"response_content est de type inattendu: {type(response_content)}. Valeur: {response_content}. Forçage à une erreur générique.")
            response_content = "Désolé, une erreur interne est survenue lors de la génération de la réponse (format inattendu)."

        # Logique de sauvegarde des messages
        if isinstance(conv_id_from_request, int): # Utilisez conv_id_from_request pour la sauvegarde
            save_message(conv_id_from_request, "user", user_message)
            save_message(conv_id_from_request, "bot", response_content)
        else:
            app.logger.info("Conversation éphémère, messages non sauvegardés en base de données.")


        return jsonify({'response': response_content})

    except Exception as e:
        app.logger.error(f"Erreur fatale lors du traitement du chat: {e}")
        import traceback
        app.logger.error(f"TRACEBACK COMPLET: \n{traceback.format_exc()}")
        print(f"PRINT ERROR: Erreur fatale lors du traitement du chat: {e}")
        print(f"PRINT ERROR: TRACEBACK COMPLET: \n{traceback.format_exc()}")
        return jsonify({'response': f"Désolé, une erreur inattendue et fatale est survenue. Détails : {str(e)}."}), 500

# Récupère et renvoie les messages d'une conversation spécifique
@chat_bp.route('/conversations/<int:conv_id>', methods=['GET'])
def get_conversation_messages(conv_id):
    from app.models import Message
    messages = Message.query.filter_by(conversation_id=conv_id).order_by(Message.timestamp).all()
    return jsonify([{'sender': msg.sender, 'content': msg.content} for msg in messages])

# Supprime une conversation persistante et tous les messages associés
@chat_bp.route('/conversations/<int:conv_id>', methods=['DELETE'])
def delete_conversation(conv_id):
    from app.models import Conversation
    conv = Conversation.query.get(conv_id)
    if not conv:
        return jsonify({'error': 'Conversation non trouvée.'}), 404
    try:
        db.session.delete(conv)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Conversation supprimée.'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Erreur lors de la suppression de la conversation: {e}'}), 500