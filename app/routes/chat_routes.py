from flask import render_template, request, jsonify
from app import app, db
from app.services.conversation_service import load_conversation_history

from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Variables globales pour les chaînes LangChain et prompts (initialisées par initialize_chains())
# Nous aurons besoin de deux prompts RAG différents: un strict et un pour le fallback
rag_strict_prompt = None
rag_fallback_prompt = None # Renommé pour plus de clarté
document_chain = None # Utilisera un des prompts RAG
general_llm_chain = None

# Cette variable globale est utilisée pour suivre l'ID de la conversation active.
current_conversation_id = None

# Fonction pour initialiser les chaînes LangChain et les prompts
def initialize_chains():
    global rag_strict_prompt, rag_fallback_prompt, document_chain, general_llm_chain

    from app.services.llm_service import get_chat_llm
    chat_llm_instance = get_chat_llm()
    if chat_llm_instance is None:
        app.logger.error("Erreur: Le chat LLM n'est pas initialisé via get_chat_llm() dans initialize_chains.")
        raise RuntimeError("Chat LLM not initialized for chain creation.")

    # 1. Prompt pour le mode "RAG uniquement" (strict)
    rag_strict_prompt = ChatPromptTemplate.from_messages([
        ("system", "En te basant STRICTEMENT et UNIQUEMENT sur le CONTEXTE fourni, réponds à la question de l'utilisateur de manière concise. Si la réponse n'est PAS dans le CONTEXTE, dis CLAIREMENT 'Je ne trouve pas cette information dans les documents fournis.' Ne fabrique pas de réponses."),
        ("system", "Contexte: {context}"),
        ("system", "Historique de la conversation: {chat_history}"),
        ("user", "{input}")
    ])

    # 2. Prompt pour le mode "Général (LLM seul ou RAG avec fallback)"
    # Ce prompt peut être le même que le strict, car la logique de fallback sera gérée par le code Python.
    # Alternativement, on pourrait le rendre légèrement moins restrictif si le fallback est souhaité directement par le prompt.
    # Pour l'instant, nous gardons la même formulation, la distinction sera dans la logique Python.
    rag_fallback_prompt = ChatPromptTemplate.from_messages([
        ("system", "En te basant sur le CONTEXTE fourni et tes connaissances générales, réponds à la question de l'utilisateur de manière précise. Si le contexte ne contient pas l'information principale, utilise tes connaissances générales. Ne fabrique pas de réponses qui ne seraient basées sur rien."), # Légèrement modifié pour indiquer l'usage des connaissances générales
        ("system", "Contexte: {context}"),
        ("system", "Historique de la conversation: {chat_history}"),
        ("user", "{input}")
    ])

    # La document_chain utilisera le prompt approprié au moment de la requête
    # Nous ne créons qu'une seule document_chain ici, mais le prompt injecté sera choisi dynamiquement
    document_chain = create_stuff_documents_chain(chat_llm_instance, rag_fallback_prompt) # Initialisation avec un prompt par défaut

    # Chaîne pour le LLM général (sans RAG)
    general_llm_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Vous êtes un assistant IA expert en développement, en codage, et en configurations Linux. Répondez aux questions de manière précise et concise, fournissez des exemples de code si nécessaire. Si vous ne trouvez pas la réponse, dites simplement que vous ne savez pas."),
        ("system", "{chat_history}"),
        ("user", "{input}")
    ])
    general_llm_chain = general_llm_prompt_template | chat_llm_instance

    app.logger.info("Chaînes LangChain (prompts RAG et général_llm_chain) initialisées.")


# Route pour la page d'accueil de l'application
@app.route('/')
def index():
    from app.models import Conversation
    conversations = Conversation.query.order_by(Conversation.timestamp.desc()).all()
    return render_template('index.html', conversations=conversations)

# Route principale pour le traitement des messages du chat
@app.route('/chat', methods=['POST'])
def chat():
    global current_conversation_id
    user_message = request.json.get('message')
    conv_id_from_request = request.json.get('conversation_id')
    ephemeral_history_from_frontend = request.json.get('ephemeral_history', [])
    rag_mode = request.json.get('rag_mode', 'fallback_rag') # Récupère le mode RAG, 'fallback_rag' par défaut

    if conv_id_from_request == "new_ephemeral":
        current_conversation_id = None
    elif conv_id_from_request:
        current_conversation_id = int(conv_id_from_request)

    if not user_message:
        return jsonify({'response': 'Aucun message fourni.'}), 400

    try:
        response_content = ""

        chat_history_list = []
        if current_conversation_id:
            chat_history_list = load_conversation_history(current_conversation_id)
        else:
            for msg in ephemeral_history_from_frontend:
                sender = msg.get('sender') if isinstance(msg, dict) and 'sender' in msg and msg['sender'] is not None else None
                content = msg.get('content') if isinstance(msg, dict) and 'content' in msg and msg['content'] is not None else None
                if sender == 'user' and content is not None:
                    chat_history_list.append(HumanMessage(content=content))
                elif sender == 'bot' and content is not None:
                    chat_history_list.append(AIMessage(content=content))

        chat_llm_instance_for_memory = app.extensions["llm_service"]["chat_llm"]
        if chat_llm_instance_for_memory is None:
            app.logger.error("Chat LLM non disponible dans app.extensions.")
            return jsonify({'response': "Erreur interne: Chat LLM non disponible."}), 500

        temp_memory = ConversationBufferMemory(
            llm=chat_llm_instance_for_memory, # type: ignore [reportCallIssue]
            memory_key="chat_history",
            return_messages=True
        )
        temp_memory.chat_memory.messages = chat_history_list

        current_vectorstore = app.extensions["rag_service"]["vectorstore"]
        current_retriever = app.extensions["rag_service"]["retriever"]

        # ********** NOUVELLE LOGIQUE DE BASCULEMENT DU MODE RAG **********
        # Si le RAG est initialisé et actif
        if current_vectorstore and current_retriever:
            app.logger.info(f"DEBUG RAG: Question utilisateur: '{user_message}' (Mode RAG: {rag_mode})")
            print(f"PRINT DEBUG RAG: Question utilisateur: '{user_message}' (Mode RAG: {rag_mode})")

            retrieved_docs = current_retriever.invoke(user_message)

            app.logger.info(f"DEBUG RAG: Nombre de documents récupérés: {len(retrieved_docs)}")
            print(f"PRINT DEBUG RAG: Nombre de documents récupérés: {len(retrieved_docs)}")

            # Choisir le prompt RAG basé sur le mode sélectionné
            selected_rag_prompt = rag_strict_prompt if rag_mode == 'strict_rag' else rag_fallback_prompt
            
            # Recréer la document_chain avec le prompt sélectionné
            chat_llm_instance_for_chain = app.extensions["llm_service"]["chat_llm"]
            dynamic_document_chain = create_stuff_documents_chain(chat_llm_instance_for_chain, selected_rag_prompt) # type: ignore [reportCallIssue]


            if len(retrieved_docs) > 0:
                print("PRINT DEBUG RAG: Documents pertinents trouvés. Utilisation du RAG.")
                app.logger.info("DEBUG RAG: Utilisation du RAG.")
                rag_chain = create_retrieval_chain(current_retriever, dynamic_document_chain) # Utilise la chaîne dynamique

                response_langchain = rag_chain.invoke({
                    "input": user_message,
                    "chat_history": temp_memory.load_memory_variables({})["chat_history"]
                })
                response_content = response_langchain["answer"]

            else: # Aucun document pertinent trouvé
                if rag_mode == 'strict_rag':
                    response_content = "Je ne trouve pas cette information dans les documents fournis."
                    app.logger.info("DEBUG RAG: Mode RAG strict activé. Aucun document trouvé, réponse générique fournie.")
                    print("PRINT DEBUG RAG: Mode RAG strict activé. Aucun document trouvé, réponse générique fournie.")
                else: # fallback_rag ou autre mode par défaut
                    print("PRINT DEBUG RAG: Aucun document pertinent trouvé. Basculement sur les connaissances générales du LLM.")
                    app.logger.info("DEBUG RAG: Basculement sur les connaissances générales du LLM.")

                    response_langchain = general_llm_chain.invoke({ # type: ignore [reportCallIssue, reportOptionalMemberAccess]
                        "input": user_message,
                        "chat_history": temp_memory.load_memory_variables({})["chat_history"]
                    })
                    response_content = response_langchain.content # type: ignore [reportAttributeAccessIssue]

        else: # Ce bloc s'exécute si le RAG n'est PAS initialisé du tout au démarrage de l'app
            app.logger.info("INFO: RAG non actif ou initialisation échouée. Utilisation du LLM général par default.")
            print("PRINT DEBUG RAG: RAG inactif. Utilisation du LLM général par default.")

            response_langchain = general_llm_chain.invoke({ # type: ignore [reportCallIssue, reportOptionalMemberAccess]
                "input": user_message,
                "chat_history": temp_memory.load_memory_variables({})["chat_history"]
            })
            response_content = response_langchain.content # type: ignore [reportAttributeAccessIssue]
        # ********** FIN DE LA NOUVELLE LOGIQUE DE BASCULEMENT DU MODE RAG **********


        from app.services.conversation_service import save_message
        if current_conversation_id:
            save_message(current_conversation_id, "user", user_message) # type: ignore [reportCallIssue]
            save_message(current_conversation_id, "bot", response_content) # type: ignore [reportCallIssue]

        return jsonify({'response': response_content})

    except Exception as e:
        app.logger.error(f"Erreur lors du traitement du chat: {e}")
        print(f"PRINT ERROR: Erreur lors du traitement du chat: {e}")
        return jsonify({'response': f"Désolé, une erreur est survenue lors de la communication avec l'IA ou la base de données. Détails : {str(e)}."}), 500

# Récupère et renvoie la liste de toutes les conversations persistantes
@app.route('/conversations', methods=['GET'])
def get_conversations():
    from app.models import Conversation
    conversations = Conversation.query.order_by(Conversation.timestamp.desc()).all()
    return jsonify([{'id': conv.id, 'name': conv.name} for conv in conversations])

# Récupère et renvoie les messages d'une conversation spécifique
@app.route('/conversations/<int:conv_id>', methods=['GET'])
def get_conversation_messages(conv_id):
    from app.models import Message
    messages = Message.query.filter_by(conversation_id=conv_id).order_by(Message.timestamp).all()
    return jsonify([{'sender': msg.sender, 'content': msg.content} for msg in messages])

# Crée une nouvelle conversation persistante
@app.route('/conversations', methods=['POST'])
def create_conversation():
    from app.models import Conversation
    data = request.json
    name = data.get('name')
    if not name:
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
        return jsonify({'error': f'Erreur lors de la création de la conversation: {e}'}), 500

# Supprime une conversation persistante et tous ses messages associés
@app.route('/conversations/<int:conv_id>', methods=['DELETE'])
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