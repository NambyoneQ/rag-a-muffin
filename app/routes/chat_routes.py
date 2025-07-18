from flask import render_template, request, jsonify
from app import app, db
from app.services.llm_service import get_chat_llm # Importe la fonction pour obtenir le LLM de chat
# Importe les fonctions pour obtenir le vectorstore et le retriever
from app.services.rag_service import get_vectorstore, get_retriever 
from app.services.conversation_service import load_conversation_history, save_message

from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Variables globales pour les chaînes LangChain et prompts (initialisées par initialize_chains())
rag_prompt = None
document_chain = None
general_llm_chain = None

current_conversation_id = None

def initialize_chains():
    global rag_prompt, document_chain, general_llm_chain

    chat_llm_instance = get_chat_llm()
    if chat_llm_instance is None:
        app.logger.error("Erreur: Le chat LLM n'est pas initialisé via get_chat_llm() dans initialize_chains.")
        raise RuntimeError("Chat LLM not initialized for chain creation.")

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "En te basant STRICTEMENT et UNIQUEMENT sur le CONTEXTE fourni, réponds à la question de l'utilisateur de manière concise. Si la réponse n'est PAS dans le CONTEXTE, dis CLAIREMENT 'Je ne trouve pas cette information dans les documents fournis.' Ne fabrique pas de réponses."),
        ("system", "Contexte: {context}"),
        ("system", "Historique de la conversation: {chat_history}"),
        ("user", "{input}")
    ])

    document_chain = create_stuff_documents_chain(chat_llm_instance, rag_prompt) # type: ignore [reportCallIssue]

    general_llm_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Vous êtes un assistant IA expert en développement, en codage, et en configurations Linux. Répondez aux questions de manière précise et concise, fournissez des exemples de code si nécessaire. Si vous ne trouvez pas la réponse, dites simplement que vous ne savez pas."),
        ("system", "{chat_history}"),
        ("user", "{input}")
    ])
    general_llm_chain = general_llm_prompt_template | chat_llm_instance # type: ignore [reportCallIssue]

    app.logger.info("Chaînes LangChain (document_chain, general_llm_chain) initialisées.")

@app.route('/')
def index():
    from app.models import Conversation 
    conversations = Conversation.query.order_by(Conversation.timestamp.desc()).all()
    return render_template('index.html', conversations=conversations)

@app.route('/chat', methods=['POST'])
def chat():
    global current_conversation_id 
    user_message = request.json.get('message')
    conv_id_from_request = request.json.get('conversation_id')
    ephemeral_history_from_frontend = request.json.get('ephemeral_history', []) # type: ignore [reportOptionalMemberAccess]

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
                sender = msg.get('sender') if isinstance(msg, dict) and msg.get('sender') is not None else None # type: ignore [reportOptionalMemberAccess]
                content = msg.get('content') if isinstance(msg, dict) and msg.get('content') is not None else None # type: ignore [reportOptionalMemberAccess]
                if sender == 'user' and content is not None:
                    chat_history_list.append(HumanMessage(content=content))
                elif sender == 'bot' and content is not None:
                    chat_history_list.append(AIMessage(content=content))

        chat_llm_instance_for_memory = get_chat_llm() # Utilise le getter pour la mémoire
        if chat_llm_instance_for_memory is None:
            raise RuntimeError("Chat LLM n'est pas initialisé pour la mémoire. Vérifiez le démarrage de l'app.")

        temp_memory = ConversationBufferMemory(
            llm=chat_llm_instance_for_memory, # type: ignore [reportCallIssue]
            memory_key="chat_history",
            return_messages=True
        )
        temp_memory.chat_memory.messages = chat_history_list

        # Obtient les instances de vectorstore et retriever via les fonctions getter
        current_vectorstore = get_vectorstore()
        current_retriever = get_retriever()

        if current_vectorstore and current_retriever: # Vérifie si les instances sont valides
            app.logger.info(f"DEBUG RAG: Question utilisateur: '{user_message}'")
            print(f"PRINT DEBUG RAG: Question utilisateur: '{user_message}'")

            retrieved_docs = current_retriever.invoke(user_message) # Utilise l'instance obtenue

            app.logger.info(f"DEBUG RAG: Nombre de documents récupérés: {len(retrieved_docs)}")
            print(f"PRINT DEBUG RAG: Nombre de documents récupérés: {len(retrieved_docs)}")

            for i, doc in enumerate(retrieved_docs):
                source_info = doc.metadata.get('source', 'N/A') if doc.metadata is not None else 'N/A' # type: ignore [reportOptionalMemberAccess]
                app.logger.info(f"  Doc {i+1} (Source: {source_info}): '{doc.page_content[:300]}...'")
                print(f"PRINT DEBUG RAG:   Doc {i+1} (Source: {source_info}): '{doc.page_content[:300]}...'")

            if len(retrieved_docs) > 0:
                print("PRINT DEBUG RAG: Documents pertinents trouvés. Utilisation du RAG.")
                app.logger.info("DEBUG RAG: Utilisation du RAG.")
                rag_chain = create_retrieval_chain(current_retriever, document_chain) # Utilise l'instance obtenue

                response_langchain = rag_chain.invoke({
                    "input": user_message,
                    "chat_history": temp_memory.load_memory_variables({})["chat_history"]
                })
                response_content = response_langchain["answer"]

            else:
                print("PRINT DEBUG RAG: Aucun document pertinent trouvé. Basculement sur les connaissances générales du LLM.")
                app.logger.info("DEBUG RAG: Basculement sur les connaissances générales du LLM.")

                response_langchain = general_llm_chain.invoke({ # general_llm_chain est global
                    "input": user_message,
                    "chat_history": temp_memory.load_memory_variables({})["chat_history"]
                })
                response_content = response_langchain["content"] # type: ignore [reportAttributeAccessIssue]

        else: # Ce bloc s'exécute si le RAG n'est pas initialisé (vectorstore ou retriever sont None)
            app.logger.info("INFO: RAG non actif ou initialisation échouée. Utilisation du LLM général par default.")
            print("PRINT DEBUG RAG: RAG inactif. Utilisation du LLM général par default.")

            response_langchain = general_llm_chain.invoke({ # general_llm_chain est global
                "input": user_message,
                "chat_history": temp_memory.load_memory_variables({})["chat_history"]
            })
            response_content = response_langchain["content"] # type: ignore [reportAttributeAccessIssue]

        if current_conversation_id:
            save_message(current_conversation_id, "user", user_message) # type: ignore [reportCallIssue]
            save_message(current_conversation_id, "bot", response_content) # type: ignore [reportCallIssue]

        return jsonify({'response': response_content})

    except Exception as e:
        app.logger.error(f"Erreur lors du traitement du chat: {e}")
        print(f"PRINT ERROR: Erreur lors du traitement du chat: {e}")
        return jsonify({'response': f"Désolé, une erreur est survenue lors de la communication avec l'IA ou la base de données. Détails : {str(e)}."}), 500