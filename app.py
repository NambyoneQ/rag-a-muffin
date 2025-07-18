import os
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List
import logging
import json
import re
import urllib.parse
from langchain_core.embeddings import Embeddings # Pour la classe de base Embeddings
from typing import List # Pour les annotations de type (List)
import requests # Pour les requêtes HTTP directes (utilisé par LMStudioCustomEmbeddings)

from dotenv import load_dotenv

# Message de confirmation affiché au démarrage de l'application Flask
print("--- FLASK APP STARTED (RAG with Fallback - Version Stable) ---")

# Charge les variables d'environnement depuis un fichier .env (si présent)
load_dotenv()

# Importe la configuration de l'application depuis le module config.py
from config import Config

# Initialisation de l'application Flask
app = Flask(__name__)
app.config.from_object(Config)

# Configure le niveau de log de l'application Flask pour afficher les messages INFO et supérieurs
app.logger.setLevel(logging.INFO)
if not app.logger.handlers:
    handler = logging.StreamHandler()
    app.logger.addHandler(handler)


# Initialisation de l'extension SQLAlchemy pour la base de données
db = SQLAlchemy(app)

# --- Définition des modèles de base de données ---

# Modèle pour représenter une conversation persistante dans la base de données
class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    # Relation : une conversation a plusieurs messages
    messages = db.relationship('Message', backref='conversation', lazy=True, cascade="all, delete-orphan")
    timestamp = db.Column(db.DateTime, default=db.func.now())

    def __repr__(self):
        return f'<Conversation {self.id}: {self.name}>'

# Modèle pour stocker chaque message (utilisateur ou bot) d'une conversation
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Clé étrangère pour lier le message à une conversation spécifique
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False) # type: ignore [reportCallIssue]
    sender = db.Column(db.String(10), nullable=False) # Expéditeur du message: 'user' ou 'bot'
    content = db.Column(db.Text, nullable=False) # Contenu textuel du message
    timestamp = db.Column(db.DateTime, default=db.func.now())

    def __repr__(self):
        return f'<Message {self.id} (Conv:{self.conversation_id}): {self.content}>'

# --- Classe d'embeddings personnalisée pour interagir avec l'API d'embeddings de LM Studio ---
class LMStudioCustomEmbeddings(Embeddings):
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        # Teste la connexion à l'API d'embeddings de LM Studio au démarrage de l'application
        try:
            test_url = f"{self.base_url}/models"
            response = requests.get(test_url, headers=self.headers, timeout=5)
            response.raise_for_status()
            app.logger.info(f"LMStudioCustomEmbeddings: Connexion réussie à {self.base_url}")
        except requests.exceptions.RequestException as e:
            app.logger.error(f"LMStudioCustomEmbeddings: Erreur de connexion à {self.base_url}: {e}")
            app.logger.error("Assurez-vous que votre instance LM Studio pour les embeddings est démarrée sur le port correct (1234).")
            raise ConnectionError(f"Impossible de se connecter au serveur d'embeddings LM Studio: {e}")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # Envoie un lot de textes à l'API d'embeddings de LM Studio pour obtenir leurs représentations vectorielles
        url = f"{self.base_url}/embeddings"
        payload = {
            "input": texts,
            "model": "text-embedding-nomic-embed-text-v1.5@f32" # <-- NOM EXACT DU MODÈLE ICI
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status() 
            
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                embeddings = [item["embedding"] for item in data["data"]]
                return embeddings
            else:
                raise ValueError(f"Réponse invalide de l'API embeddings: {data}")
        except requests.exceptions.HTTPError as e:
            app.logger.info(f"LMStudioCustomEmbeddings: Erreur HTTP lors de l'embedding: {e}")
            app.logger.info(f"Réponse serveur: {e.response.text if e.response else 'N/A'}")
            raise ConnectionError(f"Erreur HTTP de l'API embeddings LM Studio: {e}")
        except requests.exceptions.RequestException as e:
            app.logger.error(f"LMStudioCustomEmbeddings: Erreur de connexion/timeout lors de l'embedding: {e}")
            raise ConnectionError(f"Erreur de connexion à l'API embeddings LM Studio: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Méthode requise par LangChain pour l'embedding de multiples documents
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(self._embed(batch))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        # Méthode requise par LangChain pour l'embedding d'une seule requête/question
        return self._embed([text])[0]


# --- Configuration et initialisation du LLM (chat) et du Modèle d'Embeddings (RAG) ---

# LLM principal pour le chat (Llama 3 via LM Studio sur port 1234)
chat_llm = ChatOpenAI(
    base_url=app.config['LMSTUDIO_UNIFIED_API_BASE'], # Utilise l'API unifiée de LM Studio
    api_key=app.config['LMSTUDIO_API_KEY'], # La même clé API
    model="Llama-3.1-8B-UltraLong-4M-Instruct-Q4_K_M", # Nom exact du modèle de chat chargé dans LM Studio
    temperature=0.4,
)

# Modèle d'Embeddings (Nomic Embed Text via LM Studio sur port 1234)
embeddings_llm = LMStudioCustomEmbeddings(
    base_url=app.config['LMSTUDIO_UNIFIED_API_BASE'],
    api_key=app.config['LMSTUDIO_API_KEY']
)

# Variable globale pour stocker l'ID de la conversation active (None pour une conversation éphémère)
current_conversation_id = None

# Fonction pour charger l'historique des messages d'une conversation spécifique depuis la base de données
def load_conversation_history(conversation_id: int) -> List[BaseMessage]:
    messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.timestamp).all()
    chat_history = []
    for msg in messages:
        # Convertit les messages stockés en objets LangChain (HumanMessage/AIMessage)
        if msg.sender == 'user':
            chat_history.append(HumanMessage(content=msg.content))
        else:
            chat_history.append(AIMessage(content=msg.content))
    return chat_history

# Fonction pour sauvegarder un message (utilisateur ou bot) dans la conversation active de la base de données
def save_message(conversation_id: int, sender: str, content: str): # type: ignore [reportCallIssue]
    new_message = Message(conversation_id=conversation_id, sender=sender, content=content) # type: ignore [reportCallIssue]
    db.session.add(new_message)
    db.session.commit()

# --- Initialisation du Vector Store (ChromaDB) pour le RAG ---
vectorstore = None
retriever = None

# Fonction pour charger les documents et créer/mettre à jour le Vector Store
def initialize_vectorstore():
    global vectorstore, retriever
    app.logger.info(f"Vérification ou création du Vector Store ChromaDB dans '{app.config['CHROMA_PERSIST_DIRECTORY']}'...")

    # Vérifie si un Vector Store persistant existe déjà et le charge
    if os.path.exists(app.config['CHROMA_PERSIST_DIRECTORY']) and len(os.listdir(app.config['CHROMA_PERSIST_DIRECTORY'])) > 0:
        try:
            app.logger.info(f"Chargement du Vector Store existant depuis '{app.config['CHROMA_PERSIST_DIRECTORY']}'")
            vectorstore = Chroma(
                persist_directory=app.config['CHROMA_PERSIST_DIRECTORY'],
                embedding_function=embeddings_llm
            )
            count = vectorstore._collection.count()
            if count > 0:
                app.logger.info(f"Vector Store chargé avec {count} documents.")
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Configure le retriever pour récupérer 5 documents
            else:
                app.logger.info("Le Vector Store existant est vide. Procédure d'ingestion lancée.")
                _process_and_create_vectorstore() # Si vide, procède à l'ingestion
        except Exception as e:
            app.logger.error(f"Erreur lors du chargement du Vector Store existant: {e}. Une nouvelle ingestion sera tentée.")
            _process_and_create_vectorstore() # En cas d'erreur de chargement, ré-ingère
    else:
        app.logger.info(f"Dossier ChromaDB non trouvé ou vide. Initialisation de l'ingestion des documents...")
        _process_and_create_vectorstore() # Si le dossier n'existe pas ou est vide, procède à l'ingestion

# Fonction interne pour le traitement des documents et la création du vector store
def _process_and_create_vectorstore():
    global vectorstore, retriever
    raw_documents = [] # Liste pour stocker le contenu brut de tous les documents chargés
    
    # Crée le dossier des documents de la base de connaissances s'il n'existe pas
    if not os.path.exists(app.config['KNOWLEDGE_BASE_DIR']):
        app.logger.info(f"ATTENTION: Le dossier de base de connaissances '{app.config['KNOWLEDGE_BASE_DIR']}' n'existe pas. Création...")
        os.makedirs(app.config['KNOWLEDGE_BASE_DIR'])
        app.logger.info("Veuillez y placer des documents (fichiers .txt ou .pdf) pour que le RAG fonctionne.")

    # Parcours les fichiers dans le dossier de la base de connaissances pour les charger
    for root, _, files in os.walk(app.config['KNOWLEDGE_BASE_DIR']):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file_path.endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        raw_documents.append(doc)
                    app.logger.info(f"Document texte brut chargé : {file_path}")
                elif file_path.endswith('.pdf'):
                    app.logger.info(f"Document PDF brut chargé : {file_path}")
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        raw_documents.append(doc)
            except Exception as e:
                app.logger.error(f"Erreur lors du chargement brut de {file_path}: {e}")

    # Si aucun document n'a été chargé, le RAG ne peut pas être fonctionnel
    if not raw_documents:
        app.logger.info("Aucun document brut valide trouvé dans le dossier de la base de connaissances. Le RAG ne sera pas fonctionnel pour les questions basées sur vos documents.")
        vectorstore = None
        retriever = None
        return

    # Divise tous les documents bruts en "chunks" (morceaux) pour l'embedding
    # La taille de chunk est ajustée pour que les petits documents entrent en un seul morceau
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(raw_documents)
    app.logger.info(f"Documents divisés en {len(chunks)} chunks.")

    # Crée le Vector Store ChromaDB à partir des chunks et le persiste sur le disque
    try:
        vectorstore = Chroma.from_documents(
            chunks,
            embeddings_llm, # Utilise le modèle d'embeddings personnalisé
            persist_directory=app.config['CHROMA_PERSIST_DIRECTORY']
        )
        vectorstore.persist() # Sauvegarde le vector store sur le disque
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Configure le retriever
        app.logger.info("Vector Store ChromaDB créé et persisté. RAG prêt.")
    except Exception as e:
        app.logger.error(f"Erreur lors de la création ou persistance du Vector Store ChromaDB: {e}")
        app.logger.error("Vérifiez la disponibilité du service LM Studio pour les embeddings sur le port 1234.")
        vectorstore = None
        retriever = None


# --- Chaînes LangChain pour le RAG ---

# Prompt pour la chaîne RAG : guide le LLM à utiliser le contexte fourni
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "En te basant STRICTEMENT et UNIQUEMENT sur le CONTEXTE fourni, réponds à la question de l'utilisateur de manière concise. Si la réponse n'est PAS dans le CONTEXTE, dis CLAIREMENT 'Je ne trouve pas cette information dans les documents fournis.' Ne fabrique pas de réponses."),
    ("system", "Contexte: {context}"),
    ("system", "Historique de la conversation: {chat_history}"), # Inclut l'historique de la conversation
    ("user", "{input}")
])

# Chaîne pour combiner les documents avec le prompt
document_chain = create_stuff_documents_chain(chat_llm, rag_prompt)


# --- Routes de l'application Flask ---

@app.route('/')
def index():
    # Charge les conversations existantes pour les afficher dans la barre latérale du menu
    conversations = Conversation.query.order_by(Conversation.timestamp.desc()).all()
    return render_template('index.html', conversations=conversations)

# Route principale pour le chat (mode non-streaming)
@app.route('/chat', methods=['POST'])
def chat():
    global current_conversation_id
    user_message = request.json.get('message')
    conv_id_from_request = request.json.get('conversation_id')
    # Historique éphémère envoyé depuis le frontend (liste de dicts)
    ephemeral_history_from_frontend = request.json.get('ephemeral_history', []) # type: ignore [reportOptionalMemberAccess]

    # Met à jour l'ID de la conversation active globale pour la session Flask
    if conv_id_from_request == "new_ephemeral":
        current_conversation_id = None # Indique une conversation qui ne sera pas sauvegardée en BDD
    elif conv_id_from_request:
        current_conversation_id = int(conv_id_from_request)

    # Gère le cas où le message utilisateur est vide
    if not user_message:
        return jsonify({'response': 'Aucun message fourni.'}), 400

    try:
        response_content = ""
        
        # Charge l'historique de la conversation : depuis la BDD pour les persistantes, depuis le frontend pour les éphémères
        chat_history_list = []
        if current_conversation_id:
            chat_history_list = load_conversation_history(current_conversation_id)
        else:
            # Convertit l'historique du frontend en objets LangChain (HumanMessage/AIMessage)
            for msg in ephemeral_history_from_frontend:
                # Accède aux clés de dictionnaire de manière sécurisée
                sender = msg.get('sender') if isinstance(msg, dict) else None # type: ignore [reportOptionalMemberAccess]
                content = msg.get('content') if isinstance(msg, dict) else None # type: ignore [reportOptionalMemberAccess]
                if sender == 'user' and content is not None:
                    chat_history_list.append(HumanMessage(content=content))
                elif sender == 'bot' and content is not None:
                    chat_history_list.append(AIMessage(content=content))

        # Crée une instance temporaire de ConversationBufferMemory pour la requête actuelle
        temp_memory = ConversationBufferMemory(
            llm=chat_llm, # type: ignore [reportCallIssue] # Le LLM principal utilisé par la mémoire
            memory_key="chat_history",
            return_messages=True
        )
        # Injecte l'historique chargé dans la mémoire temporaire de LangChain
        temp_memory.chat_memory.messages = chat_history_list

        # Logique de Fallback : Tente le RAG d'abord, puis connaissances générales si le RAG ne trouve rien
        # Vérifie si le Vector Store et le Retriever sont correctement initialisés au démarrage de l'application
        if vectorstore and retriever:
            app.logger.info(f"DEBUG RAG: Question utilisateur: '{user_message}'")
            print(f"PRINT DEBUG RAG: Question utilisateur: '{user_message}'")
            
            # Tente de récupérer les documents pertinents du Vector Store
            retrieved_docs = retriever.invoke(user_message)
            
            app.logger.info(f"DEBUG RAG: Nombre de documents récupérés: {len(retrieved_docs)}")
            print(f"PRINT DEBUG RAG: Nombre de documents récupérés: {len(retrieved_docs)}")
            
            # Affiche les 300 premiers caractères des documents récupérés pour le débogage
            for i, doc in enumerate(retrieved_docs):
                # Vérifie si doc.metadata n'est pas None avant d'essayer d'y accéder
                source_info = doc.metadata.get('source', 'N/A') if doc.metadata is not None else 'N/A' # type: ignore [reportOptionalMemberAccess]
                app.logger.info(f"  Doc {i+1} (Source: {source_info}): '{doc.page_content[:300]}...'")
                print(f"PRINT DEBUG RAG:   Doc {i+1} (Source: {source_info}): '{doc.page_content[:300]}...'")

            # Si des documents pertinents sont trouvés, utilise la chaîne RAG
            if len(retrieved_docs) > 0:
                print("PRINT DEBUG RAG: Documents pertinents trouvés. Utilisation du RAG.")
                app.logger.info("DEBUG RAG: Utilisation du RAG.")
                rag_chain = create_retrieval_chain(retriever, document_chain)
                
                # Appelle la chaîne RAG et attend la réponse complète
                response_langchain = rag_chain.invoke({
                    "input": user_message,
                    "chat_history": temp_memory.load_memory_variables({})["chat_history"]
                })
                response_content = response_langchain["answer"] # Accède directement à la réponse finale
                
            else:
                # Si aucun document pertinent n'est récupéré, bascule sur les connaissances générales du LLM
                print("PRINT DEBUG RAG: Aucun document pertinent trouvé. Basculement sur les connaissances générales du LLM.")
                app.logger.info("DEBUG RAG: Basculement sur les connaissances générales du LLM.")
                
                # Prompt pour le LLM général, moins restrictif sur l'utilisation du contexte RAG
                general_llm_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Vous êtes un assistant IA expert en développement, en codage, et en configurations Linux. Répondez aux questions de manière précise et concise, fournissez des exemples de code si nécessaire. Si vous ne trouvez pas la réponse, dites simplement que vous ne savez pas."),
                    ("system", "{chat_history}"), # Inclut l'historique de chat pour le LLM général
                    ("user", "{input}")
                ])
                
                general_llm_chain = general_llm_prompt | chat_llm # type: ignore [reportCallIssue] # Utilise le LLM pour obtenir la réponse
                
                # Appelle la chaîne du LLM général et attend la réponse complète
                response_langchain = general_llm_chain.invoke({
                    "input": user_message,
                    "chat_history": temp_memory.load_memory_variables({})["chat_history"]
                })
                response_content = response_langchain["content"] # type: ignore [reportIndexIssue] # Accède directement à l'attribut content

        else:
            # Cette partie s'exécute si le RAG n'a pas été initialisé du tout au démarrage de l'application
            app.logger.info("INFO: RAG non actif ou initialisation échouée. Utilisation du LLM général par défaut.")
            print("PRINT DEBUG RAG: RAG inactif. Utilisation du LLM général par défaut.")
            
            general_llm_prompt = ChatPromptTemplate.from_messages([
                ("system", "Vous êtes un assistant IA expert en développement, en codage, et en configurations Linux. Répondez aux questions de manière précise et concise, fournissez des exemples de code si nécessaire. Si vous ne trouvez pas la réponse, dites simplement que vous ne savez pas."),
                ("system", "{chat_history}"),
                ("user", "{input}")
            ])
            
            general_llm_chain = general_llm_prompt | chat_llm # type: ignore [reportCallIssue]
            
            response_langchain = general_llm_chain.invoke({
                "input": user_message,
                "chat_history": temp_memory.load_memory_variables({})["chat_history"]
            })
            response_content = response_langchain["content"] # type: ignore [reportIndexIssue]
        
        # Sauvegarde le message utilisateur et la réponse complète du bot (si conversation persistante)
        if current_conversation_id:
            save_message(current_conversation_id, "user", user_message) # type: ignore [reportCallIssue]
            save_message(current_conversation_id, "bot", response_content) # type: ignore [reportCallIssue]

        return jsonify({'response': response_content}) # Retourne la réponse JSON normale
    
    except Exception as e:
        app.logger.error(f"Erreur lors du traitement du chat: {e}")
        print(f"PRINT ERROR: Erreur lors du traitement du chat: {e}")
        return jsonify({'response': f"Désolé, une erreur est survenue lors de la communication avec l'IA ou la base de données. Détails : {str(e)}."}), 500

# --- Routes pour la gestion des conversations via l'API ---

@app.route('/conversations', methods=['GET'])
def get_conversations():
    # Récupère et renvoie la liste de toutes les conversations persistantes
    conversations = Conversation.query.order_by(Conversation.timestamp.desc()).all()
    return jsonify([{'id': conv.id, 'name': conv.name} for conv in conversations])

@app.route('/conversations/<int:conv_id>', methods=['GET'])
def get_conversation_messages(conv_id):
    # Récupère et renvoie les messages d'une conversation spécifique
    messages = Message.query.filter_by(conversation_id=conv_id).order_by(Message.timestamp).all()
    return jsonify([{'sender': msg.sender, 'content': msg.content} for msg in messages])

@app.route('/conversations', methods=['POST'])
def create_conversation():
    # Crée une nouvelle conversation persistante
    data = request.json
    name = data.get('name')
    if not name:
        return jsonify({'error': 'Le nom de la conversation est requis.'}), 400
    
    # Limite le nombre de conversations persistantes à 3
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

@app.route('/conversations/<int:conv_id>', methods=['DELETE'])
def delete_conversation(conv_id):
    # Supprime une conversation persistante et tous ses messages associés
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

# Point d'entrée principal pour lancer l'application Flask
if __name__ == '__main__':
    with app.app_context():
        # Crée les tables de la base de données si elles n'existent pas
        try:
            db.create_all()
            app.logger.info("Tables de la base de données créées ou déjà existantes.")
        except Exception as e:
            app.logger.error(f"Erreur lors de la création des tables de la base de données: {e}")
            print(f"PRINT ERROR: Erreur lors de la création des tables de la base de données: {e}")
            print("Vérifiez que le serveur PostgreSQL est en cours d'exécution et que l'utilisateur 'dev_user' a les droits 'Create databases' sur la base de données 'mon_premier_rag_db'.")

        # Initialise le vector store (RAG) au démarrage de l'application
        initialize_vectorstore()

    # Lance l'application en mode debug sur le port 5000, SANS RELOADER AUTOMATIQUE (use_reloader=False)
    # L'application sera accessible via Nginx sur http://localhost/
    app.run(debug=True, use_reloader=False, port=5000)