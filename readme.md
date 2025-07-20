Local RAG Chatbot (Work in Progress)
This project is a Flask-based web application designed to demonstrate and implement a local Retrieval-Augmented Generation (RAG) chatbot. It allows users to interact with a Large Language Model (LLM) powered by LM Studio, augmented with a knowledge base for contextual responses.

Table of Contents
Features

Work in Progress / Future Enhancements

Prerequisites

Setup Instructions

1. Clone the Repository

2. Install Dependencies

3. Configure Environment Variables

4. Set up LM Studio

5. Prepare the Knowledge Base

6. Run the Application

Usage

Project Structure

Contributing

License

Features
This application currently provides the following core functionalities:

Local LLM Integration (LM Studio): Connects to a locally running LM Studio instance to power both chat completions and embeddings, ensuring data privacy and offline capability.

Conversational History Management:

Persistent Conversations: Users can create, name, and save up to 3 persistent chat conversations, with all messages stored in a PostgreSQL database.

Ephemeral Conversations: Supports temporary, client-side-only chat sessions for quick interactions without database storage.

History Loading & Saving: Seamlessly loads past conversation history and saves new messages for persistent chats.

Retrieval-Augmented Generation (RAG):

Local Knowledge Base: Utilizes a kb_documents directory to store .txt and .pdf files as a source of external knowledge.

ChromaDB Vector Store: Employs ChromaDB for efficient storage and retrieval of document embeddings, providing contextual information to the LLM.

Incremental Knowledge Base Updates: Automatically detects and processes new, modified, and deleted documents in the kb_documents folder on application startup, keeping the vector store synchronized and optimized.

Intelligent Fallback: If no relevant documents are found by the RAG system for a query, the chatbot intelligently falls back to using the LLM's general knowledge.

Flask Web Interface: A basic web UI built with Flask allows users to interact with the chatbot, manage conversations, and view chat history.

Work in Progress / Future Enhancements
This project is under active development. Planned features and improvements include:

Streaming Responses: Implement real-time, token-by-token streaming of LLM responses to the frontend for a more dynamic user experience.

Improved UI/UX: Enhance the chat interface, add markdown rendering for bot responses, and improve overall responsiveness.

Error Handling & User Feedback: Provide more detailed and user-friendly error messages directly in the UI.

Configuration Options in UI: Allow users to configure LLM parameters (e.g., temperature, model name) via the web interface.

Multi-User Support: Implement user authentication and separation of conversations for multiple users.

More Document Types: Expand support for additional document formats (e.g., Markdown, DOCX).

Advanced RAG Techniques: Explore techniques like query rewriting, re-ranking, or hybrid search.

Prerequisites
Before running this application, ensure you have the following installed:

Python 3.8+

pip (Python package installer)

Git

PostgreSQL: A running PostgreSQL server for conversation history persistence.

LM Studio: A running instance of LM Studio with your desired LLM and embedding model loaded.

Setup Instructions

1. Clone the Repository
   Bash

git clone https://github.com/NambyoneQ/rag-a-muffin.git
cd rag-a-muffin 2. Install Dependencies
Create a virtual environment (recommended) and install the required Python packages.

Bash

python -m venv venv

# On Windows:

.\venv\Scripts\activate

# On macOS/Linux:

source venv/bin/activate

pip install -r requirements.txt
(You'll need to create a requirements.txt file if you haven't already. You can generate one using pip freeze > requirements.txt after installing all your project's dependencies).

3. Configure Environment Variables
   Create a .env file in the root directory of your project (where run.py is located) and populate it with your database and LM Studio configuration.

Extrait de code

# .env

SECRET_KEY="your_super_secret_key_here"

# PostgreSQL Database Configuration

DB_USER="dev_user"
DB_PASSWORD="your_db_password"
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="mon_premier_rag_db"

# LM Studio Configuration

LMSTUDIO_UNIFIED_API_BASE="http://localhost:1234/v1" # Default LM Studio API endpoint
LMSTUDIO_API_KEY="lm-studio" # Default LM Studio API key (can be anything)
Ensure your PostgreSQL database (mon_premier_rag_db) exists and the dev_user has the necessary permissions.

4. Set up LM Studio
   Download and Install LM Studio: If you haven't already, download LM Studio from lmstudio.ai.

Download Models:

Chat Model: Download a suitable chat model (e.g., Llama-3.1-8B-UltraLong-4M-Instruct-Q4_K_M as specified in llm_service.py) and load it onto the server in LM Studio.

Embedding Model: Download an embedding model (e.g., text-embedding-nomic-embed-text-v1.5@f32 as specified in llm_service.py) and ensure it's loaded and serving embeddings.

Start Local Server: Make sure the LM Studio local server is running on http://localhost:1234 (or whatever port you configured). The application will connect to the /v1 endpoint.

5. Prepare the Knowledge Base
   Create a folder named kb_documents in the root of your project. Place your .txt or .pdf files inside this directory. On application startup, these documents will be indexed into the ChromaDB vector store.

your_project_root/
├── app/
├── kb_documents/
│ ├── document1.txt
│ └── another_doc.pdf
├── .env
├── run.py
├── requirements.txt
└── ... 6. Run the Application
Bash

python run.py
The application will start, perform initial database setup, and index your knowledge base documents. Open your web browser and navigate to http://127.0.0.1:5000.

Usage
Ephemeral Conversation: Click "Nouvelle Conversation Éphémère" to start a new temporary chat session.

Persistent Conversation: Click "Créer Nouvelle Conversation", enter a name, and start chatting. Your conversation will be saved and listed on the left sidebar.

Switch Conversations: Click on a conversation name in the sidebar to load its history.

Delete Conversation: Click the "X" button next to a persistent conversation to delete it (along with all its messages).

Interacting: Type your message in the input field and press Enter or click "Envoyer".

Project Structure
.
├── app/
│ ├── **init**.py # Flask app initialization, service registration
│ ├── models.py # SQLAlchemy database models (Conversation, Message, DocumentStatus)
│ ├── routes/
│ │ └── chat_routes.py # Flask routes for chat and conversation management
│ └── services/
│ ├── **init**.py
│ ├── conversation_service.py # Logic for loading/saving chat history
│ ├── llm_service.py # LLM (ChatOpenAI) and Embeddings (LMStudioCustomEmbeddings) initialization
│ └── rag_service.py # ChromaDB, document loading, splitting, and incremental indexing
├── config.py # Application configuration (DB, LM Studio, RAG paths)
├── kb_documents/ # Your knowledge base documents (.txt, .pdf)
├── chroma_db/ # Persistent directory for ChromaDB vector store (created by app)
├── templates/
│ └── index.html # Frontend HTML for the chat interface
├── .env.example # Example environment variables file
├── .gitignore # Files ignored by Git
├── requirements.txt # Python dependencies
└── run.py # Entry point to run the Flask application
Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests.

License
MIT License (or specify your chosen license)
