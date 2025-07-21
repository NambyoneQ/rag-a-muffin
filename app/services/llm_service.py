import os
from flask import current_app
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from typing import List
import requests
import logging

# Variables globales pour les instances de LLMs (initialisées à None, car elles seront remplies par initialize_llms)
_chat_llm_instance = None
_embeddings_llm_instance = None

# --- Classe d'embeddings personnalisée pour interagir avec l'API d'embeddings de LM Studio ---
class LMStudioCustomEmbeddings(Embeddings):
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        try:
            test_url = f"{self.base_url}/models"
            response = requests.get(test_url, headers=self.headers, timeout=5)
            response.raise_for_status()
            current_app.logger.info(f"LMStudioCustomEmbeddings: Connexion réussie à {self.base_url}")
        except requests.exceptions.RequestException as e:
            current_app.logger.error(f"LMStudioCustomEmbeddings: Erreur de connexion à {self.base_url}: {e}")
            current_app.logger.error("Assurez-vous que votre instance LM Studio pour les embeddings est démarrée sur le port correct (1234).")
            raise ConnectionError(f"Impossible de se connecter au serveur d'embeddings LM Studio: {e}")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        payload = {
            "input": texts,
            "model": "text-embedding-nomic-embed-text-v1.5@f32" # Nom du modèle d'embeddings
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
            current_app.logger.info(f"LMStudioCustomEmbeddings: Erreur HTTP lors de l'embedding: {e}")
            current_app.logger.info(f"Réponse serveur: {e.response.text if e.response else 'N/A'}")
            raise ConnectionError(f"Erreur HTTP de l'API embeddings LM Studio: {e}")
        except requests.exceptions.RequestException as e:
            current_app.logger.error(f"LMStudioCustomEmbeddings: Erreur de connexion/timeout lors de l'embedding: {e}")
            raise ConnectionError(f"Erreur de connexion à l'API embeddings LM Studio: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(self._embed(batch))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

# --- Fonction pour initialiser les instances de LLM (appelée une seule fois au démarrage) ---
def initialize_llms():
    global _chat_llm_instance, _embeddings_llm_instance

    _chat_llm_instance = ChatOpenAI(
        base_url=current_app.config['LMSTUDIO_UNIFIED_API_BASE'],
        api_key=current_app.config['LMSTUDIO_API_KEY'],
        model="llama-3.1-8b-ultralong-4m-instruct",
        temperature=0.4,
    )

    _embeddings_llm_instance = LMStudioCustomEmbeddings(
        base_url=current_app.config['LMSTUDIO_UNIFIED_API_BASE'],
        api_key=current_app.config['LMSTUDIO_API_KEY']
    )
    current_app.logger.info("LLMs (chat_llm et embeddings_llm) initialisés.")

# --- Fonctions pour accéder aux instances de LLM après leur initialisation ---
# Ces fonctions sont utilisées par d'autres modules pour obtenir les instances de LLM.
def get_chat_llm():
    if _chat_llm_instance is None:
        raise RuntimeError("Chat LLM n'a pas été initialisé. Appelez initialize_llms() au démarrage de l'application.")
    return _chat_llm_instance

def get_embeddings_llm():
    if _embeddings_llm_instance is None:
        raise RuntimeError("Embeddings LLM n'a pas été initialisé. Appelez initialize_llms() au démarrage de l'application.")
    return _embeddings_llm_instance