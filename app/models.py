# app/models.py
from app import db # Importe l'instance db depuis app/__init__.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship, Mapped, mapped_column 
import datetime 
from typing import Optional # NEW: Import Optional

# Modèle pour représenter une conversation persistante dans la base de données
class Conversation(db.Model):
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    # Relation : une conversation peut avoir plusieurs messages associés
    messages: Mapped[list["Message"]] = relationship('Message', backref='conversation', lazy=True, cascade="all, delete-orphan")
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=db.func.now())

    # Ajout d'un __init__ explicite pour satisfaire Pylance
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return f'<Conversation {self.id}: {self.name}>'

# Modèle pour stocker chaque message (utilisateur ou bot) d'une conversation
class Message(db.Model):
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Clé étrangère pour lier le message à une conversation spécifique
    conversation_id: Mapped[int] = mapped_column(Integer, ForeignKey('conversation.id'), nullable=False)
    sender: Mapped[str] = mapped_column(String(10), nullable=False) # Expéditeur du message: 'user' ou 'bot'
    content: Mapped[str] = mapped_column(Text, nullable=False) # Contenu textuel du message
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=db.func.now())
    is_rag_response: Mapped[bool] = mapped_column(Boolean, default=False)
    source_documents: Mapped[Optional[str]] = mapped_column(Text, nullable=True) # Stores JSON string of sources

    # Ajout d'un __init__ explicite pour satisfaire Pylance
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return f'<Message {self.id} (Conv:{self.conversation_id}): {self.content}>'

# Modèle pour stocker l'état et les métadonnées des documents indexés (pour l'update incrémentale du RAG)
class DocumentStatus(db.Model):
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_path: Mapped[str] = mapped_column(String(1024), unique=True, nullable=False, index=True) # Chemin absolu du fichier
    file_type: Mapped[str] = mapped_column(String(50), nullable=False) # 'kb' or 'code'
    last_modified: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False) # Date de dernière modification du fichier sur le disque
    indexed_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=db.func.now(), onupdate=db.func.now()) # Date d'indexation/mise à jour dans ChromaDB
    status: Mapped[str] = mapped_column(String(50), default='pending', nullable=False) # 'indexed', 'error', 'deleted', 'skipped'
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True) # Message d'erreur si l'indexation a échoué
    file_hash: Mapped[Optional[str]] = mapped_column(String(32), nullable=True) # Hash du fichier pour vérification rapide

    # Ajout d'un __init__ explicite pour satisfaire Pylance
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return f'<DocumentStatus {self.file_path} - {self.status}>'