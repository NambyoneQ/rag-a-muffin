from app import db # Importe l'instance db depuis app/__init__.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship

# Modèle pour représenter une conversation persistante dans la base de données
class Conversation(db.Model):
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    # Relation : une conversation peut avoir plusieurs messages associés
    messages = relationship('Message', backref='conversation', lazy=True, cascade="all, delete-orphan")
    timestamp = Column(DateTime, default=db.func.now())

    def __repr__(self):
        return f'<Conversation {self.id}: {self.name}>'

# Modèle pour stocker chaque message (utilisateur ou bot) d'une conversation
class Message(db.Model):
    id = Column(Integer, primary_key=True)
    # Clé étrangère pour lier le message à une conversation spécifique
    conversation_id = Column(Integer, ForeignKey('conversation.id'), nullable=False)
    sender = Column(String(10), nullable=False) # Expéditeur du message: 'user' ou 'bot'
    content = Column(Text, nullable=False) # Contenu textuel du message
    timestamp = Column(DateTime, default=db.func.now())

    def __repr__(self):
        return f'<Message {self.id} (Conv:{self.conversation_id}): {self.content}>'

# Modèle pour stocker l'état et les métadonnées des documents indexés (pour l'update incrémentale du RAG)
class DocumentStatus(db.Model):
    id = Column(Integer, primary_key=True)
    file_path = Column(String(500), unique=True, nullable=False) # Chemin absolu du fichier
    last_modified = Column(DateTime, nullable=False) # Date de dernière modification du fichier sur le disque
    indexed_at = Column(DateTime, default=db.func.now()) # Date d'indexation dans ChromaDB

    def __repr__(self):
        return f'<DocumentStatus {self.file_path} (Mod:{self.last_modified})>'