from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from app import db 
from app.models import Message, Conversation 

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