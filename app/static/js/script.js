// app/static/js/script.js

let currentConversationId = "new_ephemeral"; // ID de la conversation active ("new_ephemeral" pour éphémère)
let ephemeralChatHistory = []; // Historique de la conversation éphémère (stocké côté client)

// Récupération des éléments du DOM
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const loadingIndicator = document.getElementById('loading-indicator');
const conversationList = document.getElementById('conversation-list');
const newConvBtn = document.getElementById('new-conv-btn');
const ephemeralConvBtn = document.getElementById('ephemeral-conv-btn');

// Fonction pour afficher un message dans la boîte de chat
function displayMessage(sender, content) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
    messageDiv.textContent = content; // Affichage direct du texte
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight; // Fait défiler la boîte de chat vers le bas

    // Met à jour l'historique éphémère si la conversation active est éphémère
    if (currentConversationId === "new_ephemeral") {
        ephemeralChatHistory.push({ sender: sender, content: content });
    }
}

// Charge et affiche les messages d'une conversation spécifique (persistante ou éphémère)
async function loadConversation(convId) {
    currentConversationId = convId;
    chatBox.innerHTML = ''; // Vide la boîte de chat actuelle

    // Met à jour l'état visuel du bouton de conversation actif dans la barre latérale
    document.querySelectorAll('#conversation-list button').forEach(btn => btn.classList.remove('active'));
    ephemeralConvBtn.classList.remove('active');
    if (convId === "new_ephemeral") {
        ephemeralConvBtn.classList.add('active');
        ephemeralChatHistory = []; // Réinitialise l'historique éphémère lors de la sélection
    } else {
        const activeBtn = document.querySelector(`#conversation-list button[data-id="${convId}"]`);
        if (activeBtn) activeBtn.classList.add('active');
    }

    if (convId !== "new_ephemeral") {
        // Pour les conversations persistantes, charge les messages depuis le backend
        loadingIndicator.style.display = 'block';
        try {
            const response = await fetch(`/conversations/${convId}`);
            const messages = await response.json();
            messages.forEach(msg => displayMessage(msg.sender, msg.content));
        } catch (error) {
            console.error('Erreur lors du chargement de la conversation:', error);
            displayMessage('bot', 'Erreur lors du chargement de cette conversation.');
        } finally {
            loadingIndicator.style.display = 'none';
        }
    } else {
        // Pour les conversations éphémères, re-affiche l'historique stocké côté client
        ephemeralChatHistory.forEach(msg => displayMessage(msg.sender, msg.content));
    }
}

// Récupère et affiche la liste des conversations persistantes depuis le backend
async function fetchConversations() {
    try {
        const response = await fetch('/conversations');
        const conversations = await response.json();
        conversationList.innerHTML = ''; // Vide la liste actuelle
        // Ajoute chaque conversation à la liste dans la barre latérale
        conversations.forEach(conv => {
            const li = document.createElement('li');
            const button = document.createElement('button');
            button.textContent = conv.name;
            button.dataset.id = conv.id;
            button.onclick = () => loadConversation(conv.id); // Charge la conversation au clic

            const deleteBtn = document.createElement('button');
            deleteBtn.classList.add('delete-btn');
            deleteBtn.textContent = 'X';
            deleteBtn.onclick = async (e) => { // Gère la suppression d'une conversation
                e.stopPropagation(); // Empêche le clic de se propager au bouton parent
                if (confirm(`Voulez-vous vraiment supprimer la conversation "${conv.name}" ?`)) {
                    await deleteConversation(conv.id);
                }
            };
            button.appendChild(deleteBtn);
            li.appendChild(button);
            conversationList.appendChild(li);
        });
    } catch (error) {
        console.error('Erreur lors du chargement des conversations:', error);
    }
}

// Gère la création d'une nouvelle conversation persistante
newConvBtn.onclick = async () => {
    const convName = prompt('Nom de la nouvelle conversation (max 3 conversations persistantes) :');
    if (convName) {
        try {
            const response = await fetch('/conversations', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: convName })
            });
            const data = await response.json();
            if (response.ok) {
                await fetchConversations(); // Met à jour la liste des conversations
                loadConversation(data.id); // Charge la nouvelle conversation
                displayMessage('bot', `Conversation "${convName}" créée et activée.`);
            } else {
                alert('Erreur: ' + (data.error || 'Impossible de créer la conversation.'));
            }
        } catch (error) {
            console.error('Erreur lors de la création de la conversation:', error);
            alert('Erreur de connexion lors de la création de la conversation.');
        }
    }
};

// Gère le basculement vers une nouvelle conversation éphémère
ephemeralConvBtn.onclick = () => {
    loadConversation("new_ephemeral");
    displayMessage('bot', 'Nouvelle conversation éphémère activée.');
};

// Gère la suppression d'une conversation persistante
async function deleteConversation(convId) {
    try {
        const response = await fetch(`/conversations/${convId}`, { method: 'DELETE' });
        if (response.ok) {
            await fetchConversations(); // Met à jour la liste des conversations
            if (currentConversationId === convId) {
                loadConversation("new_ephemeral"); // Bascule vers l'éphémère si la conversation active est supprimée
                displayMessage('bot', 'Conversation supprimée. Basculement vers conversation éphémère.');
            } else {
                displayMessage('bot', 'Conversation supprimée.');
            }
        } else {
            const data = await response.json();
            alert('Erreur lors de la suppression: ' + (data.error || 'Inconnu.'));
        }
    } catch (error) {
            console.error('Erreur lors de la suppression de la conversation:', error);
            alert('Erreur de connexion lors de la suppression.');
    }
}

// Gère l'envoi du message de l'utilisateur au backend
sendButton.onclick = async () => {
    const message = userInput.value.trim();
    if (message === '') return;

    displayMessage('user', message); // Affiche le message utilisateur dans le chat
    userInput.value = ''; // Vide la zone de saisie
    sendButton.disabled = true; // Désactive le bouton Envoyer pendant le traitement
    loadingIndicator.style.display = 'block'; // Affiche l'indicateur de chargement

    let historyToSend = [];
    if (currentConversationId === "new_ephemeral") {
        // Pour les conversations éphémères, envoie l'historique complet (y compris le message utilisateur actuel) au backend
        historyToSend = ephemeralChatHistory.map(msg => ({
            sender: msg.sender,
            content: msg.content
        }));
    }

    const ragMode = document.querySelector('input[name="rag_mode"]:checked').value; // type: ignore [reportPropertyAccessIssue]

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                conversation_id: currentConversationId,
                ephemeral_history: historyToSend,
                rag_mode: ragMode // Ajoute le mode RAG à la requête
            })
        });

        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }

        const data = await response.json();
        displayMessage('bot', data.response);

        if (currentConversationId === "new_ephemeral") {
            ephemeralChatHistory.push({ sender: 'bot', content: data.response });
        }

    } catch (error) {
        console.error('Erreur lors de l\'envoi du message:', error);
        displayMessage('bot', 'Désolé, une erreur est survenue lors de la communication.');
        if (currentConversationId === "new_ephemeral") {
            ephemeralChatHistory.push({ sender: 'bot', content: 'Erreur.' });
        }
    } finally {
        sendButton.disabled = false;
        loadingIndicator.style.display = 'none';
    }
};

userInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendButton.click();
    }
});

document.addEventListener('DOMContentLoaded', () => {
    fetchConversations();
    loadConversation("new_ephemeral");
});