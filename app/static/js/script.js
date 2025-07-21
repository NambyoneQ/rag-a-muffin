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

// NOUVEAUX ÉLÉMENTS DU DOM POUR LE MODE DE RECHERCHE
const searchModeRadios = document.querySelectorAll('input[name="search_mode"]');
const projectSelectionDiv = document.getElementById('project-selection');
const projectSelect = document.getElementById('project-select');
const strictModeCheckbox = document.getElementById('strict-mode-checkbox');


// Fonction pour afficher un message dans la boîte de chat AVEC RENDU MARKDOWN
function displayMessage(sender, content) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(sender === 'user' ? 'user-message' : 'bot-message');

    // Pour les messages de l'utilisateur, afficher le texte brut.
    // Pour les messages du bot, utiliser marked.js pour rendre le Markdown en HTML.
    if (sender === 'user') {
        messageDiv.textContent = content;
    } else {
        // S'assurer que marked est disponible globalement (chargé par le script dans index.html)
        if (typeof marked !== 'undefined') {
            messageDiv.innerHTML = marked.parse(content);
        } else {
            messageDiv.textContent = content; // Fallback si marked.js n'est pas chargé
            console.warn("Marked.js non disponible. Le rendu Markdown ne sera pas effectué.");
        }
    }

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    // Mise à jour de l'historique éphémère si la conversation active est éphémère
    // Cette logique est cruciale : ephemeralChatHistory n'est rempli que pour les sessions éphémères
    // et il est vidé lors du changement de conversation (voir loadConversation).
    if (currentConversationId === "new_ephemeral") {
        ephemeralChatHistory.push({ sender: sender, content: content });
    }
}

// Charge et affiche les messages d'une conversation spécifique (persistante ou éphémère)
async function loadConversation(convId) {
    // 1. Mise à jour de l'ID de la conversation active
    currentConversationId = convId;

    // 2. Vider la boîte de chat visible
    chatBox.innerHTML = '';

    // 3. Mise à jour de l'état visuel du bouton de conversation actif dans la barre latérale
    document.querySelectorAll('#conversation-list button').forEach(btn => btn.classList.remove('active'));
    if (ephemeralConvBtn) ephemeralConvBtn.classList.remove('active');
    
    // 4. Gestion de l'historique interne : Vider ephemeralChatHistory AVANT de potentiellement le remplir ou le laisser vide
    ephemeralChatHistory = []; 

    if (convId === "new_ephemeral") {
        // Si c'est une nouvelle conversation éphémère
        if (ephemeralConvBtn) ephemeralConvBtn.classList.add('active');
        displayMessage('bot', 'Nouvelle conversation éphémère activée.');
    } else {
        // Si c'est une conversation persistante, activer le bouton correspondant
        const activeBtn = document.querySelector(`#conversation-list button[data-id="${convId}"]`);
        if (activeBtn) activeBtn.classList.add('active');

        // Charger les messages de la conversation persistante depuis le backend
        loadingIndicator.style.display = 'block';
        try {
            const response = await fetch(`/conversations/${convId}`);
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }
            const messages = await response.json();
            // Afficher les messages chargés. Ils ne sont PAS ajoutés à ephemeralChatHistory ici.
            // ephemeralChatHistory reste vide pour les conversations persistantes.
            messages.forEach(msg => displayMessage(msg.sender, msg.content));
        } catch (error) {
            console.error('Erreur lors du chargement de la conversation:', error);
            displayMessage('bot', 'Erreur lors du chargement de cette conversation.');
        } finally {
            loadingIndicator.style.display = 'none';
        }
    }
}

// Récupère et affiche la liste des conversations persistantes depuis le backend
async function fetchConversations() {
    try {
        const response = await fetch('/conversations'); // Requête GET sur /conversations
        if (!response.ok) { // Vérifie si la réponse HTTP est OK (200-299)
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        const conversations = await response.json();
        console.log("Conversations fetched:", conversations); // Log pour le débogage

        conversationList.innerHTML = ''; // Vide la liste actuelle
        conversations.forEach(conv => {
            const li = document.createElement('li');
            const button = document.createElement('button');
            button.textContent = conv.name;
            button.dataset.id = conv.id;
            button.onclick = () => loadConversation(conv.id);

            const deleteBtn = document.createElement('button');
            deleteBtn.classList.add('delete-btn');
            deleteBtn.textContent = 'X';
            deleteBtn.onclick = async (e) => {
                e.stopPropagation();
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
        // Afficher un message à l'utilisateur si aucune conversation ne peut être chargée
        const errorItem = document.createElement('li');
        errorItem.textContent = "Erreur de chargement des conversations.";
        errorItem.style.color = "red";
        conversationList.appendChild(errorItem);
    }
}

// Gère la création d'une nouvelle conversation persistante
if (newConvBtn) {
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
                    // displayMessage('bot', `Conversation "${convName}" créée et activée.`); // Déjà appelé par loadConversation
                } else {
                    alert('Erreur: ' + (data.error || 'Impossible de créer la conversation.'));
                }
            } catch (error) {
                console.error('Erreur lors de la création de la conversation:', error);
                alert('Erreur de connexion lors de la création de la conversation.');
            }
        }
    };
}

// Gère le basculement vers une nouvelle conversation éphémère
if (ephemeralConvBtn) {
    ephemeralConvBtn.onclick = () => {
        loadConversation("new_ephemeral"); // Appelle loadConversation pour gérer le reset
    };
}

// Gère la suppression d'une conversation persistante
async function deleteConversation(convId) {
    try {
        const response = await fetch(`/conversations/${convId}`, { method: 'DELETE' });
        if (response.ok) {
            await fetchConversations();
            if (currentConversationId === convId) {
                loadConversation("new_ephemeral"); // Bascule vers l'éphémère si la conversation active est supprimée
            } else {
                displayMessage('bot', 'Conversation supprimée.'); // Message si la conversation supprimée n'était pas active
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

// NOUVELLE LOGIQUE POUR L'AFFICHAGE DU SÉLECTEUR DE PROJET
searchModeRadios.forEach(radio => {
    radio.addEventListener('change', () => {
        if (radio.value === 'code_rag') {
            if (projectSelectionDiv) projectSelectionDiv.style.display = 'block';
        } else {
            if (projectSelectionDiv) projectSelectionDiv.style.display = 'none';
            if (projectSelect) projectSelect.value = '';
        }
        
        const parentOfStrictModeCheckbox = strictModeCheckbox ? strictModeCheckbox.parentNode : null;
        if (parentOfStrictModeCheckbox) {
            if (radio.value === 'kb_rag' || radio.value === 'code_rag') {
                parentOfStrictModeCheckbox.style.display = 'block';
            } else {
                parentOfStrictModeCheckbox.style.display = 'none';
                if (strictModeCheckbox) strictModeCheckbox.checked = false; 
            }
        }
    });
});


// Gère l'envoi du message de l'utilisateur au backend
sendButton.onclick = async () => {
    const message = userInput.value.trim();
    if (message === '') return;

    displayMessage('user', message);
    userInput.value = '';
    sendButton.disabled = true;
    loadingIndicator.style.display = 'block';

    let historyToSend = [];
    // N'envoyer l'historique que si currentConversationId est "new_ephemeral" (null côté backend)
    // Pour les persistantes, le backend le chargera lui-même de la BDD
    if (currentConversationId === "new_ephemeral") { 
        historyToSend = ephemeralChatHistory.map(msg => ({
            sender: msg.sender,
            content: msg.content
        }));
        console.log("Historique éphémère envoyé au backend:", historyToSend); 
    } else {
        console.log("Conversation persistante, l'historique sera géré par le backend.");
    }
    
    const searchMode = document.querySelector('input[name="search_mode"]:checked')?.value || 'general';
    let selectedProject = null;
    if (searchMode === 'code_rag') {
        if (projectSelect) {
            selectedProject = projectSelect.value;
            if (!selectedProject) {
                displayMessage('bot', 'Veuillez sélectionner un projet pour l\'analyse de code.');
                sendButton.disabled = false;
                loadingIndicator.style.display = 'none';
                return;
            }
        } else {
            displayMessage('bot', 'Erreur: Sélecteur de projet non trouvé.');
            sendButton.disabled = false;
            loadingIndicator.style.display = 'none';
            return;
        }
    }

    let ragModeParam;
    if (searchMode === 'general') {
        ragModeParam = 'general';
    } else if (searchMode === 'kb_rag') {
        ragModeParam = strictModeCheckbox && strictModeCheckbox.checked ? 'strict_rag' : 'fallback_rag';
    } else if (searchMode === 'code_rag') {
        ragModeParam = 'code_rag';
    } else {
        ragModeParam = 'fallback_rag';
    }

    try {
        let requestBody = {
            message: message,
            // conversation_id sera null pour l'éphémère, ou l'ID pour le persistant
            conversation_id: currentConversationId, 
            rag_mode: ragModeParam,
            selected_project: selectedProject,
            strict_mode: strictModeCheckbox ? strictModeCheckbox.checked : false 
        };

        // Seulement ajouter ephemeral_history au body si c'est une conversation éphémère (currentConversationId est "new_ephemeral")
        if (currentConversationId === "new_ephemeral") {
            requestBody.ephemeral_history = historyToSend;
        }

        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody) 
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Erreur HTTP:', response.status, errorText);
            throw new Error(`Erreur HTTP: ${response.status} - ${errorText}`);
        }

        const rawResponseText = await response.text();
        console.log("Réponse brute du serveur:", rawResponseText);

        let data;
        try {
            data = JSON.parse(rawResponseText);
        } catch (jsonError) {
            console.error("Erreur de parsing JSON:", jsonError);
            console.error("Contenu qui a échoué au parsing:", rawResponseText);
            displayMessage('bot', 'Désolé, la réponse du serveur n\'est pas au format attendu. Veuillez vérifier la console pour plus de détails.');
            throw new Error("Réponse serveur non-JSON.");
        }

        displayMessage('bot', data.response);

        // Seulement ajouter la réponse du bot à ephemeralChatHistory si c'est une conversation éphémère
        if (currentConversationId === "new_ephemeral") {
            ephemeralChatHistory.push({ sender: 'bot', content: data.response });
        }

    } catch (error) {
        console.error('Erreur lors de l\'envoi du message:', error);
        displayMessage('bot', `Désolé, une erreur est survenue lors de la communication. Détails : ${error.message || error}.`);
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
    // Au démarrage de l'application, activer la conversation éphémère par défaut,
    // ce qui va aussi vider l'historique éphémère.
    loadConversation("new_ephemeral"); 

    // Initialiser l'état d'affichage du sélecteur de projet et de la checkbox strict
    const initialSearchMode = document.querySelector('input[name="search_mode"]:checked')?.value || 'general';

    if (projectSelectionDiv) {
        if (initialSearchMode === 'code_rag') {
            projectSelectionDiv.style.display = 'block';
        } else {
            projectSelectionDiv.style.display = 'none';
        }
    }

    const parentOfStrictModeCheckbox = strictModeCheckbox ? strictModeCheckbox.parentNode : null;
    if (parentOfStrictModeCheckbox) {
        if (initialSearchMode === 'kb_rag' || initialSearchMode === 'code_rag') {
            parentOfStrictModeCheckbox.style.display = 'block';
        } else {
            parentOfStrictModeCheckbox.style.display = 'none';
            if (strictModeCheckbox) strictModeCheckbox.checked = false;
        }
    }
});