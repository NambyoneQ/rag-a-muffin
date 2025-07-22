// app/static/js/script.js

// currentConversationId sera soit un ID numérique (pour conversation persistante),
// soit un UUID (string) pour une session éphémère continue,
// soit null pour le tout premier chargement/interaction de la page.
let currentConversationId = null; 
let ephemeralChatHistory = []; // Historique de la conversation éphémère (stocké côté client)

// Récupération des éléments du DOM
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input'); // C'est maintenant un textarea
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


// Fonction pour afficher un message dans la boîte de chat AVEC RENDU MARKDOWN ET BOUTON COPIER
function displayMessage(sender, content) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(sender === 'user' ? 'user-message' : 'bot-message');

    if (sender === 'user') {
        messageDiv.textContent = content;
    } else {
        if (typeof marked !== 'undefined') {
            const renderedHtml = marked.parse(content); // Rendre le Markdown
            messageDiv.innerHTML = renderedHtml;

            // NOUVELLE LOGIQUE : Ajouter un bouton copier aux blocs de code
            const codeBlocks = messageDiv.querySelectorAll('pre'); // Trouver tous les blocs <pre>
            codeBlocks.forEach(preBlock => {
                const copyButton = document.createElement('button');
                copyButton.className = 'copy-button'; // Appliquer la classe CSS
                copyButton.textContent = 'Copier';
                copyButton.onclick = () => { // Gestionnaire d'événements au clic
                    const code = preBlock.querySelector('code'); // Trouver le <code> à l'intérieur du <pre>
                    if (code && navigator.clipboard) { // Vérifier si l'API Clipboard est disponible
                        navigator.clipboard.writeText(code.textContent || '').then(() => {
                            copyButton.textContent = 'Copié !'; // Feedback visuel
                            setTimeout(() => {
                                copyButton.textContent = 'Copier';
                            }, 2000); // Réinitialiser le texte après 2 secondes
                        }).catch(err => {
                            console.error('Erreur de copie:', err);
                            copyButton.textContent = 'Erreur';
                        });
                    } else if (code) { // Fallback si l'API Clipboard n'est pas supportée
                        // Méthode de copie plus ancienne (déconseillée)
                        const range = document.createRange();
                        range.selectNodeContents(code);
                        const selection = window.getSelection();
                        if (selection) {
                            selection.removeAllRanges();
                            selection.addRange(range);
                            document.execCommand('copy');
                            selection.removeAllRanges();
                            copyButton.textContent = 'Copié (fallback)!';
                            setTimeout(() => { copyButton.textContent = 'Copier'; }, 2000);
                        }
                    }
                };
                preBlock.appendChild(copyButton); // Ajouter le bouton au bloc <pre>
            });

        } else {
            messageDiv.textContent = content;
            console.warn("Marked.js non disponible. Le rendu Markdown ne sera pas effectué.");
        }
    }

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    // Mise à jour de l'historique éphémère SI la conversation active est bien une session éphémère (string UUID ou null)
    if (typeof currentConversationId === 'string' || currentConversationId === null) {
        ephemeralChatHistory.push({ sender: sender, content: content });
    }
}

// Charge et affiche les messages d'une conversation spécifique (persistante ou éphémère)
async function loadConversation(convId) {
    chatBox.innerHTML = ''; // Vider la boîte de chat visible

    // Mise à jour de l'état visuel du bouton de conversation actif dans la barre latérale
    document.querySelectorAll('#conversation-list button').forEach(btn => btn.classList.remove('active'));
    if (ephemeralConvBtn) ephemeralConvBtn.classList.remove('active');
    
    // 1. Gérer l'ID de la conversation et réinitialiser l'historique interne du frontend
    if (convId === "new_ephemeral_session_request") { 
        // Demande de nouvelle session éphémère via le bouton. Générer un nouvel UUID.
        currentConversationId = crypto.randomUUID(); // Génère un UUID unique (compatible navigateur moderne)
        ephemeralChatHistory = []; // Vider l'historique pour la nouvelle session
        console.log("Nouvelle session éphémère Frontend ID:", currentConversationId);
        if (ephemeralConvBtn) ephemeralConvBtn.classList.add('active');
        displayMessage('bot', `Nouvelle conversation éphémère active (ID: ${currentConversationId.substring(0, 8)}...).`);
    } else {
        // C'est un ID de conversation persistante, ou la toute première requête où currentConversationId est null.
        // Si c'est une persistante, l'ID est numérique. Si c'est null, cela deviendra une session éphémère.
        currentConversationId = convId; // Assigner l'ID passé
        ephemeralChatHistory = []; // Vider l'historique éphémère lors du changement vers/depuis une persistante
        
        // Si l'ID est un ID de conversation persistante (numérique)
        if (typeof currentConversationId === 'number') { 
            const activeBtn = document.querySelector(`#conversation-list button[data-id="${currentConversationId}"]`);
            if (activeBtn) activeBtn.classList.add('active');

            loadingIndicator.style.display = 'block';
            try {
                const response = await fetch(`/conversations/${currentConversationId}`);
                if (!response.ok) {
                    throw new Error(`Erreur HTTP: ${response.status}`);
                }
                const messages = await response.json();
                messages.forEach(msg => displayMessage(msg.sender, msg.content));
            } catch (error) {
                console.error('Erreur lors du chargement de la conversation:', error);
                displayMessage('bot', 'Erreur lors du chargement de cette conversation.');
            } finally {
                loadingIndicator.style.display = 'none';
            }
        } else { // Si c'est une session éphémère (null au départ ou un UUID existant)
            if (ephemeralConvBtn) ephemeralConvBtn.classList.add('active');
            ephemeralChatHistory.forEach(msg => displayMessage(msg.sender, msg.content)); // Réafficher l'historique existant de cette session éphémère
            if (currentConversationId === null) {
                // Si c'est le tout début et que currentConversationId est null, on lui assigne un UUID
                currentConversationId = crypto.randomUUID();
                console.log("Session éphémère initiale auto-générée ID:", currentConversationId);
                displayMessage('bot', `Conversation éphémère active (ID: ${currentConversationId.substring(0, 8)}...).`);
            }
        }
    }
}


// Récupère et affiche la liste des conversations persistantes depuis le backend
async function fetchConversations() {
    try {
        const response = await fetch('/conversations');
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        const conversations = await response.json();
        console.log("Conversations fetched:", conversations);

        conversationList.innerHTML = '';
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
                    await fetchConversations();
                    loadConversation(data.id);
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
        loadConversation("new_ephemeral_session_request"); // Demande une nouvelle session éphémère avec un UUID frais
    };
}

// Gère la suppression d'une conversation persistante
async function deleteConversation(convId) {
    try {
        const response = await fetch(`/conversations/${convId}`, { method: 'DELETE' });
        if (response.ok) {
            await fetchConversations();
            if (currentConversationId === convId) {
                loadConversation("new_ephemeral_session_request"); // Bascule vers une nouvelle éphémère si la conversation active est supprimée
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

// Logique pour l'affichage du sélecteur de projet et de la checkbox strict
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
    // N'envoyer l'historique que si currentConversationId est un string (UUID éphémère) ou null (première requête)
    if (typeof currentConversationId === 'string' || currentConversationId === null) { 
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
            // conversation_id sera l'UUID éphémère (string) ou l'ID numérique (int)
            conversation_id: currentConversationId, 
            rag_mode: ragModeParam,
            selected_project: selectedProject,
            strict_mode: strictModeCheckbox ? strictModeCheckbox.checked : false 
        };

        // Si currentConversationId est un string (UUID) ou null, inclure l'historique éphémère
        if (typeof currentConversationId === 'string' || currentConversationId === null) {
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

        // Si c'est une conversation éphémère (UUID ou null), ajouter la réponse du bot à ephemeralChatHistory
        if (typeof currentConversationId === 'string' || currentConversationId === null) {
            ephemeralChatHistory.push({ sender: 'bot', content: data.response });
        }

    } catch (error) {
        console.error('Erreur lors de l\'envoi du message:', error);
        displayMessage('bot', `Désolé, une erreur est survenue lors de la communication. Détails : ${error.message || error}.`);
        if (typeof currentConversationId === 'string' || currentConversationId === null) {
            ephemeralChatHistory.push({ sender: 'bot', content: 'Erreur.' });
        }
    } finally {
        sendButton.disabled = false;
        loadingIndicator.style.display = 'none';
    }
};

// Gestionnaire d'événements pour les sauts de ligne avec Shift+Entrée dans le textarea
userInput.addEventListener('keydown', function(e) {
    // Si Entrée est pressée SANS Shift
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // Empêche le saut de ligne par défaut dans le textarea
        sendButton.click(); // Déclenche l'envoi du message
    }
    // Si Shift+Entrée est pressé, le comportement par défaut (saut de ligne) est autorisé.
});


document.addEventListener('DOMContentLoaded', () => {
    fetchConversations();
    loadConversation("new_ephemeral"); // Au démarrage, activer la conversation éphémère par défaut, ce qui va aussi vider l'historique éphémère.

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