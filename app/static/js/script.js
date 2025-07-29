// app/static/js/script.js

document.addEventListener('DOMContentLoaded', function() {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const loadingIndicator = document.getElementById('loading-indicator');
    const conversationList = document.getElementById('conversation-list');
    const newConvBtn = document.getElementById('new-conv-btn');
    const ephemeralConvBtn = document.getElementById('ephemeral-conv-btn');
    const searchModeRadios = document.querySelectorAll('input[name="search_mode"]');
    const projectSelectionDiv = document.getElementById('project-selection');
    const projectSelect = document.getElementById('project-select');
    const strictModeCheckbox = document.getElementById('strict-mode-checkbox');
    const strictnessContainer = document.getElementById('strictness-container'); 

    const llmModelInput = document.getElementById('llm-model-input');
    const setLlmModelBtn = document.getElementById('set-llm-model-btn');
    const currentLlmModelDisplay = document.getElementById('current-llm-model-display');

    let currentConversationId = null; // null pour demander un nouvel ID au backend ou pour nouvelle éphémère
    let currentEphemeralHistory = []; 
    // Initialiser avec le modèle actuel affiché par le HTML (valeur par défaut de la config)
    let selectedLlmModel = currentLlmModelDisplay.textContent.replace('Modèle actuel: ', '').trim(); 


    // Fonction pour afficher un message dans la boîte de chat
    function displayMessage(sender, message, isNew = true) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender + '-message');

        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        messageContent.innerHTML = marked.parse(message); 

        messageElement.appendChild(messageContent);
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight; 

        if (isNew) {
            // L'historique éphémère est mis à jour SEULEMENT SI currentConversationId est un UUID (string)
            // C'est-à-dire si on est dans une session éphémère active (ID déjà reçu du backend)
            if (typeof currentConversationId === 'string' && currentConversationId.length === 36) { 
                currentEphemeralHistory.push({ sender: sender, content: message });
            }
        }
    }

    // Fonction pour charger l'historique d'une conversation persistante ou initialiser une éphémère
    async function loadConversation(convIdToLoad) {
        chatBox.innerHTML = ''; // Vider la boîte de chat

        // Mettre à jour l'état visuel des boutons de conversation actifs dans la barre latérale
        document.querySelectorAll('#conversation-list button').forEach(btn => btn.classList.remove('active'));
        if (ephemeralConvBtn) ephemeralConvBtn.classList.remove('active');
        
        loadingIndicator.style.display = 'block';

        if (convIdToLoad === "new_ephemeral_session_request") { 
            // Signal pour le backend de créer un nouvel ID UUID pour cette session éphémère
            currentConversationId = null; 
            currentEphemeralHistory = []; // Vider l'historique pour la nouvelle session
            if (ephemeralConvBtn) ephemeralConvBtn.classList.add('active');
            displayMessage('bot', `Nouvelle conversation éphémère active...`);
            // Le véritable ID UUID sera reçu avec la première réponse du chat
            loadingIndicator.style.display = 'none'; // Pas de chargement réel ici
        } else { // C'est un ID de conversation persistante (numérique)
            currentConversationId = convIdToLoad; 
            currentEphemeralHistory = []; // Vider l'historique éphémère lors du basculement vers une persistante
            
            const activeBtn = document.querySelector(`#conversation-list button[data-id="${currentConversationId}"]`);
            if (activeBtn) activeBtn.classList.add('active');

            try {
                const response = await fetch(`/conversations/${currentConversationId}`);
                if (!response.ok) {
                    throw new Error(`Erreur HTTP: ${response.status}`);
                }
                const messages = await response.json();
                messages.forEach(msg => displayMessage(msg.sender, msg.content, false)); 
                displayMessage('bot', `Mode: Conversation #${currentConversationId}. Historique chargé.`);
            } catch (error) {
                console.error('Erreur lors du chargement de la conversation:', error);
                displayMessage('bot', 'Désolé, une erreur est survenue lors du chargement de cette conversation.');
            } finally {
                loadingIndicator.style.display = 'none';
            }
        }
    }

    // Récupère et affiche la liste des conversations persistantes depuis le backend
    async function loadConversationList() {
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
                // Crée un bouton pour la conversation (pour le style)
                const convButton = document.createElement('button');
                convButton.textContent = conv.name;
                convButton.dataset.id = conv.id; // Stocke l'ID pour le chargement
                convButton.onclick = () => loadConversation(conv.id);

                const deleteBtn = document.createElement('button');
                deleteBtn.classList.add('delete-btn');
                deleteBtn.textContent = 'X';
                deleteBtn.onclick = async (e) => {
                    e.stopPropagation(); 
                    if (confirm(`Voulez-vous vraiment supprimer la conversation "${conv.name}" ?`)) {
                        await deleteConversation(conv.id);
                    }
                };
                convButton.appendChild(deleteBtn); // Le bouton de suppression est à l'intérieur du bouton de conversation
                li.appendChild(convButton); // Ajoute le bouton à l'élément de liste
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
                        await loadConversationList(); // Recharger la liste pour voir la nouvelle conversation
                        loadConversation(data.id); // Charger la nouvelle conversation
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
            // Demande une nouvelle session éphémère (l'ID sera généré par le backend lors du premier chat)
            loadConversation("new_ephemeral_session_request"); 
        };
    }

    // Gère la suppression d'une conversation persistante
    async function deleteConversation(convId) {
        try {
            const response = await fetch(`/conversations/${convId}`, { method: 'DELETE' });
            if (response.ok) {
                await loadConversationList(); 
                if (currentConversationId === convId) {
                    // Si la conversation active est supprimée, basculer vers une nouvelle éphémère
                    loadConversation("new_ephemeral_session_request"); 
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

    // Gère l'envoi du message
    sendButton.onclick = function() {
        const message = userInput.value.trim();
        if (message === '') return;

        displayMessage('user', message);
        userInput.value = '';
        sendButton.disabled = true; // Désactiver pour éviter les envois multiples
        loadingIndicator.style.display = 'block';

        const selectedMode = document.querySelector('input[name="search_mode"]:checked').value;
        const selectedProject = (selectedMode === 'code_rag') ? projectSelect.value : null;
        const strictMode = strictModeCheckbox.checked;

        let requestBody = {
            message: message,
            conversation_id: currentConversationId, // Sera null si demande de nouvelle éphémère
            rag_mode: selectedMode,
            selected_project: selectedProject,
            strict_mode: strictMode,
            llm_model_name: selectedLlmModel 
        };

        // Si conversation éphémère, inclure l'historique complet
        if (currentConversationId === null || currentConversationId === "new_ephemeral_session_request") {
            requestBody.ephemeral_history = currentEphemeralHistory;
            // Si c'est la toute première requête éphémère (currentConversationId est null),
            // le backend va générer un nouvel ID. Nous devons le récupérer.
            requestBody.conversation_id = null; 
        }

        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        })
        .then(response => response.json())
        .then(data => {
            displayMessage('bot', data.response);
            // Si le backend a généré un nouvel ID de conversation (pour une éphémère)
            if (data.conversation_id && currentConversationId === null) {
                currentConversationId = data.conversation_id; // Mettre à jour avec le nouvel UUID
                displayMessage('bot', `Session éphémère ID: ${currentConversationId.substring(0,8)}...`);
                console.log("Nouvel ID de session éphémère du backend:", currentConversationId);
            }
        })
        .catch(error => {
            console.error('Erreur:', error);
            displayMessage('bot', 'Désolé, une erreur est survenue lors de la communication avec l\'assistant.');
        })
        .finally(() => {
            sendButton.disabled = false;
            loadingIndicator.style.display = 'none';
        });
    };

    // Gestion de la sélection du mode de recherche (RAG vs Général) et VISIBILITÉ DU MODE STRICT
    searchModeRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.value === 'code_rag') {
                projectSelectionDiv.style.display = 'block';
                strictnessContainer.style.display = 'block'; 
            } else if (this.value === 'kb_rag') { 
                projectSelectionDiv.style.display = 'none'; // Pas de sélection de projet pour KB
                strictnessContainer.style.display = 'block';
            }
            else { 
                projectSelectionDiv.style.display = 'none';
                strictnessContainer.style.display = 'none'; 
            }
        });
    });

    // ÉVÉNEMENT POUR LE NOUVEAU BOUTON "Appliquer Modèle"
    setLlmModelBtn.addEventListener('click', function() {
        const newModelName = llmModelInput.value.trim();
        if (newModelName) {
            selectedLlmModel = newModelName; 
            currentLlmModelDisplay.textContent = `Modèle actuel: ${selectedLlmModel}`;
            llmModelInput.value = ''; 
            alert(`Modèle LLM mis à jour pour cette session: ${selectedLlmModel}`);
        } else {
            alert("Veuillez entrer un nom de modèle valide.");
        }
    });

    // NOUVEL ÉVÉNEMENT : Gérer la touche Entrée et Shift+Entrée dans la zone de texte
    userInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            if (event.shiftKey) {
                // Shift+Enter: Comportement par défaut (nouvelle ligne)
            } else {
                // Enter seul: Empêcher le saut de ligne et envoyer le message
                event.preventDefault(); 
                sendButton.click(); 
            }
        }
    });

    // Chargement initial au démarrage de la page
    loadConversationList();
    loadConversation("new_ephemeral_session_request"); // Charger une nouvelle conversation éphémère par défaut

    // Ajuster la visibilité initiale du mode strict au chargement de la page
    const initialMode = document.querySelector('input[name="search_mode"]:checked').value;
    if (initialMode === 'general') {
        strictnessContainer.style.display = 'none';
        projectSelectionDiv.style.display = 'none'; 
    } else if (initialMode === 'code_rag') {
        projectSelectionDiv.style.display = 'block';
        strictnessContainer.style.display = 'block';
    } else if (initialMode === 'kb_rag') {
        projectSelectionDiv.style.display = 'none'; 
        strictnessContainer.style.display = 'block';
    }
});