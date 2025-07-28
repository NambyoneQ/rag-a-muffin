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
    // NOUVELLE VARIABLE : Conteneur du mode strict
    const strictnessContainer = document.getElementById('strictness-container'); 

    // NOUVELLES VARIABLES pour la sélection du modèle LLM
    const llmModelInput = document.getElementById('llm-model-input');
    const setLlmModelBtn = document.getElementById('set-llm-model-btn');
    const currentLlmModelDisplay = document.getElementById('current-llm-model-display');

    let currentConversationId = null; 
    let currentEphemeralHistory = []; 
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
            if (currentConversationId === null || currentConversationId === "new_ephemeral_session_request") {
                currentEphemeralHistory.push({ sender: sender, content: message });
            }
        }
    }

    // Fonction pour charger l'historique d'une conversation persistante
    function loadConversation(convId) {
        if (currentConversationId === null || currentConversationId === "new_ephemeral_session_request") {
            currentEphemeralHistory = []; 
        }

        currentConversationId = convId;
        chatBox.innerHTML = ''; 

        if (convId === "new_ephemeral_session_request") {
            displayMessage('bot', "Bonjour ! Je suis votre assistant. Que puis-je faire pour vous aujourd'hui ?");
            displayMessage('bot', "Mode: Conversation éphémère. L'historique ne sera pas sauvegardé.");
            currentEphemeralHistory = []; 
            return;
        }

        loadingIndicator.style.display = 'block';
        fetch(`/conversations/${convId}`)
            .then(response => response.json())
            .then(messages => {
                messages.forEach(msg => displayMessage(msg.sender, msg.content, false)); 
                displayMessage('bot', `Mode: Conversation #${convId}. Historique chargé.`);
            })
            .catch(error => {
                console.error('Erreur lors du chargement de la conversation:', error);
                displayMessage('bot', 'Désolé, une erreur est survenue lors du chargement de l\'historique.');
            })
            .finally(() => {
                loadingIndicator.style.display = 'none';
            });
    }

    // Fonction pour charger la liste des conversations du menu latéral
    function loadConversationList() {
        fetch('/conversations')
            .then(response => response.json())
            .then(conversations => {
                conversationList.innerHTML = '';
                conversations.forEach(conv => {
                    const listItem = document.createElement('li');
                    listItem.textContent = conv.name;
                    listItem.dataset.convId = conv.id;
                    listItem.addEventListener('click', () => loadConversation(conv.id));

                    const deleteBtn = document.createElement('button');
                    deleteBtn.textContent = 'X';
                    deleteBtn.classList.add('delete-conv-btn');
                    deleteBtn.onclick = (e) => {
                        e.stopPropagation(); 
                        if (confirm(`Voulez-vous vraiment supprimer la conversation "${conv.name}" ?`)) {
                            fetch(`/conversations/${conv.id}`, {
                                method: 'DELETE'
                            })
                            .then(response => response.json())
                            .then(data => {
                                if (data.status === 'success') {
                                    alert('Conversation supprimée !');
                                    loadConversationList(); 
                                    if (currentConversationId === conv.id) {
                                        currentConversationId = null; 
                                        chatBox.innerHTML = '';
                                        displayMessage('bot', "Conversation supprimée. Veuillez en créer une nouvelle ou commencer une conversation éphémère.");
                                    }
                                } else {
                                    alert('Erreur: ' + data.error);
                                }
                            })
                            .catch(error => console.error('Erreur:', error));
                        }
                    };
                    listItem.appendChild(deleteBtn);
                    conversationList.appendChild(listItem);
                });
            })
            .catch(error => console.error('Erreur lors du chargement des conversations:', error));
    }

    // Gérer l'envoi du message
    sendButton.onclick = function() {
        const message = userInput.value.trim();
        if (message === '') return;

        displayMessage('user', message);
        userInput.value = '';
        loadingIndicator.style.display = 'block';

        const selectedMode = document.querySelector('input[name="search_mode"]:checked').value;
        const selectedProject = (selectedMode === 'code_rag') ? projectSelect.value : null;
        const strictMode = strictModeCheckbox.checked;

        let requestBody = {
            message: message,
            conversation_id: currentConversationId,
            rag_mode: selectedMode,
            selected_project: selectedProject,
            strict_mode: strictMode,
            llm_model_name: selectedLlmModel 
        };

        if (currentConversationId === null || currentConversationId === "new_ephemeral_session_request") {
            requestBody.ephemeral_history = currentEphemeralHistory;
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
        })
        .catch(error => {
            console.error('Erreur:', error);
            displayMessage('bot', 'Désolé, une erreur est survenue lors de la communication avec l\'assistant.');
        })
        .finally(() => {
            loadingIndicator.style.display = 'none';
        });
    };

    // Gestion de la sélection du mode de recherche (RAG vs Général) et VISIBILITÉ DU MODE STRICT
    searchModeRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.value === 'code_rag') {
                projectSelectionDiv.style.display = 'block';
                strictnessContainer.style.display = 'block'; // Mode strict visible pour code_rag
            } else if (this.value === 'kb_rag') { // Mode strict aussi visible pour kb_rag
                projectSelectionDiv.style.display = 'none';
                strictnessContainer.style.display = 'block';
            }
            else { // Pour le mode général
                projectSelectionDiv.style.display = 'none';
                strictnessContainer.style.display = 'none'; // Mode strict masqué pour général
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
        // Si la touche est 'Enter'
        if (event.key === 'Enter') {
            // Si Shift est maintenu, permettre le saut de ligne par défaut
            if (event.shiftKey) {
                // Le comportement par défaut (nouvelle ligne) est autorisé
            } else {
                // Si Shift n'est pas maintenu, empêcher le saut de ligne et envoyer le message
                event.preventDefault(); // Empêche le saut de ligne par défaut
                sendButton.click(); // Déclenche le click du bouton Envoyer
            }
        }
    });

    // Chargement initial
    loadConversationList();
    loadConversation("new_ephemeral_session_request"); 

    // Ajuster la visibilité initiale du mode strict au chargement de la page
    // Simuler un événement de changement de mode pour appliquer la bonne visibilité au démarrage
    const initialMode = document.querySelector('input[name="search_mode"]:checked').value;
    if (initialMode === 'general') {
        strictnessContainer.style.display = 'none';
        projectSelectionDiv.style.display = 'none'; // Assurez-vous que le projet est aussi masqué si mode général
    } else if (initialMode === 'code_rag') {
        projectSelectionDiv.style.display = 'block';
        strictnessContainer.style.display = 'block';
    } else if (initialMode === 'kb_rag') {
        projectSelectionDiv.style.display = 'none';
        strictnessContainer.style.display = 'block';
    }
});