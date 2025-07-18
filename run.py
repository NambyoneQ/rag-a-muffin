from dotenv import load_dotenv # Importe la fonction pour charger les variables d'environnement
load_dotenv() # Charge les variables du fichier .env au tout début de l'exécution

from app import app, initialize_services_on_startup # Importe l'instance de l'application Flask et la fonction d'initialisation

if __name__ == '__main__':
    # Initialise les services de l'application (base de données, LLMs, RAG)
    # Ceci doit se faire dans un contexte d'application Flask
    with app.app_context():
        initialize_services_on_startup()
    
    # Lance l'application Flask
    # debug=True active le mode débogage (rechargement automatique si code modifié, messages d'erreur détaillés)
    # use_reloader=False désactive le reloader automatique pour éviter les problèmes de contexte avec les variables globales
    # port=5000 définit le port sur lequel Flask écoutera
    app.run(debug=True, use_reloader=False, port=5000)