print("DEBUG: Lancement de run.py - Point 1")
from dotenv import load_dotenv
print("DEBUG: Lancement de run.py - Point 2: dotenv importé")
load_dotenv()
print("DEBUG: Lancement de run.py - Point 3: .env chargé")

from app import app, initialize_services_on_startup
print("DEBUG: Lancement de run.py - Point 4: app importé")

if __name__ == '__main__':
    print("DEBUG: Lancement de run.py - Point 5: Dans __main__")
    with app.app_context():
        print("DEBUG: Lancement de run.py - Point 6: Dans app_context")
        initialize_services_on_startup()
        print("DEBUG: Lancement de run.py - Point 7: initialize_services_on_startup terminé")

    print("DEBUG: Lancement de run.py - Point 8: Avant app.run()")
    app.run(debug=True, use_reloader=False, port=5000)
    print("DEBUG: Lancement de run.py - Point 9: app.run() terminé (ne devrait pas être atteint)")