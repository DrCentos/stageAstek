#firebase init
#firebase deploy
#firebase hosting:disable


import firebase_admin
from firebase_admin import credentials

import os
import subprocess


cred = credentials.Certificate("jsonkey.json")
firebase_admin.initialize_app(cred)


# Chemin vers le répertoire contenant les fichiers de l'application à déployer
chemin_repertoire_application = "./"


def deployer_application_firebase():
    # Déploiement des fichiers vers Firebase Hosting
    commande_depot_fichiers = f"firebase deploy --only hosting -m 'Déploiement automatique'"
    subprocess.run(commande_depot_fichiers, shell=True, check=True, cwd=chemin_repertoire_application)

    print("Application Firebase hébergée déployée avec succès.")

# Exemple : déploiement de l'application
deployer_application_firebase()
