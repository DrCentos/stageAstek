import os
from kaggle import KaggleApi

# Chemin vers votre fichier kaggle.json
kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")

# Instancier l'API Kaggle
api = KaggleApi()
api.authenticate()

# Configuration du chemin d'enregistrement du fichier de sortie
output_file_path = "./"

# Télécharger les données du jeu de données spécifié (pb les données sont passé en privé)
api.dataset_download_files('fanconic/skin-cancer-malignant-vs-benign', path=output_file_path, unzip=False)
