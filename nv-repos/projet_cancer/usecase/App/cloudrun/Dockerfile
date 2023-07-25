FROM python:3.9-slim

# Installe les dépendances
RUN pip install google-cloud-storage google-cloud-aiplatform

# Copie le code source dans le conteneur
COPY main.py .

# Définit la commande par défaut à exécuter lorsque le conteneur démarre
CMD ["python", "main.py"]
