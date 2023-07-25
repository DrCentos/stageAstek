#pip install -r requirements.txt && python -m unicorn app:app --host 0.0.0.0
#az login
#az acr login --name nvregistre
#docker build -t nvregistre.azurecr.io/usecase1:latest .
#docker push nvregistre.azurecr.io/usecase1:latest
#docker run nvregistre.azurecr.io/usecase1:latest
# Utilisez une image de base Python
FROM python:3.10

# Répertoire de travail dans le conteneur
WORKDIR .

# Copier les fichiers source dans le conteneur
COPY . .

# Install dependencies
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Exposer le port utilisé par l'application Flask
EXPOSE 5000

# Définir la variable d'environnement FLASK_APP
ENV FLASK_APP=application.py

# Commande pour exécuter l'application Flask
CMD ["flask", "run", "--host=0.0.0.0"]
