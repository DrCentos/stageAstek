# FROM tensorflow-training:2.1.0-cpu-py3
#az login
#az acr login --name nvregistre
#docker build -t nvregistre.azurecr.io/usecase1:latest .
#docker push nvregistre.azurecr.io/usecase1:latest
#docker run nvregistre.azurecr.io/usecase1:latest
FROM python:3.9
# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Launch the app
COPY .. .
# CMD ["python", "job.py"]
ENTRYPOINT [ "python", "pipeline.py" ]