FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3

WORKDIR /

COPY requirements.txt .
RUN pip install -r requirements.txt

#nom du bucket ou sont stocké les images
#ENV AIP_BUCKET=buckettestnv pour les petits test
ENV AIP_BUCKET=nvabucket
#chemin du bucket où on stocke notre model doit être le même que AIP_STORAGE_URI de predict
ENV AIP_STORAGE_URI=gs://nvabucket/model

COPY . .
ENTRYPOINT ["python", "train.py"]

