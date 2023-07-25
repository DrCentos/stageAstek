FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3

WORKDIR /

# Installs additional packages
RUN pip install lightgbm flask scikit-learn

ENV AIP_STORAGE_URI=gs://nvabucket/model
ENV AIP_HEALTH_ROUTE=/ping
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=8080
ENV AIP_BUCKET_PRED=gs://nvabucket/pred

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
ENTRYPOINT ["python", "predict.py"]


