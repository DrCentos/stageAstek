FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3

WORKDIR /

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
ENTRYPOINT ["python", "train.py"]