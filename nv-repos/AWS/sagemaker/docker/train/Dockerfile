# From aws training image for tensorflow
# FROM tensorflow-training:2.1.0-cpu-py3
FROM python:3.9
# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Launch the app
COPY . .
# CMD ["python", "job.py"]
ENTRYPOINT [ "python", "train.py" ]