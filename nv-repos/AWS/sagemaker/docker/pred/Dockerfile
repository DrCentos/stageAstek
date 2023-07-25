FROM python:3.9
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Launch the app
COPY . .
ENTRYPOINT [ "python", "pred.py" ]