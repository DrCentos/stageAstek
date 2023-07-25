from os import path

import tensorflow as tf
import pandas as pd
import flask, joblib

prefix = "/opt/ml/"
model_path = path.join(prefix, "model")
tf_path = path.join(model_path, "tf-model")
sc_path = path.join(model_path, "scaler")

model = tf.keras.models.load_model(tf_path)
with open(path.join(sc_path, "scaler.bin"), 'rb') as f:
    scaler = joblib.load(f)

# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=['GET'])
def ping():
    status = 200 if model and scaler else 404
    return flask.Response(response='\n', status=status, mimetype="application/json")

@app.route("/invocations", methods=['POST'])
def transformation():
    # Convert from CSV to pandas
    if flask.request.content_type in ["text/csv", "text/plain"]:
        data = flask.request.data.decode('utf-8')
    else:
        return flask.Response(
            response=f"This predictor only supports CSV data, got {flask.request.content_type}",
            status=415,
            mimetype="text/plain"
        )

    # Do the prediction
    data = scaler.transform([[float(i)] for i in data.split(',')])
    predictions = model.predict(data).flatten().tolist()
    predictions = scaler.inverse_transform([[i] for i in predictions]).flatten().tolist()

    # Convert from numpy back to CSV
    out = pd.DataFrame(predictions)
    result = out.to_csv(header=False, index=False)

    return flask.Response(response=result, status=200, mimetype="text/csv")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
