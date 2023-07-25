from flask import Flask, render_template, request,jsonify
from google.cloud import storage
import base64
from google.cloud import aiplatform
import google.cloud.storage
from google.cloud import storage

app = Flask(__name__, template_folder='template')

@app.route("/")
def template_test():
    return render_template('index.html', my_string="Wheeeee!", my_list=[0,1,2,3,4,5])



@app.route('/upload', methods=['POST'])
def upload():
    #TODO : trouver un moyen d'avoir l'enpoint id ici on peu le récupérer a la fin du train par exemple
    # le stocker dans une variable d'en ou encore dans un fichier txt dans un bucket
    #endpoint_id = "7266285319878606848"

    endpoint_id = "2528498511884845056"

    REGION = 'europe-west1'
    PROJECT_ID = 'glossy-precinct-371813'
    staging_bucket = 'gs://nvabucket/model'  # Should be same as AIP_STORAGE_URI specified in docker file
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=staging_bucket)
    endpoint = aiplatform.Endpoint(endpoint_id)

    # Récupérer le fichier envoyé dans la requête POST
    file = request.files['file']
    print(file)
    encoded_string = base64.b64encode(file.read()).decode('utf-8')

    payload = {
        "instances": [
            {
                "image_bytes": {
                    "b64": encoded_string
                }
            }
        ]
    }

    prediction = endpoint.predict(instances=payload["instances"])[0][0]
    print('prediction :', prediction)
    if prediction[0] > prediction[1]:
        result = "benign"
    else:
        result = "malignant"

    return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
