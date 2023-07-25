from google.cloud import aiplatform
from PIL import Image
import numpy as np
import json
import google.cloud.storage
from google.cloud import storage
from tensorflow.keras.models import load_model
import io


print("start")
# Prétraitement de l'image
client = storage.Client()
bucket = client.get_bucket("nvabucket")
blobs = bucket.list_blobs(prefix='{}/'.format("pred"), delimiter='/')
file = []
for blob in blobs:
    if blob.name.endswith('.jpg'):
        file.append(blob.name)
        
        
"""

logging.info('IMAGE_PATH :', "pred")
print("BUCKET_PATH :", "nvabucket")
command = f"gcloud storage cp -r gs://nvabucket/pred ."
subprocess.run(command, shell=True, stdout=subprocess.PIPE)

pred_dir = "./pred"
for filename in os.listdir(pred_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Si le fichier est une image, ouvrir le fichier et ajouter son contenu au tableau
        with open(os.path.join(pred_dir, filename), "rb") as f:
            image_content = f.read()
            file.append(image_content)

"""

ims_malignant = []
for blob in file:
    print("dans la boucle malegnant load")
    blobs = bucket.blob(blob)
    with io.BytesIO(blobs.download_as_bytes()) as f:
        with Image.open(f) as img:
            img = img.resize((224, 224))
            ims_malignant.append(np.array(img))

X_test = np.array(ims_malignant, dtype='uint8')
X_test = X_test / 255

model = load_model("model")
print("load ok")
y_pred = model.predict(X_test)
print(y_pred[0])

if(y_pred[0][0] > y_pred[0][1]):
    print("benin")
else:
    print("malignant")
print("end")

#je supprime les photos dans le bucket possiblement faudra le faire autre part apres
# Sélectionner le fichier à supprimer
for blob in file:
    blobs = bucket.blob(blob)
    # Supprimer le fichier
    blobs.delete()