from google.cloud import aiplatform
from PIL import Image
import numpy as np
import json
import google.cloud.storage
from google.cloud import storage
from tensorflow.keras.models import load_model
import io
import base64
import io


#ce qu'on envoi au predict
with open("melanoma_0.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')



payload = {
  "instances": [
    {
      "image_bytes": {
        "b64": encoded_string
      }
    }
  ]
}

print('type payload :',type(payload))
#dans le predict


image_bytes = payload['instances'][0]['image_bytes']['b64']
print('image_bytes :', image_bytes)
image_data = base64.b64decode(image_bytes)


img = Image.open(io.BytesIO(image_data))

#img = Image.open(image_file)

model = load_model("model")


img = img.resize((224, 224))
ims_malignant = []
ims_malignant.append(np.array(img))


X_test = np.array(ims_malignant, dtype='uint8')
X_test = X_test / 255
y_pred = model.predict(X_test)
print(y_pred[0])

