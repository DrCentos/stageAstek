from tensorflow import keras
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

from datetime import datetime
from os import path, getenv, makedirs, listdir
"""
# Step 1 : importing Essential Libraries
"""
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image

np.random.seed(3)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import itertools
import io
import keras
from keras.utils.np_utils import to_categorical  # used for converting labels to one-hot-encoding
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
#from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50
from keras import backend as K
import google.cloud.storage
from google.cloud import storage
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


client = storage.Client()

bucket = client.get_bucket(getenv('AIP_BUCKET'))


#on va vérifier s'il existe un model et si c'est le cas on prend le plus récent pour fine train
import os
from google.cloud import storage

original_string = getenv('AIP_STORAGE_URI') #"gs://nvabucket/model

suffix = original_string.split("//")[1]
bucket_name  = suffix.split("/")[0]
folder_name = suffix.split("/")[1] + "/"

print("folder_name: ",folder_name)
print("bucket_name : ",bucket_name)

storage_client = storage.Client()

blobs = storage_client.list_blobs(bucket_name, prefix=folder_name)#,delimiter=delimiter)
blob_names = [blob.name for blob in blobs]

BUCKET_PATH = getenv('AIP_MODEL_DIR')  # "gs://nvabucket/model/aiplatform-custom-training-2023-05-04-13:58:50.370/model/"
MODEL_PATH = path.join(BUCKET_PATH, 'model')  # gs://nvabucket/model/aiplatform-custom-training-2023-05-04-13:58:50.370/model/model
print("BUCKET_PATH: ", BUCKET_PATH)
print("MODEL_PATH: ", MODEL_PATH)
# Create output directory
makedirs(BUCKET_PATH, exist_ok=True)
makedirs(MODEL_PATH, exist_ok=True)
print("blob_names :", blob_names)
#je regarde s'il existe déjà un model (dans le cas oui alors il fera automatiquement un retrain)
if len(blob_names) > 1:
    print("retrain")
    retrain = True
    #si vous voulez retrain il faut créer un dossier retrain/malignant et retrain/benign dans votre bucket sinon le faire via l'interface qui le fait tout seul
    blobs_benign = bucket.list_blobs(prefix='{}/{}/'.format("retrain","benign"), delimiter='/')
    size_b = sum(1 for _ in bucket.list_blobs(prefix='{}/{}/'.format("retrain","benign"), delimiter='/'))
    print(f"La taille du fichier benign est : {size_b}")
    blobs_malignant = bucket.list_blobs(prefix='{}/{}/'.format("retrain","malignant"), delimiter='/')
    size_m = sum(1 for _ in bucket.list_blobs(prefix='{}/{}/'.format("retrain","malignant"), delimiter='/'))
    print(f"La taille du fichier malignant est : {size_m}")
    #dans le cas ou on ne passe rien dans le bucket du retrain on va alors faire comme si c'était un nouveau train
    if size_b<1 or size_m<1:
        print("pas retrain car dossier retrain vide")
        retrain = False
        blobs_benign = bucket.list_blobs(prefix='{}/'.format("benign"), delimiter='/')
        blobs_malignant = bucket.list_blobs(prefix='{}/'.format("malignant"), delimiter='/')
else :
    print("pas retrain car dans else")
    retrain = False
    blobs_benign = bucket.list_blobs(prefix='{}/'.format("benign"), delimiter='/')
    blobs_malignant = bucket.list_blobs(prefix='{}/'.format("malignant"), delimiter='/')


# Load in training pictures
print("avant load bening")

filesbenign = []
for blob in blobs_benign:
    if blob.name.endswith('.jpg'):
        filesbenign.append(blob.name)

ims_benign = []
for blob in filesbenign:
    print("dans la boucle benign load")
    blobs_benign = bucket.blob(blob)
    with io.BytesIO(blobs_benign.download_as_bytes()) as f:
        with Image.open(f) as img:
            img = img.resize((224, 224))
            ims_benign.append(np.array(img))


X_benign = np.array(ims_benign, dtype='uint8')
print("après load benign")

print("avant load malegnant")


filesmalignant = []
for blob in blobs_malignant:
    if blob.name.endswith('.jpg'):
        filesmalignant.append(blob.name)

ims_malignant = []
for blob in filesmalignant:
    print("dans la boucle malegnant load")
    blobs_malignant = bucket.blob(blob)
    with io.BytesIO(blobs_malignant.download_as_bytes()) as f:
        with Image.open(f) as img:
            img = img.resize((224, 224))
            ims_malignant.append(np.array(img))


X_malignant = np.array(ims_malignant, dtype='uint8')

print("après load malegnant")

print("après load")



# Create labels
y_benign = np.zeros(X_benign.shape[0])
y_malignant = np.ones(X_malignant.shape[0])


# Merge data
X_train = np.concatenate((X_benign, X_malignant), axis=0)
y_train = np.concatenate((y_benign, y_malignant), axis=0)


# Shuffle data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
y_train = y_train[s]




"""# Step 3: Categorical Labels
Turn labels into one hot encoding
"""
y_train = to_categorical(y_train, num_classes=2)
#y_test = to_categorical(y_test, num_classes=2)

"""# Step 4 : Normalization
Normalize all Values of the pictures by dividing all the RGB values by 255
"""
X_train = X_train / 255.
# With data augmentation to prevent overfitting

#X_test = X_test / 255.

"""# Step 5: Model Building 
## CNN
"""

def build(input_shape=(224, 224, 3), lr=1e-3, num_classes=2,
          init='normal', activ='relu', optim='adam'):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), padding='Same', input_shape=input_shape,
                     activation=activ, kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='Same',
                     activation=activ, kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=init))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    if optim == 'rmsprop':
        optimizer = RMSprop(lr=lr)

    else:
        optimizer = Adam(lr=lr)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(factor=0.5,
                                            min_lr=1e-7,
                                            patience=5,
                                            verbose=1)

early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=10,
                               verbose=1,
                               restore_best_weights=True)

input_shape = (224, 224, 3)
lr = 1e-5
init = 'normal'
activ = 'relu'
optim = 'adam'
epochs = 50
batch_size = 64

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


if retrain:
    print("on charge un model déjà existant")
    latest_model_folder = sorted(blob_names)[-1]
    print("latest_model_folder : ", latest_model_folder)
    old_model_name = MODEL_PATH.split("/")[4]
    new_model_name = latest_model_folder.split("/")[1]
    print("new_model_name : ", new_model_name)
    folder_model = MODEL_PATH.replace(old_model_name, new_model_name)
    print("folder_model : ", folder_model)
    model = load_model(folder_model)
    #supprime tous les model pour garder de l'espace (on va seulement avoir le dernier)
    modeldest = 'model/'
    malignant_blobs = bucket.list_blobs(prefix= modeldest)
    for blob in malignant_blobs:
        blob.delete()


else:
    print("on va créer notre premier model")
    model = build(lr=lr,
                  init=init,
                  activ=activ,
                  optim=optim,
                  input_shape=input_shape)


print("avant le fit")

model.fit(X_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[learning_rate_reduction, early_stopping])

print("après le fit")
# Testing model on test data to evaluate
#y_pred = model.predict_classes(X_test)

#print(accuracy_score(np.argmax(y_test, axis=1), y_pred))

# Save model
model.save(MODEL_PATH)
print("the model is saved")

# Clear memory, because of memory overload
del model
K.clear_session()


