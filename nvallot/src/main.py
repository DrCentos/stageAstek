import sys
from os import path



current_folder = path.dirname(path.abspath(__file__))
import subprocess
subprocess.check_call([sys.executable, "-m", "pip","install","--upgrade","tensorflow","mlflow","protobuf==3.20.0" , "-t", current_folder])


import numpy as np
import os
import keras
import mlflow
import mlflow.keras as mlk
# Affiche la version de MLflow
print("Version de MLflow :", mlflow.__version__)

from keras.callbacks import Callback
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from keras.optimizers import Adam, RMSprop
import io
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from azure.storage.blob import BlobServiceClient
import tensorflow as tf

from PIL import Image

connection_string = "DefaultEndpointsProtocol=https;AccountName=nvastockage;AccountKey=BhpwOnQK0S4VBbksgH6ptRk7Jo9Db14tXCFoEWA6cp1iKQ6VdcQ07pcB0+ew4lDQ6lx5b314i7vX+AStPgJcTg==;EndpointSuffix=core.windows.net"

# Afficher la version de MLflow
print("Version de MLflow:", mlflow.__version__)

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        print('log :', log)
        #run.log('Loss', log['val_loss'])
        #run.log('Accuracy', log['val_accuracy'])




print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)

# Start Logging
mlflow.start_run()

# enable autologging
mlflow.sklearn.autolog()

OUTPUT_DIR = './outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Définit les informations du bucket S3
bucket_name = 'nvabucket'

# Liste les objets dans le bucket avec le préfixe spécifié


#on va vérifier s'il existe un model et si c'est le cas on prend le plus récent pour fine train
#on regarde si on model existe déjà
#TODO faire le système check si model existe retrain ...
retrain = False


# Load in training pictures
print("avant load bening")

# Se connecter au service Blob Storage
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Récupérer le conteneur spécifié
container_client = blob_service_client.get_container_client(bucket_name)

# Liste pour stocker les images chargées
ims_benign = []

# Parcourir tous les blobs (images) dans le conteneur
for blob in container_client.list_blobs(name_starts_with="benign"):
    # Vérifier si le blob est une image
    if blob.name.endswith(('.png', '.jpg', '.jpeg')):
        # Récupérer les données de l'image
        blob_client = container_client.get_blob_client(blob.name)
        image_data = blob_client.download_blob().readall()

        # Convertir les données de l'image en objet Image
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((224, 224))
        image_array = np.array(image)

        ims_benign.append(image_array)


X_benign = np.array(ims_benign, dtype='uint8')
print("après load benign")

print("avant load malegnant")


# Se connecter au service Blob Storage
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Récupérer le conteneur spécifié
container_client = blob_service_client.get_container_client(bucket_name)

# Liste pour stocker les images chargées
ims_malin = []

# Parcourir tous les blobs (images) dans le conteneur
for blob in container_client.list_blobs(name_starts_with="malignant"):
    # Vérifier si le blob est une image
    if blob.name.endswith(('.png', '.jpg', '.jpeg')):
        # Récupérer les données de l'image
        blob_client = container_client.get_blob_client(blob.name)
        image_data = blob_client.download_blob().readall()

        # Convertir les données de l'image en objet Image
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((224, 224))
        image_array = np.array(image)

        ims_malin.append(image_array)


X_malignant = np.array(ims_malin, dtype='uint8')

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
    #à revoir possible qu'il faut aller chercher dans le bucket ou jsp
    #model_root = Model.get_model_path('credit_defaults_model')
    #print('model_root : ', model_root)
    #model = mlflow.keras.load_model(model_root)
    print('après le load')
    # supprime tous les model pour garder de l'espace (on va seulement avoir le dernier)
    #TODO changer la fonction
    #delete_model_file_from_s3(bucket_name, "output")

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

# Create directory
# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)
registered_model_name = 'credit_defaults_model'


print("Registering the model via MLFlow")
mlk.log_model(
    model,
    registered_model_name=registered_model_name,
    artifact_path=registered_model_name
)

# Saving the model to a file
mlk.save_model(
    model,
    path=os.path.join(registered_model_name, "trained_model"),
)


# Clear memory, because of memory overload
del model



