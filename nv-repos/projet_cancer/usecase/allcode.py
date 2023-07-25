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


"""# Step 2 : Loading pictures and making Dictionary of images and labels
In this step I load in the pictures and turn them into numpy arrays using their RGB values. As the pictures have already been resized to 224x224, there's no need to resize them. As the pictures do not have any labels, these need to be created. Finally, the pictures are added together to a big training set and shuffeled.
"""

"""
folder_benign_train = 'gs://nvallot_bucket/train/train/benign'
folder_malignant_train = 'gs://nvallot_bucket/train/train/malignant'

folder_benign_test = 'gs://nvallot_bucket/train/test/benign'
folder_malignant_test = 'gs://nvallot_bucket/train/test/malignant'

read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

# Load in training pictures
ims_benign = [read(os.path.join(folder_benign_train, filename)) for filename in os.listdir(folder_benign_train)]
X_benign = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_train, filename)) for filename in
                 os.listdir(folder_malignant_train)]
X_malignant = np.array(ims_malignant, dtype='uint8')


# Load in testing pictures
ims_benign = [read(os.path.join(folder_benign_test, filename)) for filename in os.listdir(folder_benign_test)]
X_benign_test = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_test, filename)) for filename in os.listdir(folder_malignant_test)]
X_malignant_test = np.array(ims_malignant, dtype='uint8')

# Create labels
y_benign = np.zeros(X_benign.shape[0])
y_malignant = np.ones(X_malignant.shape[0])

y_benign_test = np.zeros(X_benign_test.shape[0])
y_malignant_test = np.ones(X_malignant_test.shape[0])

# Merge data
X_train = np.concatenate((X_benign, X_malignant), axis=0)
y_train = np.concatenate((y_benign, y_malignant), axis=0)

X_test = np.concatenate((X_benign_test, X_malignant_test), axis=0)
y_test = np.concatenate((y_benign_test, y_malignant_test), axis=0)


# Shuffle data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
y_train = y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
y_test = y_test[s]
"""

"""
# Authenticate with Google Cloud
client = storage.Client("glossy-precinct-371813")

# Get a reference to the GCS bucket
bucket = client.get_bucket("nvafter")#getenv('AIP_MODEL'))

# Define a function to read an image file from a GCS bucket
def read_image(blob):
    image_bytes = io.BytesIO(blob.download_as_bytes())
    return np.asarray(Image.open(image_bytes).convert('RGB'))

# Load in training pictures for benign images
folder_benign_train = 'benign'
blobs_benign = bucket.list_blobs(prefix=folder_benign_train)
ims_benign = [read_image(blob) for blob in blobs_benign]
X_benign = np.array(ims_benign, dtype='uint8')

# Load in training pictures for malignant images
folder_malignant_train = 'malignant'
blobs_malignant = bucket.list_blobs(prefix=folder_malignant_train)
ims_malignant = [read_image(blob) for blob in blobs_malignant]
X_malignant = np.array(ims_malignant, dtype='uint8')
"""



client = storage.Client()
#TODO : remettre getenv('AIP_BUCKET')
bucket = client.get_bucket(getenv('AIP_BUCKET'))

# Load in training pictures
print("avant load bening")
blobs = bucket.list_blobs(prefix='{}/'.format("benign"), delimiter='/')
filesbenign = []
for blob in blobs:
    if blob.name.endswith('.jpg'):
        filesbenign.append(blob.name)

ims_benign = []
for blob in filesbenign:
    print("dans la boucle benign load")
    blobs = bucket.blob(blob)
    with io.BytesIO(blobs.download_as_bytes()) as f:
        with Image.open(f) as img:
            #img = img.resize((300, 300))
            ims_benign.append(np.array(img))

X_benign = np.array(ims_benign, dtype='uint8')
print("après load benign")

print("avant load malegnant")

blobs = bucket.list_blobs(prefix='{}/'.format("malignant"), delimiter='/')
filesmalignant = []
for blob in blobs:
    if blob.name.endswith('.jpg'):
        filesmalignant.append(blob.name)

ims_malignant = []
for blob in filesmalignant:
    print("dans la boucle malegnant load")
    blobs = bucket.blob(blob)
    with io.BytesIO(blobs.download_as_bytes()) as f:
        with Image.open(f) as img:
            #img = img.resize((300, 300))
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

# With data augmentation to prevent overfitting
X_train = X_train / 255.
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



"""
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=5,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=1e-7)


input_shape = (300, 300, 3)
lr = 1e-5
init = 'normal'
activ = 'relu'
optim = 'adam'
epochs = 50
batch_size = 64
"""

"""
model = build(lr=lr, init=init, activ=activ, optim=optim, input_shape=input_shape)

history = model.fit(X_train, y_train, validation_split=0.2,
                    epochs=epochs, batch_size=batch_size, verbose=0,
                    callbacks=[learning_rate_reduction])

K.clear_session()
del model
del history

"""



"""# Step 6: Cross-Validating Model
"""
# define 3-fold cross validation test harness
"""
kfold = KFold(n_splits=3, shuffle=True, random_state=11)


cvscores = []
for train, test in kfold.split(X_train, y_train):
    # create model
    model = build(lr=lr,
                  init=init,
                  activ=activ,
                  optim=optim,
                  input_shape=input_shape)

    # Fit the model
    model.fit(X_train[train], y_train[train], epochs=epochs, batch_size=batch_size, verbose=0)
    # evaluate the model
    scores = model.evaluate(X_train[test], y_train[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)
    K.clear_session()
    del model

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
"""

"""# Step 7: Testing the model
First the model has to be fitted with all the data, such that no data is left out.
"""

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(factor=0.5,
                                            min_lr=1e-7,
                                            patience=5,
                                            verbose=1)

early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=10,
                               verbose=1,
                               restore_best_weights=True)

input_shape = (300, 300, 3)
lr = 1e-5
init = 'normal'
activ = 'relu'
optim = 'adam'
epochs = 50
batch_size = 64


model = build(lr=lr,
              init=init,
              activ=activ,
              optim=optim,
              input_shape=input_shape)

model.fit(X_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          verbose=1,
          #validation_data=(X_test, y_test),
          callbacks=[learning_rate_reduction, early_stopping])
"""
# Fitting model to all data
model = build(lr=lr,
              init=init,
              activ=activ,
              optim=optim,
              input_shape=input_shape)
print("avant le fit")
model.fit(X_train, y_train,
          epochs=epochs, batch_size=batch_size, verbose=0,
          callbacks=[learning_rate_reduction]
          )
"""

print("après le fit")
# Testing model on test data to evaluate
#y_pred = model.predict_classes(X_test)

#print(accuracy_score(np.argmax(y_test, axis=1), y_pred))


"""
# save model
# serialize model to JSON
model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")"""


#TODO : remettre getenv('AIP_MODEL_DIR')
BUCKET_PATH = getenv('AIP_MODEL_DIR') #"gs://nvbucket/model"
MODEL_PATH = path.join(BUCKET_PATH, 'model')
print("BUCKET_PATH: ", BUCKET_PATH)

# Create output directory
makedirs(BUCKET_PATH, exist_ok=True)
makedirs(MODEL_PATH, exist_ok=True)
# Save model
model.save(MODEL_PATH)
print("the model is saved")

# Clear memory, because of memory overload
del model
K.clear_session()


