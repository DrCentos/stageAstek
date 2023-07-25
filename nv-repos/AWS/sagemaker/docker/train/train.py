from os import path, makedirs
import sys
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM #, Dropout

from sklearn.preprocessing import MinMaxScaler

# Sagemaker variables
prefix = '/opt/ml'
input_path = path.join(prefix, 'input', 'data')
output_path = path.join(prefix, 'output')
model_path = path.join(prefix, 'model')
# data_path = path.join(output_path, 'data')
failure_path = path.join(prefix, 'failure')

channel_name='training'
training_path = path.join(input_path, channel_name)

def create_model(units=32, activation='relu', optimizer='adam', dropout_rate=0.0):
    # Building the RNN
    regressor = Sequential()
    regressor.add(LSTM(units=units, activation=activation, input_shape=(None, 1)))
    # regressor.add(Dropout(dropout_rate))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer=optimizer, loss='mean_squared_error')
    return regressor

def train():
    try:
        input_file = path.join(training_path, 'coinbaseUSD.csv')
        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
        group = df.groupby('date')
        Real_Price = group['Weighted_Price'].mean()

        prediction_days = 30
        df_train = Real_Price[:len(Real_Price)-prediction_days]
        # df_test = Real_Price[len(Real_Price)-prediction_days:]

        # Adaptation to the RNN
        training_set = df_train.values
        training_set = np.reshape(training_set, (len(training_set), 1))
        sc = MinMaxScaler()
        training_set = sc.fit_transform(training_set)
        X_train = training_set[0:len(training_set)-1]
        y_train = training_set[1:len(training_set)]
        X_train = np.reshape(X_train, (len(X_train), 1, 1))

        regressor = create_model()
        regressor.fit(X_train, y_train, batch_size=5, epochs=20)

        # Create directory
        tf_path = path.join(model_path, "tf-model")
        sc_path = path.join(model_path, "scaler")
        makedirs(model_path, exist_ok=True)
        makedirs(tf_path, exist_ok=True)
        makedirs(sc_path, exist_ok=True)

        # Save the model
        regressor.save(tf_path)

        # Save the scaler
        import joblib
        with open(path.join(sc_path, 'scaler.bin'), 'wb') as f:
            joblib.dump(sc, f)
    except Exception as e:
        makedirs(failure_path, exist_ok=True)
        with open(path.join(failure_path, 'failure'), 'w') as f:
            f.write('Exception during training: ' + str(e))
        sys.exit(255)

if __name__ == '__main__':
    train()
    sys.exit(0)