import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

def create_lstm_model(timestep, features, units=8):
    model = Sequential()
    model.add(Input(shape=(timestep, features)))
    #model.add(LSTM(units, return_sequences=True))
    #model.add(Dropout(0.5))
    model.add(LSTM(units))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=BinaryCrossentropy(label_smoothing=0.2), optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model

