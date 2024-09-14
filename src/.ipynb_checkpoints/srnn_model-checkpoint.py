from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np

def create_srnn_model(timestep, features, units=8):
    model = Sequential()
    model.add(Input(shape=(timestep, features)))
    model.add(SimpleRNN(units))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=BinaryCrossentropy(label_smoothing=0.2), optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    
    return model


