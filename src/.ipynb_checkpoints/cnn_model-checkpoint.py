from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import log_loss
import numpy as np

def create_cnn_model(timestep, features, kernel_size=4):
    model = Sequential()
    model.add(Input(shape=(timestep, features)))
    model.add(Conv1D(filters=32, kernel_size=kernel_size, activation='relu'))
    #model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu'))
    #model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu'))
    #model.add(Conv1D(filters=128, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    #model.add(Dropout(0.1))
    #model.add(Dense(8, activation='relu'))
    #model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(loss=BinaryCrossentropy(label_smoothing=0.2), optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    
    return model


