from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, SimpleRNN, Dense, TimeDistributed, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

def create_cnn_srnn_model(timestep, features):
    
    model = Sequential()
    model.add(Input(shape=(timestep, features, 1)))
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(SimpleRNN(units=64))
    model.add(Dropout(0.2))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid')) 

    model.compile(loss=BinaryCrossentropy(label_smoothing=0.2), optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    
    return model

