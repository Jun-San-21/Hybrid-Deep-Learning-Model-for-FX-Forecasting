import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.stattools import adfuller

def load_mt5_data(file_path): #load csv file and set columns
    column_names=["Date","Time","Open","High","Low","Close","TickVol","Vol","Spread"]
    return pd.read_csv(file_path, names=column_names, header=0)

def set_datetime_index(df):
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)

def drop_col(df, col):
    df.drop(columns=col, inplace=True)

def clean_data(df): #Drop null rows and duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
def feature_engineering(df): #include technical indicators
    df['RSI'] = ta.rsi(df['Close'], length=7)
    df['EMA_5'] = ta.ema(df['Close'], length=5)
    df['EMA_10'] = ta.ema(df['Close'], length=10)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    #MACD
    #macd_df = ta.macd(df['Close'])
    #df['MACD'] = macd_df['MACD_12_26_9']
    #df['MACD_S'] = macd_df['MACDs_12_26_9']
    #df['MACD_H'] = macd_df['MACDh_12_26_9']
    df.dropna(inplace=True)
    
def convert_stationary_features(features, df):
    for feature in features:
        df[f'{feature}_s'] = df[f'{feature}'].diff()
        df.dropna(inplace=True)
    
def set_target(df): #Set target close price and convert to binary (1 up, 0 down)
    df["NextClose"] = df["Close"].shift(-1) #the proceeding interval
    df["Target"] = (df["NextClose"] > df["Close"]).astype(int)
    df.drop(df.tail(1).index,inplace=True)

def split_data(X, y, train_size=0.8): #Split data into training and testing
    split_index = int(len(X) * train_size)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train, X_test, y_train, y_test

def preprocess_3d_data(data, feature_col, target_col, sequence_length):
    
    target_values = data[target_col].values
    #scaler = StandardScaler() leads to data leakage
    features = data[feature_col]
    
    X = []
    y = target_values[sequence_length:]  # y values start from sequence_length to the end

    # Generate sequences of length `sequence_length`
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
    
    # Convert lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y
    
    