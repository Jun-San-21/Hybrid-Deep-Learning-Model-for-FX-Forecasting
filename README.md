# Hybrid Deep Learning Model for Forex Forecasting

This project is a final year project that aims to forecast the closing price of currency pairs through deep learning techniques. Specifically, hybrid deep learning models were developed and compared with their component models. The proposed models were CNN-LSTM, CNN-RNN, LSTM, SRNN and CNN. SRNN and LSTM's are utilized for sequence predictions whereas CNN's for feature extraction.

Key Techniques: Rolling Cross-Validation, Hyperparameter Tuning, Pruning, Feature Engineering, EDA, Dickey-Fuller Test, ROC AUC

--------------
## Empirical Scenario
The target was set to predict the closing price of the 7 most traded currency pairs. These were eur/usd, usd/jpy, gbp/usd, nzd/usd, usd/cad, usd/chf and aud/usd. The output target was set to follow a binary classification problem whereby "1" signals a price increase and "0" for a price decrease. 

---
## Dataset
The dataset consists of the OHLC data with the inclusion of technical indicators; Relative Strength Index (RSI) and Exponential Moving Average of time lenghts 5, 10 and 20. 

---------
## Technologies
- Jupyter Lab 3.3.2 
- Python 3.9.12
- Keras 3.4.1
- TensorFlow 2.10.0 

------
## Usage

Open `/notebooks/...`

Execute in order: 

|Notebooks|Description|
|--------------------------|-|
|_01_data_preprocessing_1_|Cleaning and Feature engineering|
| _01_data_preprocessing_2_|Data transformation and train/test split|
|_02_lstm_|Executes the lstm train and results|
|_03_simplernn_|Executes the srnn train and results|
|_04_cnn_|Executes the cnn train and results|
|_05_cnn-lstm_|Executes the cnn-lstm train and results|
|_06_cnn-srnn_|Executes the cnn-srnn train and results|
|_07_evaluation_|Comparative analysis of all models results|
|eda|Exploratory data analysis|

### Setting Parameters (Optional)
If you would like to set your own configurations...
#### Hyperparameter tuning

Open `/src/...`


For _batch size, epochs, and early stopping patience value_: 

In `utils.py`, locate `cross_val(model, X, y, epochs=50, batch_size=128)` (for cross validation) or `train_model(model, X_train, y_train, epochs=50, batch_size=128)` (for model training) and set epochs and batch size (Default at 50, 128). To configure individually per model, locate respective notebooks and set parameter. For early stopping, locate `early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)` and set patience value (Default at 5). 

For _model layers, hidden units, learning rate etc._ :

Locate respective `"modelname"_model.py` file. Tune as desired under function, eg. `create_cnn_lstm_model(timestep, features)`-> `optimizer=Adam(learning_rate=0.0001)`, `model.add(LSTM(units=64)`. 

#### Past predictions/Sequence length
For _sequence length_:
Open `../notebooks/_01_data_preprocessing_2` and locate `sequence_length = 30`.

#### Technical indicators
Note: Pandas_ta was used to to develop the indicators

Open `/src/preprocessing.py`, locate 

```
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
```
-------

That is all. 




## Authors

- [@tanjunsan](https://www.github.com/Jun-San-21)


