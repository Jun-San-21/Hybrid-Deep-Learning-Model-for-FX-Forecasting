import pandas as pd
import os
import matplotlib.pyplot as plt
import mplfinance as mpf
from keras.callbacks import EarlyStopping
from sklearn.metrics import log_loss
import numpy as np

def plot_close_price(currency_pairs, titles, nrows, ncols, figsize):
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for count, (pair,name) in enumerate(zip(currency_pairs,titles)):
        pair.plot(ax=axes[count], y="Close", title=name)
        axes[count].set_ylabel('Close Price')
        axes[count].legend(title='Price')

    plt.tight_layout()
    plt.show()

def plot_ohlc_candle(currency_pairs, titles, nrows, ncols=1, figsize=(30,30)):
    # Plotting
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    for count, (pair,name) in enumerate(zip(currency_pairs,titles)):
        mpf.plot(pair[:96], type='candle', ax=axes[count])
        axes[count].set_title(name)

    plt.tight_layout()
    plt.show()

def save_metrics_to_csv(df_metrics, model_name, currency_pair):
    
    base_dir = '../data/results/'
    filename = os.path.join(base_dir, f'{currency_pair}.csv')
    df_metrics['model'] = model_name
    
    if not os.path.isfile(filename):
        # File doesn't exist, write with header
        df_metrics.to_csv(filename, index=False)
    else:
        # File exists, append without writing the header
        df_metrics.to_csv(filename, mode='a', index=False, header=False)
        
def cross_val(model, X, y, epochs=50, batch_size=128):
    n_samples = len(X)
    n_splits = 5
    fold_size = n_samples // n_splits
    history = []
    loss_set = []
    for i in range(0, n_splits):
        train_end = (i + 1) * fold_size
        val_start = train_end
        val_end = val_start + fold_size

        if val_end > n_samples:
            break
        print(f"Split {i+1}, Train sample {train_end}, Val sample {val_end-val_start}")
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history_split = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stop],
                            shuffle=False)
        
        pred = model.predict(X_val)
        loss = log_loss(y_val, pred)
        loss_set.append(loss)
        
        
        history.append(history_split)
    mean_loss = np.mean(np.array(loss_set))
    print(f'Mean Log Loss: {mean_loss}')
    
    return history, mean_loss

def train_model(model, X_train, y_train, epochs=50, batch_size=128):
    
    early_stop = EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stop])
    
    return history