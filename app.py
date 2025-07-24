import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from config import get_model_path, get_plot_path, model_exists, plot_exists, create_ticker_dirs

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- 0. App Configuration & Title ---
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# --- Header ---
with st.container():
    st.title("LSTM Stock Price Predictor")
    st.write("Select a stock from the list, or enter a custom ticker from Yahoo Finance (e.g., 'RELIANCE.NS', 'TATAMOTORS.NS').")
    st.write("The model will be trained live if a pre-trained version for the selected parameters isn't available. This might take a few minutes.")
    st.write("Train the model on tmrw date after the market closes so that you can predict for next day close price")
    

# --- 1. LSTM Model Definition ---


def get_metrics(actuals, predictions):
    actuals = actuals.flatten()
    predictions = predictions.flatten()
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    mape = np.mean(np.abs((actuals - predictions) / np.where(actuals == 0, 1, actuals))) * 100
    smape = np.mean(2 * np.abs(predictions - actuals) / (np.abs(actuals) + np.abs(predictions) + 1e-8)) * 100
    return rmse, mape, smape

def create_tf_model(input_shape, hidden_units, learning_rate):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(hidden_units, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error')
    return model

@st.cache_data
def load_data(ticker_symbol, start, end):
    try:
        data = yf.download(ticker_symbol, start=start, end=end)
        if data.empty:
            st.error(f"No data found for {ticker_symbol} in the selected date range.")
            return None
        return data
    except Exception as e:
        st.error(f"Could not download data for ticker '{ticker_symbol}'. Reason: {e}")
        return None

def create_inout_sequences(input_data, seq_len):
    X, y = [], []
    for i in range(len(input_data) - seq_len):
        X.append(input_data[i:i + seq_len])
        y.append(input_data[i + seq_len, 0])
    return np.array(X), np.array(y)

def get_first_trading_date(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="max")
        if not hist.empty:
            return hist.index[0].to_pydatetime()
        else:
            st.warning(f"No history found for {ticker_symbol}, it may be delisted.")
            return datetime(2005, 1, 1)
    except Exception as e:
        st.warning(f"Could not fetch first trading date for {ticker_symbol}: {e}")
    return datetime(2005, 1, 1)

def is_valid_ticker(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        # A valid ticker should have a shortName or some historical data.
        if stock.info.get('shortName') or not stock.history(period="max").empty:
            return True
        return False
    except Exception:
        return False

# --- 3. Main Content ---
with st.container():
    st.sidebar.title("Stock Selection")
    ticker_list = ["", "RELIANCE.NS", "TATAMOTORS.NS", "SBIN.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "TCS.NS", "HDFC.NS"]
    selected_ticker = st.sidebar.selectbox("Select Stock Ticker:", ticker_list)
    custom_ticker = st.sidebar.text_input("Or Enter Custom Ticker:").upper()

    ticker = custom_ticker if custom_ticker else selected_ticker

    if ticker:
        if not is_valid_ticker(ticker):
            st.sidebar.error(f"The ticker '{ticker}' is not valid or has no data on Yahoo Finance.")
            st.stop()
        first_trade_date = get_first_trading_date(ticker)
    else:
        first_trade_date = datetime(2005, 1, 1)

    end_date = st.sidebar.date_input("End Date", value=datetime.now(), max_value=datetime.now() + timedelta(days=1))
    
    # Set default start date to 10 years before the end date
    default_start_date = end_date - timedelta(days=365 * 10)
    
    # Ensure default start date is not earlier than the first trade date
    if default_start_date < first_trade_date.date():
        default_start_date = first_trade_date.date()

    start_date = st.sidebar.date_input("Start Date", value=default_start_date, min_value=first_trade_date.date(), max_value=end_date)

    # --- Date Validation ---
    if (end_date - start_date).days < 365:
        st.sidebar.error("The difference between the start and end date must be at least 1 year.")
        st.stop()

    st.sidebar.header("Hyperparameter Tuning")
    use_default_hyperparameters = st.sidebar.checkbox("Use Default Hyperparameters", True)

    if use_default_hyperparameters:
        seq_length = 60
        hidden_layer_size = 200
        epochs = 75
        batch_size = 32
        learning_rate = 0.001
        st.sidebar.write("Using Default Hyperparameters:")
        st.sidebar.write(f"- Sequence Length: {seq_length}")
        st.sidebar.write(f"- Hidden Layer Size: {hidden_layer_size}")
        st.sidebar.write(f"- Epochs: {epochs}")
        st.sidebar.write(f"- Batch Size: {batch_size}")
        st.sidebar.write(f"- Learning Rate: {learning_rate}")
    else:
        st.sidebar.write("Select Custom Hyperparameters:")
        seq_length = st.sidebar.slider("Sequence Length", 10, 120, 60)
        hidden_layer_size = st.sidebar.slider("Hidden Layer Size", 50, 500, 200)
        epochs = st.sidebar.slider("Epochs", 25, 200, 75)
        batch_size = st.sidebar.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
        learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01, 0.1], value=0.001)

    predict_button = st.sidebar.button("Predict")

    if predict_button and ticker:
        with st.spinner(f"Running prediction for {ticker}... This may take a moment."):
            try:
                # --- 4. Setup and Configuration ---
                create_ticker_dirs(ticker)
                model_name = f"{ticker}_s{seq_length}_h{hidden_layer_size}_e{epochs}_b{batch_size}_lr{learning_rate}.keras"
                model_path = get_model_path(ticker, model_name)
                plot_name = model_name.replace('.keras', '.png')
                plot_path = get_plot_path(ticker, plot_name)

                # --- 5. Data Fetching and Preprocessing --
                df = load_data(ticker, start_date, end_date)

                if df is None or df.empty:
                    if end_date > datetime.now().date():
                        st.error(f"Could not fetch data for '{ticker}'. Data for future dates is not available yet. Please select today or an earlier date.")
                    else:
                        st.error(f"Could not download data for ticker '{ticker}'. Please check the ticker symbol and the selected date range.")
                else:
                    # Feature Engineering
                    df['SMA'] = df['Close'].rolling(window=20).mean()
                    df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                    df['Day_of_week'] = df.index.dayofweek
                    df['Month'] = df.index.month
                    df.dropna(inplace=True)

                    if len(df) < seq_length * 2:
                        st.error(f"Not enough data for the selected date range and sequence length. Please select a longer date range.")
                        st.stop()

                    FEATURES = ['Close', 'Volume', 'Open', 'High', 'Low', 'SMA', 'EMA', 'RSI', 'Day_of_week', 'Month']
                    INPUT_SIZE = len(FEATURES)
                    
                    # --- 6. Train-Test Split ---
                    test_data_size = int(len(df) * 0.2)
                    train_data_df = df[:-test_data_size]
                    test_data_df = df[-test_data_size:]

                    # --- 7. Scaling ---
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    train_data_scaled = scaler.fit_transform(train_data_df[FEATURES].values)
                    test_data_scaled = scaler.transform(test_data_df[FEATURES].values)

                    close_price_scaler = MinMaxScaler(feature_range=(0, 1))
                    close_price_scaler.fit(train_data_df[['Close']])

                    X_train, y_train = create_inout_sequences(train_data_scaled, seq_length)

                    if len(X_train) == 0:
                        st.error(f"Not enough data for the selected date range and sequence length to create a training set. Please select a longer date range.")
                        st.stop()
                    X_test, y_test = create_inout_sequences(test_data_scaled, seq_length)

                    if len(X_test) == 0:
                        st.error(f"Not enough data for the selected date range and sequence length to create a test set. Please select a longer date range.")
                        st.stop()

                    # --- 8. Model Training or Loading ---
                    if not model_exists(model_path):
                        st.info(f"No pre-trained model found for {ticker}. Training a new model...")
                        model = create_tf_model((seq_length, INPUT_SIZE), hidden_layer_size, learning_rate)
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        class StqdmCallback(tf.keras.callbacks.Callback):
                            def on_epoch_end(self, epoch, logs=None):
                                progress_bar.progress((epoch + 1) / epochs)
                                status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {logs['loss']:.6f}")

                        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                                            callbacks=[StqdmCallback()], verbose=0)

                        model.save(model_path)
                        st.success(f"Model trained and saved to {model_path}")
                        progress_bar.empty()
                        status_text.empty()
                    else:
                        st.success(f"Loading pre-trained model for {ticker} from {model_path}")
                        try:
                            model = load_model(model_path)
                        except Exception as e:
                            st.error(f"Could not load model: {e}. Training a new model instead.")
                            model = create_tf_model((seq_length, INPUT_SIZE), hidden_layer_size, learning_rate)
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            class StqdmCallback(tf.keras.callbacks.Callback):
                                def on_epoch_end(self, epoch, logs=None):
                                    progress_bar.progress((epoch + 1) / epochs)
                                    status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {logs['loss']:.6f}")

                            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                                                callbacks=[StqdmCallback()], verbose=0)

                            model.save(model_path)
                            st.success(f"Model trained and saved to {model_path}")
                            progress_bar.empty()
                            status_text.empty()
                    
                    with open(model_path, "rb") as fp:
                        st.download_button(
                            label="Download Model",
                            data=fp,
                            file_name=model_name,
                            mime="application/octet-stream"
                        )

                    # --- 9. Prediction and Evaluation ---
                    test_predictions_scaled = model.predict(X_test)
                    actual_predictions = close_price_scaler.inverse_transform(test_predictions_scaled)
                    actuals = test_data_df['Close'].values[seq_length:]

                    rmse, mape, smape = get_metrics(actuals, actual_predictions.flatten())

                    full_scaled_data = scaler.transform(df[FEATURES].values)
                    last_sequence = np.expand_dims(full_scaled_data[-seq_length:], axis=0)
                    next_day_prediction_scaled = model.predict(last_sequence)
                    next_day_prediction = close_price_scaler.inverse_transform(next_day_prediction_scaled)
                    
                    st.subheader(f"Predicted Close Price for next trading day:")
                    st.metric(label=f"{ticker}", value=f"₹{next_day_prediction[0][0]:.2f}")

                    st.subheader("Model Performance on Unseen Test Data")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("RMSE", f"₹{rmse:.2f}")
                    col2.metric("MAPE", f"{mape:.2f}%")
                    col3.metric("SMAPE", f"{smape:.2f}%")

                    # --- 10. Plotting ---
                    plot_index = test_data_df.index[seq_length:]

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(plot_index, actuals, label='Actual Price', color='blue')
                    ax.plot(plot_index, actual_predictions, label='Predicted Price', color='red', linestyle='--')
                    ax.set_title(f'{ticker} Price Prediction on Test Set')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price (INR)')
                    ax.legend()
                    ax.grid(True)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    fig.savefig(plot_path)
                    st.info(f"Plot saved to {plot_path}")
                    
                    with open(plot_path, "rb") as fp:
                        st.download_button(
                            label="Download Plot",
                            data=fp,
                            file_name=plot_name,
                            mime="image/png"
                        )
                    plt.close(fig)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("This could be due to an invalid ticker, no data available, or a model training issue. Please check the ticker and try again.")


# --- Footer ---
with st.container():
    st.write("---")
    st.header("Disclaimer")
    st.write("This is a tool for educational purposes and not financial advice. Always conduct your own thorough research before making any investment decisions.")
    st.write("Stock market predictions are inherently uncertain, and past performance is not indicative of future results.")
