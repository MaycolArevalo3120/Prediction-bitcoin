from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime as dt
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "bitcoin_lstm_model.h5"
CSV_PATH = "bitcoin_2020-02-16_2025-02-14.csv"
SEQUENCE_LENGTH = 60

scaler = MinMaxScaler(feature_range=(0, 1))
model = None
close_prices = None
df = None
scaled_close = None
ultima_fecha = None


def entrenar_y_guardar_modelo():
    global model, scaler, close_prices, df, scaled_close, ultima_fecha
    df = pd.read_csv(CSV_PATH)
    df['Start'] = pd.to_datetime(df['Start'])
    df.sort_values('Start', inplace=True)

    close_prices = df['Close'].values.reshape(-1, 1)
    scaled_close = scaler.fit_transform(close_prices)

    x_train = []
    y_train = []
    for i in range(SEQUENCE_LENGTH, len(scaled_close)):
        x_train.append(scaled_close[i - SEQUENCE_LENGTH:i, 0])
        y_train.append(scaled_close[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=20, batch_size=32)
    model.save(MODEL_PATH)
    ultima_fecha = df['Start'].max().date()


@app.route('/predict', methods=['POST'])
def predict():
    global model, scaled_close
    data = request.get_json()
    fecha_str = data.get('fecha')

    try:
        fecha_objetivo = dt.datetime.strptime(fecha_str, '%Y-%m-%d').date()
    except Exception:
        return jsonify({"error": "Formato de fecha inválido. Usa YYYY-MM-DD."}), 400

    dias_a_predecir = (fecha_objetivo - ultima_fecha).days
    if dias_a_predecir <= 0:
        return jsonify({"error": "La fecha debe ser posterior al último dato registrado."}), 400

    last_60_days = scaled_close[-SEQUENCE_LENGTH:]
    predictions = []

    for _ in range(dias_a_predecir):
        input_seq = np.reshape(last_60_days, (1, SEQUENCE_LENGTH, 1))
        predicted_scaled_price = model.predict(input_seq, verbose=0)
        predictions.append(predicted_scaled_price[0, 0])
        last_60_days = np.append(last_60_days, predicted_scaled_price)[-SEQUENCE_LENGTH:]

    predicted_price = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return jsonify({
        "fecha_predicha": fecha_str,
        "precio_estimado": float(predicted_price[-1][0])
    })


if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        entrenar_y_guardar_modelo()
    else:
        model = load_model(MODEL_PATH)
        df = pd.read_csv(CSV_PATH)
        df['Start'] = pd.to_datetime(df['Start'])
        df.sort_values('Start', inplace=True)
        close_prices = df['Close'].values.reshape(-1, 1)
        scaled_close = scaler.fit_transform(close_prices)
        ultima_fecha = df['Start'].max().date()

    app.run(debug=True)
