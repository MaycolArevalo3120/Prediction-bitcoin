from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime as dt
import os

app = Flask(__name__)
CORS(app)

CSV_PATH = 'bitcoin_2020-02-16_2025-02-14.csv'
SEQUENCE_LENGTH = 60
PLOT_PATH = 'prediccion_transformer.png'

scaler = MinMaxScaler()
df = pd.read_csv(CSV_PATH)
df['Start'] = pd.to_datetime(df['Start'])
df.sort_values('Start', inplace=True)
close_prices = df['Close'].values.reshape(-1, 1)
scaled_close = scaler.fit_transform(close_prices)
ultima_fecha = df['Start'].max().date()

# ----------- Positional Encoding -----------
def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads, dtype=tf.float32)

# ----------- Self-Attention Layer -----------
class TimeSeriesSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.norm = tf.keras.layers.LayerNormalization()
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, activation="relu"),
            tf.keras.layers.Dense(d_model)
        ])

    def call(self, x):
        attn_output = self.mha(x, x)
        x = self.norm(x + attn_output)
        ff_output = self.ff(x)
        return self.norm(x + ff_output)

# ----------- Modelo Transformer -----------
def build_transformer_model(input_shape, d_model=64, num_heads=4, num_layers=2):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(d_model)(inputs)
    x += positional_encoding(input_shape[0], d_model)
    for _ in range(num_layers):
        x = TimeSeriesSelfAttention(d_model, num_heads)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs)

model = build_transformer_model(input_shape=(SEQUENCE_LENGTH, 1))
x_data, y_data = [], []
for i in range(SEQUENCE_LENGTH, len(scaled_close)):
    x_data.append(scaled_close[i - SEQUENCE_LENGTH:i])
    y_data.append(scaled_close[i])
x_data, y_data = np.array(x_data), np.array(y_data)
model.compile(optimizer='adam', loss='mse')
model.fit(x_data, y_data, epochs=20, batch_size=32)

@app.route('/predict', methods=['POST'])
def predict():
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
    precio_final = float(predicted_price[-1][0])

    # Graficar
    fechas_predichas = [ultima_fecha + dt.timedelta(days=i+1) for i in range(dias_a_predecir)]
    plt.figure(figsize=(12, 6))
    plt.plot(df['Start'][-100:], df['Close'].values[-100:], label='Histórico')
    plt.plot(fechas_predichas, predicted_price, label='Predicción (Transformer)', color='orange')
    plt.title('Predicción futura del precio de cierre de Bitcoin (Transformer)')
    plt.xlabel('Fecha')
    plt.ylabel('Precio USD')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    return jsonify({
        "fecha_predicha": fecha_str,
        "precio_estimado": precio_final,
        "imagen_url": "/plot"
    })

@app.route('/plot', methods=['GET'])
def get_plot():
    if os.path.exists(PLOT_PATH):
        return send_file(PLOT_PATH, mimetype='image/png')
    else:
        return jsonify({"error": "Imagen no disponible."}), 404

if __name__ == '__main__':
    app.run(debug=True)
