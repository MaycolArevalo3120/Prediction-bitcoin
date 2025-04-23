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
MODEL_PATH = 'model_transformer.keras'

# ----------- Cargar y preparar datos -----------
df = pd.read_csv(CSV_PATH)
df['Start'] = pd.to_datetime(df['Start'])
df.sort_values('Start', inplace=True)
ultima_fecha = df['Start'].max().date()

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
data = df[features].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Separar conjunto de entrenamiento y prueba (últimos 10 días para prueba)
test_days = 10
train_data = scaled_data[:-test_days]
test_data = scaled_data[-(test_days + SEQUENCE_LENGTH):]  # para crear secuencias

# ----------- Función para crear secuencias -----------
def create_sequences(data):
    x, y = [], []
    for i in range(SEQUENCE_LENGTH, len(data)):
        x.append(data[i - SEQUENCE_LENGTH:i])
        y.append(data[i][3])  # posición 3 = 'Close'
    return np.array(x), np.array(y)

x_train, y_train = create_sequences(train_data)
x_test, y_test = create_sequences(test_data)

# ----------- Positional Encoding -----------
def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads, dtype=tf.float32)

# ----------- Self-Attention Layer -----------
class TimeSeriesSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model=64, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.norm = tf.keras.layers.LayerNormalization()
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(d_model)
        ])

    def call(self, x):
        attn_output = self.mha(x, x)
        x = self.norm(x + attn_output)
        ff_output = self.ff(x)
        return self.norm(x + ff_output)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return config

# ----------- Modelo Transformer -----------
def build_transformer_model(input_shape, d_model=64, num_heads=4, num_layers=2):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(d_model)(inputs)
    x += positional_encoding(input_shape[0], d_model)
    for _ in range(num_layers):
        x = TimeSeriesSelfAttention(d_model=d_model, num_heads=num_heads)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs)

# ----------- Cargar o entrenar modelo -----------
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
        'TimeSeriesSelfAttention': TimeSeriesSelfAttention
    })
else:
    model = build_transformer_model(input_shape=(SEQUENCE_LENGTH, len(features)))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))
    model.save(MODEL_PATH, save_format='keras')

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

    last_seq = scaled_data[-SEQUENCE_LENGTH:].copy()
    predictions = []

    for _ in range(dias_a_predecir):
        input_seq = np.reshape(last_seq, (1, SEQUENCE_LENGTH, len(features)))
        predicted_scaled_close = model.predict(input_seq, verbose=0)[0, 0]
        new_entry = last_seq[-1].copy()
        new_entry[3] = predicted_scaled_close  # reemplaza solo el Close
        last_seq = np.append(last_seq, [new_entry], axis=0)[-SEQUENCE_LENGTH:]
        predictions.append(predicted_scaled_close)

    predicted_close = scaler.inverse_transform(np.column_stack([
        np.zeros((dias_a_predecir, 3)),  # Open, High, Low (omitidos)
        predictions,                    # Close
        np.zeros((dias_a_predecir, 2))  # Volume, Market Cap
    ]))[:, 3]

    precio_final = float(predicted_close[-1])

    # Métricas reales
    y_pred_test = model.predict(x_test)
    y_real_test = y_test
    y_pred_test_denorm = scaler.inverse_transform(np.column_stack([
        np.zeros((len(y_pred_test), 3)), y_pred_test, np.zeros((len(y_pred_test), 2))
    ]))[:, 3]
    y_real_test_denorm = scaler.inverse_transform(np.column_stack([
        np.zeros((len(y_real_test), 3)), y_real_test.reshape(-1, 1), np.zeros((len(y_real_test), 2))
    ]))[:, 3]

    mae = float(np.mean(np.abs(y_pred_test_denorm - y_real_test_denorm)))
    rmse = float(np.sqrt(np.mean((y_pred_test_denorm - y_real_test_denorm) ** 2)))
    mape = float(np.mean(np.abs((y_real_test_denorm - y_pred_test_denorm) / y_real_test_denorm)) * 100)

    # Graficar
    fechas_predichas = [ultima_fecha + dt.timedelta(days=i + 1) for i in range(dias_a_predecir)]
    plt.figure(figsize=(12, 6))
    plt.plot(df['Start'][-100:], df['Close'].values[-100:], label='Histórico')
    plt.plot(fechas_predichas, predicted_close, label='Predicción (Transformer)', color='orange')
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
        "imagen_url": "/plot",
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2)
    })

@app.route('/plot', methods=['GET'])
def get_plot():
    if os.path.exists(PLOT_PATH):
        return send_file(PLOT_PATH, mimetype='image/png')
    else:
        return jsonify({"error": "Imagen no disponible."}), 404

if __name__ == '__main__':
    app.run(debug=True)
