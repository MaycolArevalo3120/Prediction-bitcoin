import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime as dt

# Cargar datos
file_path = 'bitcoin_2020-02-16_2025-02-14.csv'
df = pd.read_csv(file_path)
df['Start'] = pd.to_datetime(df['Start'])
df.sort_values('Start', inplace=True)

# Trabajamos solo con el precio de cierre
close_prices = df['Close'].values.reshape(-1, 1)

# Normalización
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(close_prices)

# Crear secuencias (60 días para predecir 1 día)
x_train = []
y_train = []
sequence_length = 60

for i in range(sequence_length, len(scaled_close)):
    x_train.append(scaled_close[i-sequence_length:i, 0])
    y_train.append(scaled_close[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Construir el modelo LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Precio de cierre

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=20, batch_size=32)

# Pedir al usuario una fecha futura para predecir
fecha_objetivo = input("Ingresa una fecha futura (YYYY-MM-DD): ")
fecha_objetivo = dt.datetime.strptime(fecha_objetivo, '%Y-%m-%d').date()
ultima_fecha = df['Start'].max().date()

dias_a_predecir = (fecha_objetivo - ultima_fecha).days
if dias_a_predecir <= 0:
    print("La fecha debe ser posterior al último dato registrado en el histórico.")
    exit()

# Predicción múltiple hacia adelante (día a día)
last_60_days = scaled_close[-60:]
predictions = []

for _ in range(dias_a_predecir):
    input_seq = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
    predicted_scaled_price = model.predict(input_seq)
    predictions.append(predicted_scaled_price[0, 0])
    last_60_days = np.append(last_60_days, predicted_scaled_price)[-60:]

# Desnormalizar la última predicción (la correspondiente a la fecha objetivo)
predicted_price = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

print(f"Predicción del precio de cierre de Bitcoin para {fecha_objetivo}: ${predicted_price[-1][0]:,.2f}")

# Opcional: graficar últimas observaciones + predicciones futuras
fechas_predichas = [ultima_fecha + dt.timedelta(days=i+1) for i in range(dias_a_predecir)]
plt.figure(figsize=(12, 6))
plt.plot(df['Start'][-100:], df['Close'].values[-100:], label='Histórico')
plt.plot(fechas_predichas, predicted_price, label='Predicción', color='red')
plt.title('Predicción futura del precio de cierre de Bitcoin')
plt.xlabel('Fecha')
plt.ylabel('Precio USD')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
