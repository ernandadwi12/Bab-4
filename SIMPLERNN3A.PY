# Example 4.20
# SimpleRNN untuk prediksi data Air Passengers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN


# ==========================================
# 1. Mengubah data menjadi dataset matrix
# ==========================================

def convertToMatrix(data, step):
    X, Y = [], []

    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d])
        Y.append(data[d])

    return np.array(X), np.array(Y)


# ==========================================
# 2. Membaca dataset Air Passengers
# ==========================================

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

df = pd.read_csv(
    url,
    usecols=[1],
    engine="python"
)

print(df.head())


# ==========================================
# 3. Menampilkan data
# ==========================================

plt.figure(figsize=(10, 5))
plt.plot(df)
plt.title("Data Air Passengers")
plt.xlabel("Bulan")
plt.ylabel("Jumlah Penumpang")
plt.show()


# ==========================================
# 4. Membagi data training dan testing
# ==========================================

step = 4

N = df.shape[0]

Tp = int(df.shape[0] * 0.8)

values = df.values

train = values[0:Tp, :]
test = values[Tp:N, :]


# ==========================================
# 5. Menambahkan step elemen terakhir
# ==========================================

test = np.append(
    test,
    np.repeat(test[-1,], step)
)

train = np.append(
    train,
    np.repeat(train[-1,], step)
)


# ==========================================
# 6. Membuat data X dan Y
# ==========================================

trainX, trainY = convertToMatrix(train, step)

testX, testY = convertToMatrix(test, step)


# ==========================================
# 7. Reshape data untuk SimpleRNN
# ==========================================

trainX = np.reshape(
    trainX,
    (trainX.shape[0], 1, trainX.shape[1])
)

testX = np.reshape(
    testX,
    (testX.shape[0], 1, testX.shape[1])
)


# ==========================================
# 8. Membuat model SimpleRNN
# ==========================================

model = Sequential()

model.add(
    SimpleRNN(
        units=32,
        input_shape=(1, step),
        activation="relu"
    )
)

model.add(
    Dense(
        8,
        activation="relu"
    )
)

model.add(
    Dense(1)
)


# ==========================================
# 9. Compile model
# ==========================================

model.compile(
    loss="mean_squared_error",
    optimizer="rmsprop"
)


# ==========================================
# 10. Menampilkan struktur model
# ==========================================

model.summary()


# ==========================================
# 11. Melatih model
# ==========================================

history = model.fit(
    trainX,
    trainY,
    epochs=10,
    batch_size=16,
    verbose=2
)


# ==========================================
# 12. Melakukan prediksi
# ==========================================

trainPredict = model.predict(trainX)

testPredict = model.predict(testX)


# ==========================================
# 13. Menggabungkan hasil prediksi
# ==========================================

predicted = np.concatenate(
    (trainPredict, testPredict),
    axis=0
)


# ==========================================
# 14. Evaluasi model
# ==========================================

trainScore = model.evaluate(
    trainX,
    trainY,
    verbose=0
)

print("Training Loss:", trainScore)


# ==========================================
# 15. Menampilkan grafik hasil prediksi
# ==========================================

index = df.index.values

plt.figure(figsize=(12, 6))

plt.plot(
    df,
    label="Data"
)

plt.plot(
    predicted,
    label="Prediction"
)

plt.axvline(
    df.index[Tp],
    color="red",
    linestyle="--",
    label="Batas Training/Testing"
)

plt.title("Prediksi Air Passengers Menggunakan SimpleRNN")
plt.xlabel("Bulan")
plt.ylabel("Jumlah Penumpang")
plt.legend()

plt.show()