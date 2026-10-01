#Exercise 4.3 - Perbandingan layer Dense 10 neuron vs 5 neuron
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import set_random_seed
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.3, random_state=7)

# Standardisasi fitur (fit hanya pada data training)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


def run(units):
    set_random_seed(42)  # same seed so the comparison is fair
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    start = time.time()
    model.fit(X_train, y_train, batch_size=50, validation_split=0.2,
              epochs=100, verbose=0)
    train_time = time.time() - start

    loss, mse = model.evaluate(X_test, y_test, verbose=0)
    # The output layer is linear, so threshold at 0.5 to get real accuracy
    pred = (model.predict(X_test, verbose=0).ravel() > 0.5).astype(int)
    acc = np.mean(pred == y_test)
    return model.count_params(), train_time, mse, acc


print(f"{'Units':>6} {'Params':>8} {'Time (s)':>9} {'Test MSE':>9} {'Accuracy':>9}")
for units in (10, 5):
    params, t, mse, acc = run(units)
    print(f"{units:>6} {params:>8} {t:>9.2f} {mse:>9.4f} {acc:>9.4f}")