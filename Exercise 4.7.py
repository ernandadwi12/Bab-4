#Exercise 4.7 - bandingkan jumlah filter (32 vs 64) dan ukuran kernel ((5,5) vs (3,3))
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical, set_random_seed

# load data (sekali saja)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

EPOCHS = 10   # sama dengan Example 4.9


def run(filters, kernel):
    set_random_seed(42)   # seed sama agar perbandingan adil
    model = Sequential()
    model.add(Conv2D(filters, kernel, input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    start = time.time()
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=EPOCHS, batch_size=200, verbose=0)
    train_time = time.time() - start
    scores = model.evaluate(X_test, y_test, verbose=0)
    return model.count_params(), train_time, 100 - scores[1] * 100


configs = [(32, (5, 5)),   # asli (Example 4.9)
           (64, (5, 5)),   # ubah jumlah filter
           (32, (3, 3)),   # ubah ukuran kernel
           (64, (3, 3))]   # kombinasi keduanya

print(f"{'Filter':>7} {'Kernel':>8} {'Params':>11} {'Waktu (s)':>10} {'Error (%)':>10}")
for f, k in configs:
    params, t, err = run(f, k)
    print(f"{f:>7} {str(k):>8} {params:>11,} {t:>10.1f} {err:>10.2f}")