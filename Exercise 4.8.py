#Exercise 4.8 - tambah Conv2D, MaxPooling, Dropout sebelum Flatten (MNIST)
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

EPOCHS = 10    # sama dengan Example 4.9
REPEATS = 1    # naikkan (misal 3) untuk rata-rata beberapa kali latih


def build(extra_block):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    if extra_block:                                   # blok tambahan
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def run(extra_block):
    times, errs = [], []
    for r in range(REPEATS):
        set_random_seed(42 + r)
        model = build(extra_block)
        start = time.time()
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=EPOCHS, batch_size=200, verbose=0)
        times.append(time.time() - start)
        scores = model.evaluate(X_test, y_test, verbose=0)
        errs.append(100 - scores[1] * 100)
    return model.count_params(), sum(times) / REPEATS, sum(errs) / REPEATS


print(f"{'Model':<28} {'Params':>10} {'Waktu (s)':>10} {'Error (%)':>10}")
for name, extra in [("Asli (Example 4.9)", False),
                    ("+ Conv2D/MaxPool/Dropout", True)]:
    params, t, err = run(extra)
    print(f"{name:<28} {params:>10,} {t:>10.1f} {err:>10.2f}")