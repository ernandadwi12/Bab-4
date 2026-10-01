from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D, Input


def build_lenet(f1, f2):
    model = Sequential()
    model.add(Input(shape=(32, 32, 1)))
    model.add(Conv2D(filters=f1, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=f2, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    return model


# (filter conv1, filter conv2); (6, 16) adalah versi asli di buku
configs = [(3, 8), (6, 16), (12, 32), (16, 32), (32, 64)]

print(f"{'Conv1':>6} {'Conv2':>6} {'Conv1 par':>10} {'Conv2 par':>10} "
      f"{'Dense1 par':>11} {'Total':>9}")
for f1, f2 in configs:
    m = build_lenet(f1, f2)
    p = [l.count_params() for l in m.layers]
    # urutan layer: conv1, pool, conv2, pool, flatten, dense1, dense2, dense3
    print(f"{f1:>6} {f2:>6} {p[0]:>10} {p[2]:>10} {p[5]:>11} {m.count_params():>9}")

# Untuk melihat summary salah satu model:
# build_lenet(12, 32).summary()