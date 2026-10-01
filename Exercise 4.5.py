from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input


def build_alexnet(d1, d2, d3):
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (11, 11), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(d1, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(d2, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(d3, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(17, activation='softmax'))
    return model


# (unit Dense 1, unit Dense 2, unit Dense 3); (4096, 4096, 1000) adalah versi asli di buku
configs = [(512, 512, 128), (1024, 1024, 256), (2048, 2048, 500),
           (4096, 4096, 1000), (8192, 8192, 1000)]

print(f"{'D1':>6} {'D2':>6} {'D3':>6} {'Conv':>10} {'Dense':>12} {'Total':>12}")
for d1, d2, d3 in configs:
    m = build_alexnet(d1, d2, d3)
    conv = sum(l.count_params() for l in m.layers if isinstance(l, Conv2D))
    dense = sum(l.count_params() for l in m.layers if isinstance(l, Dense))
    print(f"{d1:>6} {d2:>6} {d3:>6} {conv:>10,} {dense:>12,} {m.count_params():>12,}")