#Exercise 4.6 - tambah Conv2D, MaxPooling, Dropout sebelum Flatten
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


def build(extra_filters=None):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    if extra_filters:                      # blok tambahan
        model.add(Conv2D(extra_filters, (5, 5), activation='relu'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


# Model asli (Example 4.7) dan model dengan blok tambahan
original = build()
modified = build(extra_filters=64)

print("=== Model asli ===")
original.summary()
print("\n=== Model dengan Conv2D + MaxPooling + Dropout tambahan ===")
modified.summary()

print("\nPerbandingan jumlah parameter")
print(f"Asli     : {original.count_params():>9,}")
print(f"Modifikasi: {modified.count_params():>8,}")
print(f"Selisih  : {modified.count_params() - original.count_params():>9,}")