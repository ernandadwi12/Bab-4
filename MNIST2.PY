#Example 4.9
# Simple CNN for the MNIST Dataset
import matplotlib.pyplot as plt                 # DITAMBAH: di buku lupa di-import
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D   # DIUBAH: keras.layers.convolutional sudah tidak ada
from keras.utils import to_categorical          # DIUBAH: np_utils sudah tidak ada


def book_summary(model):
    """Cetak ringkasan model dengan gaya tabel seperti di buku."""
    line = "_" * 65
    print("Using TensorFlow backend.")
    print(f'Model: "{model.name}"')
    print(line)
    print(f"{'Layer (type)':<30}{'Output Shape':<27}Param #")
    print("=" * 65)
    for i, l in enumerate(model.layers):
        label = (l.name + " (" + type(l).__name__ + ")")[:28]
        print(f"{label:<30}{str(l.output.shape):<27}{l.count_params()}")
        if i < len(model.layers) - 1:
            print(line)
    total = model.count_params()
    train = sum(w.numpy().size for w in model.trainable_weights)
    print("=" * 65)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {train:,}")
    print(f"Non-trainable params: {total - train:,}")
    print(line)


# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
plt.subplot(141)
plt.imshow(X_train[0])
plt.subplot(142)
plt.imshow(X_train[1])
plt.subplot(143)
plt.imshow(X_train[2])
plt.subplot(144)
plt.imshow(X_train[3])
plt.show()
# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')   # DIUBAH: jadi satu baris
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')      # DIUBAH: jadi satu baris
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = to_categorical(y_train)               # DIUBAH: tanpa np_utils
y_test = to_categorical(y_test)                 # DIUBAH: tanpa np_utils
num_classes = y_test.shape[1]
# Create model (nama layer disamakan dengan buku)
model = Sequential(name='sequential_1')
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1),
activation='relu', name='conv2d_1'))
model.add(MaxPooling2D(name='max_pooling2d_1'))
model.add(Dropout(0.2, name='dropout_1'))
model.add(Flatten(name='flatten_1'))
model.add(Dense(128, activation='relu', name='dense_1'))
model.add(Dense(num_classes, activation='softmax', name='dense_2'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam',
metrics=['accuracy'])
book_summary(model)    # DITAMBAH: tampilkan ringkasan model gaya buku
print(None)
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
epochs=10, batch_size=200)
# Evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))