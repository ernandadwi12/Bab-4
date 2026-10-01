#Example 4.4 LeNet-5 Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D
from keras import optimizers

model = Sequential(name='sequential_1')
model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu',
                 input_shape=(32, 32, 1), name='conv2d_1'))
model.add(AveragePooling2D(pool_size=(2, 2), name='average_pooling2d_1'))
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', name='conv2d_2'))
model.add(AveragePooling2D(pool_size=(2, 2), name='average_pooling2d_2'))
model.add(Flatten(name='flatten_1'))
model.add(Dense(units=120, activation='relu', name='dense_1'))
model.add(Dense(units=84, activation='relu', name='dense_2'))
model.add(Dense(units=10, activation='softmax', name='dense_3'))
model.summary()