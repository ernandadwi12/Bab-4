#Example 4.5 AlexNet with Keras
#Modified from:
#https://engmrk.com/alexnet-implementation-using-keras/
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization  # perbaikan: path lama keras.layers.normalization sudah tidak ada

#Create the AlexNet model
model = Sequential(name='sequential_1')
# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), activation='relu', strides=(4,4), padding='valid', name='conv2d_1'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', name='max_pooling2d_1'))
# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), activation='relu', strides=(1,1), padding='valid', name='conv2d_2'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', name='max_pooling2d_2'))
# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid', name='conv2d_3'))
# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid', name='conv2d_4'))
# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid', name='conv2d_5'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', name='max_pooling2d_3'))
# 1st Fully Connected layer
model.add(Flatten(name='flatten_1'))
model.add(Dense(4096, activation='relu', name='dense_1'))
model.add(Dropout(0.4, name='dropout_1'))
# 2nd Fully Connected Layer
model.add(Dense(4096, activation='relu', name='dense_2'))
model.add(Dropout(0.4, name='dropout_2'))
# 3rd Fully Connected Layer
model.add(Dense(1000, activation='relu', name='dense_3'))
model.add(Dropout(0.4, name='dropout_3'))
# Output Layer
model.add(Dense(17, activation='softmax', name='dense_4'))
model.summary()
# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam', metrics=["accuracy"])
# Fit the model
#model.fit()
# Prediction with the model
#model.evaluate()