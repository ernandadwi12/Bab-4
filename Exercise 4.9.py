#Exercise 4.9 - Example 4.10 dengan model VGG19 (sebelumnya VGG16)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19              # DIUBAH: vgg16 -> vgg19
from tensorflow.keras.applications.vgg19 import preprocess_input   # DIUBAH: vgg16 -> vgg19
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
model = VGG19(weights='imagenet')                                  # DIUBAH: VGG16 -> VGG19
print(model.summary())