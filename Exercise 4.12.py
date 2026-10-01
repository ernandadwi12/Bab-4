# Exercise 4.12: Example 4.14 + DenseNet
import time  # tambahan untuk perbandingan performa
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3,
    densenet  # tambahan
)
import numpy as np

# init the models
# model = vgg16.VGG16(weights='imagenet')
# model = resnet50.ResNet50(weights='imagenet')
# model = mobilenet.MobileNet(weights='imagenet')
# model = inception_v3.InceptionV3(weights='imagenet')
model = densenet.DenseNet121(weights='imagenet')  # tambahan
print(model.summary())

camera = cv2.VideoCapture(0)
image_size = 224
# image_size = 299

times = []  # tambahan: menyimpan waktu prediksi tiap frame

while camera.isOpened():
    ok, cam_frame = camera.read()
    frame = cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)

    # processed_image = vgg16.preprocess_input(image_batch.copy())
    # processed_image = resnet50.preprocess_input(image_batch.copy())
    # processed_image = mobilenet.preprocess_input(image_batch.copy())
    # processed_image = inception_v3.preprocess_input(image_batch.copy())
    processed_image = densenet.preprocess_input(image_batch.copy())  # tambahan

    # get the predicted probabilities for each class
    t0 = time.perf_counter()  # tambahan
    predictions = model.predict(processed_image)
    times.append(time.perf_counter() - t0)  # tambahan
    label = decode_predictions(predictions)

    # format final image visualization to display the results of experiments
    cv2.putText(cam_frame, "{}, {:.1f}".format(label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('video image', cam_frame)

    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

# tambahan: ringkasan performa setelah ESC ditekan
camera.release()
cv2.destroyAllWindows()
if len(times) > 1:
    avg = np.mean(times[1:])  # frame pertama diabaikan (inisialisasi)
    print("Parameter: {:,}".format(model.count_params()))
    print("Rata-rata waktu prediksi: {:.1f} ms/frame ({:.1f} FPS)".format(avg * 1000, 1 / avg))