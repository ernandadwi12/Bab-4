# Example 4.13
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.imagenet_utils import decode_predictions

image_size = 224
model = vgg16.VGG16(weights='imagenet')
print(model.summary())

camera = cv2.VideoCapture(0)

while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break

    # resize frame ke 224x224 untuk input VGG16
    frame = cv2.resize(cam_frame, (image_size, image_size))
    # OpenCV memakai BGR, VGG16 preprocess_input sudah mengubahnya sendiri
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = vgg16.preprocess_input(image_batch.copy())

    # prediksi
    predictions = model.predict(processed_image, verbose=0)
    label = decode_predictions(predictions)

    # tampilkan hasil di frame
    cv2.putText(cam_frame,
                "VGG16: {}, {:.1f}".format(label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('video image', cam_frame)

    key = cv2.waitKey(30)
    if key == 27:  # tekan ESC untuk keluar
        break

camera.release()
cv2.destroyAllWindows()