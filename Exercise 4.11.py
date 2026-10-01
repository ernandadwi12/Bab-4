# Exercise 4.11: webcam classification VGG16 vs VGG19
import time
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import vgg16, vgg19
from tensorflow.keras.applications.imagenet_utils import decode_predictions

image_size = 224

model16 = vgg16.VGG16(weights='imagenet')
model19 = vgg19.VGG19(weights='imagenet')

camera = cv2.VideoCapture(0)

# statistik untuk perbandingan
t16_list, t19_list = [], []
agree, total = 0, 0


def classify(model, preprocess, batch):
    t0 = time.perf_counter()
    preds = model.predict(preprocess(batch.copy()), verbose=0)
    dt = time.perf_counter() - t0
    name, prob = decode_predictions(preds, top=1)[0][0][1:3]
    return name, float(prob), dt


while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break

    # resize ke 224x224 dan ubah BGR (OpenCV) -> RGB
    frame = cv2.resize(cam_frame, (image_size, image_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    batch = np.expand_dims(img_to_array(frame), axis=0)

    n16, p16, t16 = classify(model16, vgg16.preprocess_input, batch)
    n19, p19, t19 = classify(model19, vgg19.preprocess_input, batch)

    t16_list.append(t16)
    t19_list.append(t19)
    total += 1
    agree += int(n16 == n19)

    cv2.putText(cam_frame, "VGG16: {}, {:.2f} ({:.0f} ms)".format(n16, p16, t16 * 1000),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(cam_frame, "VGG19: {}, {:.2f} ({:.0f} ms)".format(n19, p19, t19 * 1000),
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('video image', cam_frame)

    if cv2.waitKey(30) == 27:  # ESC untuk keluar
        break

camera.release()
cv2.destroyAllWindows()

# ringkasan perbandingan
if total > 0:
    # frame pertama diabaikan karena ada overhead inisialisasi
    a16 = np.mean(t16_list[1:]) if total > 1 else t16_list[0]
    a19 = np.mean(t19_list[1:]) if total > 1 else t19_list[0]
    print("\n===== RINGKASAN PERBANDINGAN =====")
    print(f"Jumlah frame           : {total}")
    print(f"Rata-rata waktu VGG16  : {a16 * 1000:.1f} ms ({1 / a16:.1f} FPS)")
    print(f"Rata-rata waktu VGG19  : {a19 * 1000:.1f} ms ({1 / a19:.1f} FPS)")
    print(f"VGG19 lebih lambat     : {(a19 / a16 - 1) * 100:.1f}%")
    print(f"Prediksi top-1 sama    : {agree}/{total} frame ({agree / total * 100:.1f}%)")