import os
import time
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as prep16
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as prep19
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# Path gambar relatif terhadap lokasi script
base_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(base_dir, "Elephant.jpg")

# Load gambar (ukuran input sama untuk VGG16 dan VGG19)
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

def run_model(name, model_class, preprocess):
    t0 = time.time()
    model = model_class(weights='imagenet')
    load_time = time.time() - t0

    x_in = preprocess(x.copy())

    t0 = time.time()
    preds = model.predict(x_in)
    pred_time = time.time() - t0

    results = decode_predictions(preds, top=5)[0]

    print(f"\n===== {name} =====")
    print(f"Jumlah parameter : {model.count_params():,}")
    print(f"Waktu load model : {load_time:.2f} detik")
    print(f"Waktu prediksi   : {pred_time:.2f} detik")
    print("Top-5 prediksi:")
    for i, (_, label, prob) in enumerate(results, 1):
        print(f"  {i}. {label:<20} {prob*100:.2f}%")

run_model("VGG16", VGG16, prep16)
run_model("VGG19", VGG19, prep19)