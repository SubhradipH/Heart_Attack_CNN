import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

MODEL_PATH = "best_ecg_cnn.h5"
IMG_SIZE = (160, 160)   

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["abnormal", "mi", "normal"]

def predict_ecg(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  

    predictions = model.predict(img_array)
    probs = predictions[0]
    index = np.argmax(probs)
    predicted_class = class_names[index]
    confidence = probs[index]

    print("Raw probabilities:")
    for cls, p in zip(class_names, probs):
        print(f"  {cls}: {p:.3f}")

    print(f"\nPredicted class : {predicted_class}")
    print(f"Confidence      : {confidence:.3f}")

    # ---------- MESSAGE PART ----------
    if predicted_class == "normal":
        print("\n✅ RESULT: ECG appears NORMAL No need to worry.")
        print("   Everything looks OK .")
    else:
        print("\n⚠️  RESULT: WARNING . You should go to a doctor")
        print(f"   The model detected: {predicted_class.upper()}.")
       

if __name__ == "__main__":
    # put your test image path here
    path = "Normal(122)_lead_12.jpg"          # or e.g. "data/test/mi/MI(1).jpg"
    predict_ecg(path)
