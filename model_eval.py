import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

# ========= CONFIG =========
MODEL_PATH = "best_ecg_cnn.h5"   # or "ecg_cnn_final.h5"
IMG_SIZE = (160, 160)            # MUST be same as in train_model.py
BATCH_SIZE = 32
DATA_DIR = Path("data")          # base data folder
# ==========================

# 1. Load test dataset
test_dir = DATA_DIR / "test"

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = test_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# 2. Load trained model
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

# 3. Collect predictions + true labels
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 4. Compute metrics
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n======================")
print(f"Test accuracy: {acc:.4f}")
print("======================\n")

print("Confusion Matrix (rows = true, cols = predicted):")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# 5. Plot confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Write numbers inside boxes
thresh = cm.max() / 2.0
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black"
        )

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()
