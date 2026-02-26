import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import cv2

# ================= SETTINGS =================
DATA_DIR = "data"
IMG_HEIGHT = 224
IMG_WIDTH  = 512
BATCH_SIZE = 32
SEED = 42
EPOCHS = 15

CLASS_NAMES = ["normal", "mi", "abnormal"]
NUM_CLASSES = 3


# ================= LOAD DATASETS =================
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\n✅ Class mapping:", train_ds.class_names)


# ================= NORMALIZATION =================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(1000).cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)


# ================= FOCAL LOSS =================
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        fl = weight * cross_entropy
        return tf.reduce_sum(fl, axis=1)
    return loss


# ================= MODEL =================
def build_model():
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # normalization
    x = tf.keras.layers.Rescaling(1./255)(inputs)

    # Block 1
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Block 4
    x = tf.keras.layers.Conv2D(256, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,1))(x)

    # CNN → Sequence
    # CNN → Sequence (safe + ECG-aware)
    h = int(x.shape[1])
    w = int(x.shape[2])
    c = int(x.shape[3])

    x = tf.keras.layers.Reshape((w, h * c))(x)

    # LSTM
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)

    # Attention
    attention_output = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.GlobalAveragePooling1D()(attention_output)

    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)


model = build_model()
model.summary()
for layer in model.layers:
    print(layer.name, layer.__class__.__name__)
# ================= GRAD-CAM FUNCTION =================
# ================= HELPER FUNCTION =================
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()


# ================= COMPILE =================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=focal_loss(),
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-6,
    verbose=1
)
# ================= CALLBACKS =================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    lr_scheduler
]


# ================= TRAIN =================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ================= TRAINING GRAPH =================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Accuracy Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Loss Curve")
plt.legend()

plt.tight_layout()
plt.show()


# ================= EVALUATION =================
test_loss, test_acc, test_prec, test_rec = model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nAccuracy : {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall   : {test_rec:.4f}")


# ================= PREDICTIONS (FIXED METHOD) =================
predictions = model.predict(test_ds, verbose=1)
y_pred = np.argmax(predictions, axis=1)

y_true = np.concatenate(
    [np.argmax(y.numpy(), axis=1) for x, y in test_ds],
    axis=0
)


# ================= CONFUSION MATRIX =================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# ================= CLASSIFICATION REPORT =================
print("\n✅ Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


# ================= ROC CURVE =================
y_true_bin = label_binarize(y_true, classes=[0,1,2])

plt.figure(figsize=(7,6))

for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], predictions[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})")

plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest)")
plt.legend()
plt.grid(True)
plt.show()

# ================= GRAD-CAM VISUALIZATION =================

# choose one batch from test set
for images, labels in test_ds.take(1):
    sample_image = images[0:1]
    true_label = np.argmax(labels[0].numpy())
    break

# generate heatmap
heatmap = make_gradcam_heatmap(
    sample_image,
    model,
    last_conv_layer_name=get_last_conv_layer(model)
)

# convert image for display
img = sample_image[0].numpy()

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + (img * 255).astype(np.uint8)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title(f"Original ECG (True: {CLASS_NAMES[true_label]})")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(superimposed_img.astype("uint8"))
plt.title("Grad-CAM Attention")
plt.axis("off")

plt.show()


