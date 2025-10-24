import sys
print(sys.executable)

import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

# data Loading
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    'data_split/train',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    'data_split/val',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    'data_split/test',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# layers
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(256, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Summary
model.summary()

# 5Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# Evaluate
loss, acc = model.evaluate(test_data)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

# Save
os.makedirs('models', exist_ok=True)
model.save('models/cnn_ecg_model.h5')
print("✅ Model saved as models/cnn_ecg_model.h5")

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy Curve")
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss Curve")
plt.show()
