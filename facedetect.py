# -*- coding: utf-8 -*-
"""
Face Detection Model Training Script
Optimized: Simple LeNet-style architecture for small datasets.
"""

import os
import pickle
import numpy as np
import collections
import tensorflow as tf

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

# -------------------------
# Config
# -------------------------
RANDOM_STATE = 42
# CHANGED: Increased Batch Size to 80. 
# approx 450 training images / 80 = ~6 steps per epoch (reducing 28 -> 6)
BATCH_SIZE = 80  
EPOCHS = 15
TARGET_SIZE = (160, 160)
FINAL_MODEL_FILENAME = "final_model.h5"

# -------------------------
# 1) Load data
# -------------------------
print("Loading data...")
with open("images.p", "rb") as f:
    images = pickle.load(f)
with open("labels.p", "rb") as f:
    labels = pickle.load(f)

images = np.asarray(images)
labels = np.asarray(labels)

# -------------------------
# 2) Preprocess images
# -------------------------
print("Preprocessing images...")
images = images.astype("float32")

# Reshape if needed (N, H, W) -> (N, H, W, 1) or 3
if images.ndim == 3:
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
if images.shape[-1] == 1:
    images = np.concatenate([images, images, images], axis=-1)

# Resize
images_resized = tf.image.resize(images, TARGET_SIZE).numpy()

# MobileNet Preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
images_pre = mobilenet_preprocess(images_resized)

# -------------------------
# 3) Labels & Weights
# -------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
class_names = le.classes_
num_classes = len(class_names)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

# -------------------------
# 4) Stratified Split
# -------------------------
# Split 1: Train+Val (80%) vs Test (20%)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
train_idx, test_idx = next(sss.split(images_pre, y_encoded))

X_train_full = images_pre[train_idx]
y_train_full = y_encoded[train_idx]
X_test = images_pre[test_idx]
y_test = y_encoded[test_idx]

# Split 2: Train (80%) vs Val (20%)
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
train_idx2, val_idx2 = next(sss2.split(X_train_full, y_train_full))

X_train = X_train_full[train_idx2]
y_train = y_train_full[train_idx2]
X_val = X_train_full[val_idx2]
y_val = y_train_full[val_idx2]

print(f"Train shape: {X_train.shape} (Will result in approx {len(X_train)//BATCH_SIZE} steps)")

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# -------------------------
# 5) Generators
# -------------------------
# Stronger augmentation to fight overfitting
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.10,
    zoom_range=0.20,
    horizontal_flip=True,
    fill_mode="reflect"
)

train_generator = train_datagen.flow(
    X_train, y_train_cat, batch_size=BATCH_SIZE, shuffle=True
)

val_datagen = ImageDataGenerator()
val_generator = val_datagen.flow(X_val, y_val_cat, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# 6) Build Model
# -------------------------
base = MobileNetV2(
    input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
# CHANGED: Increased dropout to 0.5 for stronger regularization
x = Dropout(0.5)(x)
# CHANGED: Reduced Dense layer size 128 -> 64 to reduce complexity
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------
# 7) Train
# -------------------------
callbacks = [
    ModelCheckpoint(FINAL_MODEL_FILENAME, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
]

print("TRAINING STAGE 1 (Head)")
history1 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# Fine-Tuning
print("TRAINING STAGE 2 (Fine-Tune)")
base.trainable = True
fine_tune_at = len(base.layers) - 30
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base.layers[fine_tune_at:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Re-flow generator
train_generator = train_datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE, shuffle=True)

history2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# -------------------------
# 8) Evaluate
# -------------------------
print("Saving & Evaluating...")
model.save(FINAL_MODEL_FILENAME, include_optimizer=True)

y_pred_proba = model.predict(X_test, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred_proba, axis=1)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))