import os
import math
import random
import numpy as np
from io import BytesIO
import cv2
from PIL import Image, ImageChops, ImageEnhance

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ===================== GPU / SEED =====================
if tf.config.list_physical_devices("GPU"):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ===================== CONFIG =====================
DATASET_ROOT = "dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-4
ELA_QUALITY = 90
MODEL_PATH = "ela_cnn_image_model.keras"

# ===================== FACE DETECTOR =====================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def face_present(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    return len(faces) > 0

# ===================== ELA =====================
def convert_to_ela(pil_img, quality=ELA_QUALITY):
    try:
        buffer = BytesIO()
        pil_img.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)

        ela = ImageChops.difference(pil_img, compressed)
        extrema = ela.getextrema()
        max_diff = max(e[1] for e in extrema)
        max_diff = max(max_diff, 1)

        scale = 255.0 / max_diff
        ela = ImageEnhance.Brightness(ela).enhance(scale)
        ela = ela.resize(IMG_SIZE)

        return np.asarray(ela).astype("float32") / 255.0

    except Exception:
        fallback = pil_img.resize(IMG_SIZE)
        return np.asarray(fallback).astype("float32") / 255.0

# ===================== DATA GENERATOR =====================
class ImageSequence(Sequence):
    def __init__(self, files, labels, batch_size=BATCH_SIZE):
        self.files = files
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.files) / self.batch_size)

    def __getitem__(self, idx):
        batch_files = self.files[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_labels = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]

        X, y = [], []

        for fp, label in zip(batch_files, batch_labels):
            try:
                img = Image.open(fp).convert("RGB")

                # Face is only a gate, NOT cropped
                if not face_present(img):
                    continue

                ela = convert_to_ela(img)
                X.append(ela)
                y.append(label)

            except Exception:
                continue

        if len(X) < 8:
            return self.__getitem__((idx + 1) % self.__len__())

        return np.array(X), np.array(y)

# ===================== MODEL =====================
def build_model():
    model = Sequential([
        Input(shape=(*IMG_SIZE, 3)),

        Conv2D(32, 3, activation="relu", padding="same"),
        MaxPooling2D(),
        BatchNormalization(),

        Conv2D(64, 3, activation="relu", padding="same"),
        MaxPooling2D(),
        BatchNormalization(),

        Conv2D(128, 3, activation="relu", padding="same"),
        MaxPooling2D(),
        BatchNormalization(),

        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid", dtype="float32")
    ])

    model.compile(
        optimizer=Adam(LR),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ===================== TRAIN =====================
def train():
    real_dir = os.path.join(DATASET_ROOT, "images/real")
    fake_dir = os.path.join(DATASET_ROOT, "images/fake")

    real = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
    fake = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)]

    print(f"Real images: {len(real)}")
    print(f"Fake images: {len(fake)}")

    files = real + fake
    labels = [0]*len(real) + [1]*len(fake)

    data = list(zip(files, labels))
    random.shuffle(data)
    files, labels = zip(*data)

    split = int(0.8 * len(files))
    train_gen = ImageSequence(files[:split], labels[:split])
    val_gen = ImageSequence(files[split:], labels[split:])

    class_weight = {
        0: len(fake) / len(real),
        1: 1.0
    }

    model = build_model()
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=[
            ModelCheckpoint(MODEL_PATH, save_best_only=True),
            EarlyStopping(patience=4, restore_best_weights=True),
            ReduceLROnPlateau(patience=2)
        ]
    )

    model.save(MODEL_PATH)
    print("Model saved:", MODEL_PATH)

# ===================== PREDICT =====================
def predict_image(model, img_path):
    img = Image.open(img_path).convert("RGB")

    if not face_present(img):
        return "Non-human / No face detected"

    ela = convert_to_ela(img)
    ela = np.expand_dims(ela, axis=0)

    score = model.predict(ela)[0][0]

    if 0.4 < score < 0.6:
        return f"UNCERTAIN ({score:.2f})"

    return f"FAKE ({score:.2f})" if score >= 0.5 else f"REAL ({score:.2f})"

# ===================== MAIN =====================
if __name__ == "__main__":
    train()
