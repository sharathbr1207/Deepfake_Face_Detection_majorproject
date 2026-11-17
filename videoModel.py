import os
import math
import random
import numpy as np
from io import BytesIO
import cv2
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, TimeDistributed, LSTM, BatchNormalization
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ============================================================
# MIXED PRECISION (ONLY IF GPU EXISTS)
# ============================================================
from tensorflow.keras import mixed_precision
if tf.config.list_physical_devices("GPU"):
    mixed_precision.set_global_policy("mixed_float16")

# ============================================================
# CONFIG (VIDEO ONLY)
# ============================================================
DATASET_ROOT = "dataset"
IMG_SIZE = (128, 128)
NUM_FRAMES = 20
BATCH_VIDEO = 4
EPOCH_VIDEO = 15
LR = 1e-4
ELA_QUALITY = 90

OUTPUT_VIDEO_MODEL = "ela_cnn_lstm_video_model.keras"

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# FACE DETECTOR (OPENCV HAAR CASCADE)
# ============================================================
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

def detect_and_crop_face(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    if w * h < 0.05 * img.shape[0] * img.shape[1]:
        return None

    face_img = img[y:y+h, x:x+w]
    if face_img.size == 0:
        return None

    return Image.fromarray(face_img)

# ============================================================
# ELA FUNCTION
# ============================================================
def convert_to_ela(pil_img, quality=ELA_QUALITY):
    try:
        buf = BytesIO()
        pil_img.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        compressed = Image.open(buf)

        ela_img = ImageChops.difference(pil_img, compressed)
        extrema = ela_img.getextrema()
        max_diff = max([e[1] for e in extrema])
        max_diff = max(max_diff, 1)

        scale = 255.0 / max_diff
        ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)
        ela_img = ela_img.resize(IMG_SIZE)

        return np.asarray(ela_img).astype("float32") / 255.0

    except Exception:
        fallback = pil_img.resize(IMG_SIZE)
        return np.asarray(fallback).astype("float32") / 255.0

# ============================================================
# VIDEO GENERATOR
# ============================================================
class VideoSeq(Sequence):
    def __init__(self, files, labels, batch_size=BATCH_VIDEO):
        self.files = files
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.files) / self.batch_size)

    def extract_frames(self, path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total < NUM_FRAMES:
            cap.release()
            return None

        idxs = np.linspace(0, total - 1, NUM_FRAMES).astype(int)
        frames = []

        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = detect_and_crop_face(Image.fromarray(frame))
            if face is not None:
                frames.append(convert_to_ela(face))

        cap.release()

        if len(frames) < int(0.7 * NUM_FRAMES):
            return None

        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1])

        return np.array(frames[:NUM_FRAMES])

    def __getitem__(self, index):
        batch_files = self.files[index*self.batch_size:(index+1)*self.batch_size]
        batch_labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]

        X, y = [], []

        for fp, label in zip(batch_files, batch_labels):
            frames = self.extract_frames(fp)
            if frames is None:
                continue
            X.append(frames)
            y.append(label)

        if len(X) == 0:
            return self.__getitem__((index + 1) % self.__len__())

        return np.array(X), np.array(y)

# ============================================================
# VIDEO MODEL
# ============================================================
def build_video_model():
    inp = Input(shape=(NUM_FRAMES, *IMG_SIZE, 3))

    extractor = Sequential([
        Conv2D(32, 3, activation="relu", padding="same"),
        MaxPooling2D(),
        BatchNormalization(),

        Conv2D(64, 3, activation="relu", padding="same"),
        MaxPooling2D(),
        BatchNormalization(),

        GlobalAveragePooling2D(),
    ])

    x = TimeDistributed(extractor)(inp)
    x = LSTM(128)(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(1, activation="sigmoid", dtype="float32")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(LR),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ============================================================
# TRAINING (VIDEO ONLY)
# ============================================================
def train_video():
    real = [os.path.join(DATASET_ROOT, "videos/real", x)
            for x in os.listdir(DATASET_ROOT + "/videos/real")]
    fake = [os.path.join(DATASET_ROOT, "videos/fake", x)
            for x in os.listdir(DATASET_ROOT + "/videos/fake")]

    files = real + fake
    labels = [0] * len(real) + [1] * len(fake)

    combined = list(zip(files, labels))
    random.shuffle(combined)
    files, labels = zip(*combined)

    split = int(0.8 * len(files))
    train_gen = VideoSeq(files[:split], labels[:split])
    val_gen = VideoSeq(files[split:], labels[split:])

    model = build_video_model()
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCH_VIDEO,
        callbacks=[
            ModelCheckpoint(OUTPUT_VIDEO_MODEL, save_best_only=True),
            EarlyStopping(patience=4, restore_best_weights=True),
            ReduceLROnPlateau(patience=2)
        ]
    )

    model.save(OUTPUT_VIDEO_MODEL)
    print("✅ Video Training Complete")

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    train_video()
