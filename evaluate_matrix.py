import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall

from image_model import ImageSequence

# ================= CONFIG =================
DATASET_ROOT = "dataset"
MODEL_PATH = "model/ela_cnn_image_model.keras"
BATCH_SIZE = 32
EVAL_STEPS = 50   # 🔴 limits evaluation (50 batches only)

# ================= LOAD MODEL =================
model = load_model(
    MODEL_PATH,
    compile=False
)

# Re-compile ONLY for evaluation
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        Precision(name="precision"),
        Recall(name="recall")
    ]
)

# ================= LOAD DATA =================
real_dir = os.path.join(DATASET_ROOT, "images/real")
fake_dir = os.path.join(DATASET_ROOT, "images/fake")

real = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
fake = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)]

files = real + fake
labels = [0]*len(real) + [1]*len(fake)

test_gen = ImageSequence(files, labels, batch_size=BATCH_SIZE)

# ================= EVALUATE =================
results = model.evaluate(
    test_gen,
    steps=EVAL_STEPS,   # ✅ does NOT scan full dataset
    verbose=1
)

accuracy  = results[1]
precision = results[2]
recall    = results[3]

f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

# ================= PRINT =================
print("\n📊 MODEL PERFORMANCE (Sampled Evaluation)")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1_score:.4f}")
