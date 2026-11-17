import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import render_template, url_for

app = Flask(__name__)

IMAGE_SIZE = (128, 128)
VIDEO_FRAMES = 20

# -----------------------
# LOAD MODELS
# -----------------------
try:
    image_model = load_model("model/ela_cnn_image_model.keras")
    print("✔ Image model loaded successfully")
except:
    image_model = None
    print("❌ ERROR: Failed to load image model")

try:
    video_model = load_model("model/ela_cnn_lstm_video_model.keras")
    print("✔ Video model loaded successfully")
except:
    video_model = None
    print("❌ ERROR: Failed to load video model")


# -----------------------
# IMAGE PREPROCESSING
# -----------------------
def preprocess_image(path):
    img = image.load_img(path, target_size=IMAGE_SIZE)
    img = image.img_to_array(img)
    img = img / 255.0
    return np.expand_dims(img, axis=0)


# -----------------------
# VIDEO FRAME EXTRACTION
# -----------------------
def extract_frames(video_path, num_frames=VIDEO_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total_frames // num_frames, 1)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame = cv2.resize(frame, IMAGE_SIZE)
            frames.append(frame)
        count += 1

    cap.release()

    frames = np.array(frames[:num_frames])

    # Pad missing frames
    if len(frames) < num_frames:
        missing = num_frames - len(frames)
        pad = np.zeros((missing, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        frames = np.vstack((frames, pad))

    frames = frames / 255.0
    return np.expand_dims(frames, axis=0)


# -----------------------
# IMAGE ANALYSIS ROUTE
# -----------------------
@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    if image_model is None:
        return render_template("result.html", error="Image model is not loaded!")

    file = request.files["file"]
    path = "temp_image.jpg"
    file.save(path)

    img = preprocess_image(path)
    pred = float(image_model.predict(img)[0][0])

    result = "FAKE" if pred >= 0.5 else "REAL"
    confidence = round(pred, 3)

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        filetype="Image"
    )


# -----------------------
# VIDEO ANALYSIS ROUTE
# -----------------------
@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    if video_model is None:
        return render_template("result.html", error="Video model is not loaded!")

    file = request.files["file"]
    path = "temp_video.mp4"
    file.save(path)

    frames = extract_frames(path)
    pred = float(video_model.predict(frames)[0][0])

    result = "FAKE" if pred >= 0.5 else "REAL"
    confidence = round(pred, 3)

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        filetype="Video"
    )


# -----------------------
# HOME ROUTE
# -----------------------
@app.route("/")
def index():
    return render_template("index.html")

# -----------------------
# RUN APP
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)