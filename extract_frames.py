import cv2
import os

def extract_video_frames(video_dir, output_dir, frames_per_video=10):
    os.makedirs(output_dir, exist_ok=True)

    classes = ['real', 'fake']
    for cls in classes:
        input_path = os.path.join(video_dir, cls)
        output_path = os.path.join(output_dir, cls)
        os.makedirs(output_path, exist_ok=True)

        for filename in os.listdir(input_path):
            if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                continue

            video_path = os.path.join(input_path, filename)
            cap = cv2.VideoCapture(video_path)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(total_frames // frames_per_video, 1)

            count = 0
            saved = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if count % interval == 0:
                    out_name = f"{filename}_{saved}.jpg"
                    cv2.imwrite(os.path.join(output_path, out_name), frame)
                    saved += 1

                count += 1

            cap.release()

    print("Frame extraction complete!")

# -----------------------------
# Use project folder structure
# -----------------------------
video_dir = "dataset/videos"     # contains /real and /fake
output_dir = "dataset/frames"    # will create /real and /fake inside it

extract_video_frames(video_dir, output_dir, frames_per_video=10)
