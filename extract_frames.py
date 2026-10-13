import cv2
import os

video = r"C:\Users\EmageVision\Documents\head_training_dataset\videos\new_video\video7.mp4"

out_dir = r"C:\Users\EmageVision\Documents\extracted_frames\video6"

os.makedirs(out_dir, exist_ok=True)

# Enter the date manually
date = input("Enter date (YYYY-MM-DD): ")

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
every_n = int(round(fps))  # 1 frame per second

i = 0
saved = 0

while True:
    ok, frame = cap.read()

    if not ok:
        break

    if i % every_n == 0:
        filename = f"{date}_frame{saved:05d}.jpg"
        filepath = os.path.join(out_dir, filename)

        cv2.imwrite(filepath, frame)
        saved += 1

    i += 1

cap.release()

print(f"Saved {saved} frames to {out_dir}")