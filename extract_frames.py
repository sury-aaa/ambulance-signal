import cv2, os

os.makedirs("negatives", exist_ok=True)
cap = cv2.VideoCapture("traffic.mp4")
i = saved = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if i % 15 == 0:                 # one frame every 15 frames
        cv2.imwrite(f"negatives/frame_{saved:04d}.jpg", frame)
        saved += 1
    i += 1
cap.release()
print("Saved", saved, "frames")