import cv2
from ultralytics import YOLO

# Load the small YOLO model
model = YOLO("yolo11n.pt")

# Only these YOLO classes count as vehicles
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Open the input video and read its size and speed
cap = cv2.VideoCapture("traffic.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video (AVI with MJPG works on every Windows setup)
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))

max_frames = int(fps * 20)  # stop after 20 seconds
frame_no = 0

while cap.isOpened() and frame_no < max_frames:
    ok, frame = cap.read()
    if not ok:
        break

    # Run YOLO on this one frame
    results = model(frame, verbose=False)[0]

    count = 0
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in VEHICLES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, VEHICLES[cls], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            count += 1

    # Show the vehicle count on the frame
    cv2.putText(frame, f"Vehicles: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)
    frame_no += 1

    # Print progress so the terminal doesn't look stuck
    if frame_no % 20 == 0:
        print("Processed", frame_no, "frames")

cap.release()
out.release()
print("Done. Processed", frame_no, "frames. Open output.avi")