import cv2
from ultralytics import YOLO

# Two models: one for normal vehicles, one for ambulances
vehicle_model = YOLO("yolov8n.pt")
ambulance_model = YOLO("best.pt")
print(ambulance_model.names)

# Only these YOLO classes count as vehicles (for yolov8n.pt)
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

AMBULANCE_CONF = 0.75   # raise to reduce false alarms, lower to catch more

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

    # 1) Normal vehicles
    results = vehicle_model(frame, verbose=False)[0]
    count = 0
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in VEHICLES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, VEHICLES[cls], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            count += 1

    # 2) Ambulances (class name "Ambulance" in best.pt)
    amb_results = ambulance_model(frame, conf=AMBULANCE_CONF, verbose=False)[0]
    ambulance_count = 0
    for box in amb_results.boxes:
        cls = int(box.cls[0])
        if ambulance_model.names[cls] != "Ambulance":
            continue  # ignore "Misc Vehicle", it is not reliable yet
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"AMBULANCE {conf:.2f}", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ambulance_count += 1

    # Show the counts on the frame
    cv2.putText(frame, f"Vehicles: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if ambulance_count:
        cv2.putText(frame, f"AMBULANCE DETECTED ({ambulance_count})", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)
    frame_no += 1

    # Print progress so the terminal doesn't look stuck
    if frame_no % 20 == 0:
        print("Processed", frame_no, "frames")

cap.release()
out.release()
print("Done. Processed", frame_no, "frames. Open output.avi")