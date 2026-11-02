import cv2
from ultralytics import YOLO
from signal_controller import SignalController, GREEN, YELLOW

VIDEO_PATH = "traffic.mp4"     # later: your real ambulance clip
AMBULANCE_CONF = 0.5
CONFIRM_FRAMES = 5             # ambulance must appear this many frames in a row
RELEASE_FRAMES = 30            # lane stays priority this long after it disappears
MY_LANE = "Lane 1"             # this video is Lane 1

# For testing the pipeline only: pretend an ambulance appears from 4s to 12s.
# Set to None when you use a real ambulance clip.
TEST_FORCE = (4, 12)

vehicle_model = YOLO("yolov8n.pt")
ambulance_model = YOLO("best.pt")
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

controller = SignalController(["Lane 1", "Lane 2", "Lane 3"],
                              green_time=8, yellow_time=3,
                              all_red_time=1, priority_hold=3)
controller.current = 1         # start with Lane 2 green so the switch is visible

COLOURS = {"GREEN": (0, 200, 0), "YELLOW": (0, 220, 255), "RED": (0, 0, 255)}

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))

max_frames = int(fps * 20)
frame_no = 0
seen_streak = missed_streak = 0
lane_priority = False
last_signals = None

while cap.isOpened() and frame_no < max_frames:
    ok, frame = cap.read()
    if not ok:
        break
    now = frame_no / fps       # video time in seconds

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

    # 2) Ambulances
    amb_results = ambulance_model(frame, conf=AMBULANCE_CONF, verbose=False)[0]
    ambulance_count = 0
    for box in amb_results.boxes:
        cls = int(box.cls[0])
        if ambulance_model.names[cls] != "Ambulance":
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"AMBULANCE {conf:.2f}", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ambulance_count += 1

    forced = TEST_FORCE is not None and TEST_FORCE[0] <= now < TEST_FORCE[1]
    seen = ambulance_count > 0 or forced

    # 3) Lane priority logic
    if seen:
        seen_streak += 1
        missed_streak = 0
    else:
        missed_streak += 1
        seen_streak = 0
    if seen_streak >= CONFIRM_FRAMES:
        lane_priority = True
    elif missed_streak >= RELEASE_FRAMES:
        lane_priority = False

    # 4) Signal controller
    signals = controller.update(MY_LANE if lane_priority else None, now=now)
    if signals != last_signals:
        print(f"t={now:5.1f}s  {controller.state:16s} {signals}")
        last_signals = signals

    # 5) Draw overlay
    cv2.putText(frame, f"Vehicles: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if lane_priority:
        cv2.putText(frame, "PRIORITY LANE", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    for i, (lane, colour) in enumerate(signals.items()):
        y = 40 + i * 40
        cv2.circle(frame, (width - 200, y), 14, COLOURS[colour], -1)
        label = lane + (" (camera)" if lane == MY_LANE else "")
        cv2.putText(frame, label, (width - 175, y + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)
    frame_no += 1
    if frame_no % 20 == 0:
        print("Processed", frame_no, "frames")

cap.release()
out.release()
print("Done. Processed", frame_no, "frames. Open output.avi")