from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("https://ultralytics.com/images/bus.jpg")
results[0].save("result.jpg")
print("Detected classes:", [model.names[int(c)] for c in results[0].boxes.cls])
