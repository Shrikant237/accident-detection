from ultralytics import YOLO

model = YOLO("models/yolo11n.pt")

results = model.predict(
    source="testing2.mp4",
    save=True,
    conf=0.25
)

print("Detection completed!")
print("Results saved in the runs/detect folder.")