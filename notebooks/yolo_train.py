from ultralytics import YOLO

# Load a model
model = YOLO("yolo11l.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="./data.yaml", epochs=250, imgsz=640)