from ultralytics import YOLOWorld

# Automatically downloads the "small" YOLO-World model
model = YOLOWorld('yolov8s-world.pt') 

# Define your custom text prompts
model.set_classes(["tent"])

# Run inference on your test photo
results = model.predict('test_media/test3.jpg', save=True, conf=0.1)

# The annotated image will be saved in a 'runs/detect/predict' folder