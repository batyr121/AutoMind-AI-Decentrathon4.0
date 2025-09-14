from ultralytics import YOLO

# Загружаем предобученную модель
model = YOLO("yolov8n.pt")  # можно yolov8s.pt если мощнее ноут

# Обучение
model.train(
    data="dataset-part-cars/dataofpart.yaml",   # твой yaml с классами (roboflow сгенерил)
    epochs=2,             # можешь увеличить
    imgsz=640,             # размер входа
    batch=16               # подгони под VRAM
)
