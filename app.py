import os
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # разрешаем CORS-запросы с фронтенда

# =====================
# Загрузка моделей
# =====================
# YOLO (детекция частей машины)
yolo_model = YOLO("models/best.pt")  # твоя модель на 17 классов

# Классификатор чистая/грязная (ResNet18)
dirty_model = models.resnet18(pretrained=False)
dirty_model.fc = nn.Linear(dirty_model.fc.in_features, 2)  # 2 класса
dirty_model.load_state_dict(torch.load("models/car_clean_dirty.pth", map_location="cpu"))
dirty_model.eval()

# Классификатор повреждений (ResNet18)
damage_model = models.resnet18(pretrained=False)
damage_model.fc = nn.Linear(damage_model.fc.in_features, 4)  # 4 класса
damage_model.load_state_dict(torch.load("models/car_damage_classifier.pth", map_location="cpu"))
damage_model.eval()

# =====================
# Трансформации
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================
# Классы
# =====================
dirty_classes = ["clean", "dirty"]
damage_classes = ["none", "dent", "rust", "scratch"]

# =====================
# Вспомогательная функция
# =====================
def classify(image: Image.Image, model: nn.Module, classes: list):
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, 0)
    return {
        "label": classes[pred.item()],
        "confidence": float(conf.item())
    }

# =====================
# API
# =====================
@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")

    results = yolo_model(img)  # YOLO предсказание
    parts_info = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, cls_ids, confs):
            x1, y1, x2, y2 = map(int, box)
            cropped = img.crop((x1, y1, x2, y2))

            dirty_pred = classify(cropped, dirty_model, dirty_classes)
            damage_pred = classify(cropped, damage_model, damage_classes)

            parts_info.append({
                "part": yolo_model.names[int(cls_id)],
                "bbox": [x1, y1, x2, y2],
                "yolo_confidence": float(conf),
                "dirty": dirty_pred,
                "damage": damage_pred
            })

    return jsonify({"parts": parts_info})

# =====================
# Запуск
# =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5500, debug=True)
