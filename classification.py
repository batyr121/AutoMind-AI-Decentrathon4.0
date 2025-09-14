import os
import shutil

# пути к YOLO датасету
DATASET_DIR = "dataset"       # здесь твой путь к скачанному датасету
OUTPUT_DIR = "dataset_cls"    # куда сохраняем классификационный датасет

# файлик с классами (создается при экспорте YOLO в Roboflow)
NAMES_FILE = os.path.join(DATASET_DIR, "data.yaml")

# читаем список классов
classes = []
with open(NAMES_FILE, "r") as f:
    for line in f:
        if line.strip().startswith("names:"):
            # парсим список классов из yaml
            classes = eval(line.split("names:")[1].strip())
            break

print("Классы:", classes)

def process_split(split):
    img_dir = os.path.join(DATASET_DIR, split, "images")
    lbl_dir = os.path.join(DATASET_DIR, split, "labels")
    
    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(lbl_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))
        
        if not os.path.exists(label_path):
            continue
        
        with open(label_path, "r") as f:
            lines = f.readlines()
            if not lines:
                continue
            # берём класс первого объекта (для классификации этого достаточно)
            class_id = int(lines[0].split()[0])
            class_name = classes[class_id]
        
        # создаём папку под класс
        out_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # копируем картинку
        shutil.copy(img_path, os.path.join(out_dir, img_file))

# прогоняем train/valid/test
for split in ["train", "valid", "test"]:
    if os.path.exists(os.path.join(DATASET_DIR, split)):
        process_split(split)

print("✅ Датасет сконвертирован в классификацию:", OUTPUT_DIR)
