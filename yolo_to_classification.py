import os
import shutil

# Папки YOLO-датасета
IMAGES_DIR = "dirty-car-dataset/train/images"   # тут все фото
LABELS_DIR = "dirty-car-dataset/train/labels"   # тут txt аннотации
OUTPUT_DIR = "dataset_classification"

# Сопоставление id → имя класса
# Если в labels/*.txt написано "0 ..." → это clean, "1 ..." → dirty
CLASS_MAP = {
    0: "clean",
    1: "dirty"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Читаем все аннотации
for label_file in os.listdir(LABELS_DIR):
    if not label_file.endswith(".txt"):
        continue

    with open(os.path.join(LABELS_DIR, label_file), "r") as f:
        lines = f.readlines()

    if not lines:
        continue

    # Берём первый класс из файла (если YOLO → каждая строка: class x y w h)
    class_id = int(lines[0].split()[0])

    if class_id not in CLASS_MAP:
        continue

    cls_name = CLASS_MAP[class_id]

    # Создаём папку
    class_folder = os.path.join(OUTPUT_DIR, cls_name)
    os.makedirs(class_folder, exist_ok=True)

    # Копируем картинку
    image_file = label_file.replace(".txt", ".jpg")
    src_path = os.path.join(IMAGES_DIR, image_file)
    dst_path = os.path.join(class_folder, image_file)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"✔ {image_file} → {cls_name}")
    else:
        print(f"⚠ Нет картинки для {label_file}")

print("✅ Все фото разложены по папкам clean/dirty")
