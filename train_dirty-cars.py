import torch
import torch.nn as nn
import torch.optim as optim
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# === Параметры ===
DATA_DIR = "dataset_classification-dirty"
BATCH_SIZE = 16
EPOCHS = 5
LR = 0.001

# === Аугментации и нормализация ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Загружаем датасет ===
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Разделяем train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Модель ===
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 класса: clean / dirty

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Оптимизатор и loss ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === Тренировка ===
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")

    # === Валидация ===
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Validation Acc: {val_acc:.4f}")

# === Сохраняем модель ===
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/car_clean_dirty.pth")
print("✅ Модель сохранена: models/car_clean_dirty.pth")
