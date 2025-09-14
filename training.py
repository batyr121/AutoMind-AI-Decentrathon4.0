import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# ======== Трансформации =========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ======== Датасеты =========
train_dataset = datasets.ImageFolder(root="dataset_cls/train", transform=transform)
val_dataset = datasets.ImageFolder(root="dataset_cls/valid", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class_names = train_dataset.classes
print("Классы:", class_names)
print("Всего классов:", len(class_names))

# ======== Модель =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)  # без скачивания весов
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ======== Функция Accuracy =========
def calculate_accuracy(loader):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return 100 * correct / total

# ======== Обучение =========
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_acc = calculate_accuracy(train_loader)
    val_acc = calculate_accuracy(val_loader)
    print(f"Эпоха {epoch+1}/{num_epochs}, loss={running_loss/len(train_loader):.4f}, "
          f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

# ======== Сохранение модели =========
torch.save(model.state_dict(), "car_damage_classifier.pth")
print("✅ Модель сохранена как car_damage_classifier.pth")
