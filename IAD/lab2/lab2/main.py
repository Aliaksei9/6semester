import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import random
import numpy as np
import os

# ====================== КОНСТАНТЫ ======================
DATA_ROOT = './data'

IMG_SIZE = 128
BATCH_SIZE = 64
NUM_EPOCHS = 15

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

LEARNING_RATE = 0.01
DROPOUT_P = 0.5


IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]

SAVE_EXAMPLES = True
EXAMPLES_DIR = "./transformed_examples"
NUM_EXAMPLES_TO_SAVE = 3

WEIGHT_DECAY=1e-4

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def save_transformed_examples(dataset, train_transform, val_transform, num=NUM_EXAMPLES_TO_SAVE, save_dir=EXAMPLES_DIR):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Сохраняем {num} примеров трансформаций в папку: {save_dir}")

    for i in range(num):
        img_pil, label = dataset[i]
        class_name = dataset.classes[label]

        img_pil.save(f"{save_dir}/original_{i}_{class_name}.jpg")

        img_val = val_transform(img_pil)
        img_val_denorm = denormalize(img_val, IMAGE_MEAN, IMAGE_STD).clamp_(0, 1)
        save_image(img_val_denorm, f"{save_dir}/val_transformed_{i}_{class_name}.jpg")

        img_train = train_transform(img_pil)
        img_train_denorm = denormalize(img_train, IMAGE_MEAN, IMAGE_STD).clamp_(0, 1)
        save_image(img_train_denorm, f"{save_dir}/train_transformed_{i}_{class_name}.jpg")

    print("Примеры успешно сохранены!\n")


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])
    return train_transform, val_transform


def load_full_dataset(data_root):
    return datasets.ImageFolder(root=data_root, transform=None)


def split_datasets(full_dataset):
    total = len(full_dataset)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)
    test_size = total - train_size - val_size
    return random_split(full_dataset, [train_size, val_size, test_size])


def create_dataloaders(train_subset, val_subset, test_subset, train_transform, val_transform):
    train_dataset = TransformedSubset(train_subset, train_transform)
    val_dataset   = TransformedSubset(val_subset,   val_transform)
    test_dataset  = TransformedSubset(test_subset,  val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


class WeatherCNN(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()

        self.features = nn.Sequential(
            # Блок 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Блок 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Блок 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # НОВЫЙ Блок 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # НОВЫЙ Блок 5: 14x14 -> 7x7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Global Average Pooling превращает (512, 7, 7) в (512, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def build_model(num_classes):
    return WeatherCNN(num_classes)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(model, loader, device):
    """Универсальная функция для оценки (валидация или тест)"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_accuracies = []

    print("Начало обучения...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)   # ← используем новую функцию

        train_losses.append(train_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.4f}  Val Accuracy: {val_acc:.2f}%")
    return train_losses, val_accuracies


def plot_metrics(train_losses, val_accuracies):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-o', label='Train Loss')
    plt.xlabel('Эпоха'); plt.ylabel('Loss'); plt.title('Train Loss'); plt.grid(True); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, 'r-o', label='Val Accuracy')
    plt.xlabel('Эпоха'); plt.ylabel('Accuracy (%)'); plt.title('Val Accuracy'); plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    full_dataset = load_full_dataset(DATA_ROOT)
    print(f"Найдено классов: {len(full_dataset.classes)} → {full_dataset.classes}")

    train_transform, val_transform = get_transforms()

    if SAVE_EXAMPLES:
        save_transformed_examples(full_dataset, train_transform, val_transform)

    train_subset, val_subset, test_subset = split_datasets(full_dataset)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_subset, val_subset, test_subset, train_transform, val_transform
    )

    print(f"Размеры → train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}")

    model = build_model(len(full_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Следим за минимизацией loss
        factor=0.5,  # Уменьшаем вдвое
        patience=3,  # Ждем 3 эпохи без улучшений
    )

    train_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device)

    final_val_acc = val_accuracies[-1]
    test_acc = evaluate(model, test_loader, device)

    print("\n" + "="*60)
    print(f"ФИНАЛЬНАЯ ТОЧНОСТЬ НА ВАЛИДАЦИОННОЙ ВЫБОРКЕ: {final_val_acc:.2f}%")
    print(f"ТОЧНОСТЬ НА ТЕСТОВОМ НАБОРЕ:               {test_acc:.2f}%")
    print("="*60)

    plot_metrics(train_losses, val_accuracies)

    torch.save(model.state_dict(), 'weather_cnn_final.pth')
    print("Модель сохранена в weather_cnn_final.pth")