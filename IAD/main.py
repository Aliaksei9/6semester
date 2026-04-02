import os
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Подавляем предупреждения (в т.ч. таймаут проверки версий)
warnings.filterwarnings("ignore")
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"


# ====================== КОНСТАНТЫ ======================
# Пути и данные
DATA_ROOT = "Dataset_BUSI_with_GT"          # корневая папка датасета
IMAGE_SIZE = 128                            # размер после ресайза
NUM_CLASSES = 3                             # фон + benign + malignant
IN_CHANNELS = 3                             # RGB

# Разбиение
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42

# Обучение
BATCH_SIZE = 32  # уменьшено для стабильности на GPU
EPOCHS = 50
LEARNING_RATE = 3e-4
NUM_WORKERS = 0

# Модель U-Net
UNET_BASE_FILTERS = 64        # начальное число фильтров в encoder
UNET_DEPTH = 3                # количество уровней (глубина)
UNET_GROWTH_FACTOR = 2        # во сколько раз растёт число фильтров на каждом уровне

# Аугментации
AUG_HORIZONTAL_FLIP_PROB = 0.5
AUG_VERTICAL_FLIP_PROB = 0.5
AUG_BRIGHTNESS_CONTRAST_PROB = 0.3

# Нормализация (ImageNet)
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# Пути для сохранения
SAVE_MODEL_PATH = "best_unet.pth"
PLOT_CURVES_PATH = "training_curves.png"
VIS_PREDICTIONS_PATH = "predictions.png"
NUM_VIS_SAMPLES = 6


# ====================== DATASET ======================
class BUSIDataset(Dataset):
    """Датасет для BUSI (Breast Ultrasound Images)."""
    def __init__(self, image_paths, mask_paths, class_labels, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.class_labels = class_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Загрузка изображения
        image = plt.imread(self.image_paths[idx])
        # Приводим к float32 и диапазону [0,1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        # Отбрасываем альфа-канал
        if image.shape[-1] == 4:
            image = image[..., :3]

        # grayscale → RGB (3 канала)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Загрузка маски через PIL (для 1-bit изображений)
        mask_pil = Image.open(self.mask_paths[idx])
        mask = np.array(mask_pil)
        
        # Конвертируем bool → int64
        if mask.dtype == bool:
            mask = mask.astype(np.int64)
        elif mask.max() > 1.0:  # диапазон [0, 255]
            mask = (mask > 128).astype(np.int64)
        else:  # диапазон [0, 1]
            mask = (mask > 0.5).astype(np.int64)

        # Присваиваем класс: для benign=1, malignant=2, normal=0
        class_id = self.class_labels[idx]
        if class_id > 0:
            mask = mask * class_id
        else:
            mask = np.zeros_like(mask)

        # Гарантируем 2D маску
        if mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            elif mask.shape[-1] == 3:
                mask = mask[:, :, 0]
            else:
                mask = mask[0]

        # Применяем аугментации
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        mask = mask.long()
        return image, mask


def get_data_paths(root_dir):
    """
    Возвращает списки путей к изображениям, маскам и меткам классов.
    """
    image_paths = []
    mask_paths = []
    class_labels = []
    
    for folder, label in [('benign', 1), ('malignant', 2), ('normal', 0)]:
        folder_path = os.path.join(root_dir, folder)
        all_files = glob(os.path.join(folder_path, "*.png"))
        pairs = {}
        for f in all_files:
            basename = os.path.basename(f)
            if '_mask' in basename:
                base = basename.replace('_mask', '')
                if base not in pairs:
                    pairs[base] = {'image': None, 'mask': None}
                pairs[base]['mask'] = f
            else:
                base = basename
                if base not in pairs:
                    pairs[base] = {'image': None, 'mask': None}
                pairs[base]['image'] = f

        for base, files in pairs.items():
            if files['image'] is not None and files['mask'] is not None:
                image_paths.append(files['image'])
                mask_paths.append(files['mask'])
                class_labels.append(label)
            else:
                print(f"Warning: Skipping incomplete pair for {base} in {folder}")
    
    return image_paths, mask_paths, class_labels


def split_data(image_paths, mask_paths, class_labels):
    """Разбиение на train, val, test с использованием констант."""
    random.seed(SEED)
    indices = list(range(len(image_paths)))
    random.shuffle(indices)
    n_train = int(TRAIN_RATIO * len(indices))
    n_val = int(VAL_RATIO * len(indices))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    train_images = [image_paths[i] for i in train_idx]
    train_masks  = [mask_paths[i] for i in train_idx]
    train_labels = [class_labels[i] for i in train_idx]

    val_images   = [image_paths[i] for i in val_idx]
    val_masks    = [mask_paths[i] for i in val_idx]
    val_labels   = [class_labels[i] for i in val_idx]

    test_images  = [image_paths[i] for i in test_idx]
    test_masks   = [mask_paths[i] for i in test_idx]
    test_labels  = [class_labels[i] for i in test_idx]

    return (train_images, train_masks, train_labels), (val_images, val_masks, val_labels), (test_images, test_masks, test_labels)


# ====================== MODEL ======================
class UNet(nn.Module):
    """U-Net архитектура (с нуля)."""
    def __init__(self, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._block(256, 512)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)  # 256 + 256 = 512 -> 256
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)  # 128 + 128 = 256 -> 128
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)   # 64 + 64 = 128 -> 64

        # Выходная свёртка
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        x = self.bottleneck(self.pool(e3))

        # Decoder с skip-connections
        x = self.upconv3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)

        return self.out_conv(x)


# ====================== METRICS ======================
import torch.nn.functional as F

class FocalDiceLoss(nn.Module):
    """
    Focal + Dice Loss для медицинской сегментации.
    Focal Loss фокусируется на сложных пикселях (опухоли), 
    Dice Loss оптимизирует перекрытие областей.
    """
    def __init__(self, alpha=1.0, gamma=2, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        # 1. Focal Loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        focal_loss = focal_loss.mean()

        # 2. Dice Loss (Multiclass)
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Игнорируем фон (индекс 0), считаем Dice только для опухолей (классы 1 и 2)
        dice_loss = 1 - dice_score[1:].mean()

        return focal_loss + dice_loss


class MultiClassDiceLoss(nn.Module):
    """
    Dice Loss для нескольких классов. 
    Помогает, когда один класс (фон) намного больше других (опухоли).
    """
    def __init__(self, smooth=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [Batch, Classes, H, W]
        # targets: [Batch, H, W]
        
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        
        # Переводим маску в One-Hot: [B, H, W] -> [B, C, H, W]
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)  # Считаем по батчу, высоте и ширине
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Мы хотим минимизировать (1 - Dice)
        return 1 - dice_score.mean()


class SegmentationLoss(nn.Module):
    """Комбинация CE и Dice для стабильности."""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = MultiClassDiceLoss()

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


def iou_score(pred_mask, true_mask, num_classes):
    """Вычисляет IoU для каждого класса и возвращает средний (mIoU) только для опухолей."""
    ious = []
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()
    # Считаем IoU только для классов 1 и 2 (опухоли), игнорируем фон (класс 0)
    for c in range(1, num_classes):
        pred_c = (pred_mask == c)
        true_c = (true_mask == c)
        intersection = (pred_c & true_c).sum().float()
        union = (pred_c | true_c).sum().float()
        
        # Если класса нет ни в маске, ни в предсказании — это успех (IoU=1)
        if union == 0:
            ious.append(1.0)
        else:
            ious.append((intersection / union).item())
    return np.mean(ious) if ious else 0.0


def dice_score(pred_mask, true_mask, num_classes):
    """Вычисляет Dice Score для каждого класса и возвращает средний только для опухолей."""
    dices = []
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()
    # Считаем Dice только для классов 1 и 2 (опухоли), игнорируем фон (класс 0)
    for c in range(1, num_classes):
        pred_c = (pred_mask == c)
        true_c = (true_mask == c)
        intersection = (pred_c & true_c).sum().float()
        dice = (2. * intersection) / (pred_c.sum().float() + true_c.sum().float() + 1e-8)
        dices.append(dice.item())
    return np.mean(dices) if dices else 0.0


def evaluate(model, loader, num_classes, device):
    """Оценка mIoU и Dice на всём DataLoader."""
    model.eval()
    ious, dices = [], []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            for i in range(images.size(0)):
                ious.append(iou_score(preds[i], masks[i], num_classes))
                dices.append(dice_score(preds[i], masks[i], num_classes))
    return np.mean(ious), np.mean(dices)


# ====================== VISUALIZATION ======================
def visualize_predictions(model, loader, device, num_samples=NUM_VIS_SAMPLES, save_path=VIS_PREDICTIONS_PATH):
    """Визуализация с правильным отображением grayscale УЗИ-изображений."""
    model.eval()
    images_list, masks_list, preds_list = [], [], []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            images_list.extend(images.cpu())
            masks_list.extend(masks.cpu())
            preds_list.extend(preds.cpu())
            if len(images_list) >= num_samples:
                break

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    for i in range(num_samples):
        # Денормализация
        img_np = images_list[i].permute(1, 2, 0).numpy()
        img_np = img_np * np.array(NORM_STD) + np.array(NORM_MEAN)
        img_np = np.clip(img_np, 0, 1)

        # БЕРЁМ ОДИН КАНАЛ + cmap='gray' → теперь изображения НЕ чёрные!
        img_gray = img_np[:, :, 0]

        axes[i, 0].imshow(img_gray, cmap='gray')
        axes[i, 0].set_title(f"Image")
        axes[i, 0].axis('off')

        gt_mask = masks_list[i].numpy()
        axes[i, 1].imshow(gt_mask, cmap='tab10', vmin=0, vmax=NUM_CLASSES - 1)
        unique_gt = np.unique(gt_mask)
        axes[i, 1].set_title(f"Ground Truth (cls={unique_gt})")
        axes[i, 1].axis('off')

        pred_mask = preds_list[i].numpy()
        axes[i, 2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=NUM_CLASSES - 1)
        unique_pred = np.unique(pred_mask)
        axes[i, 2].set_title(f"Prediction (cls={unique_pred})")
        axes[i, 2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def plot_training_curves(train_losses, val_ious, save_path=PLOT_CURVES_PATH):
    """Построение графиков потерь и mIoU."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(val_ious, label='Val mIoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mIoU')
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ====================== TRAINING ======================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    
    return total_loss / len(loader.dataset)


def train(model, train_loader, val_loader, optimizer, criterion, device, epochs, num_classes, save_best_path):
    train_losses = []
    val_ious = []
    best_val_iou = 0.0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(avg_loss)

        val_iou, val_dice = evaluate(model, val_loader, num_classes, device)
        val_ious.append(val_iou)
        
        # Scheduler step
        scheduler.step(val_iou)

        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Val mIoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), save_best_path)
            print(f"  --> Saved best model (mIoU={val_iou:.4f})")

    return train_losses, val_ious


# ====================== MAIN ======================
def main():
    # Установка seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Загрузка путей и меток
    print("Loading data paths...")
    image_paths, mask_paths, class_labels = get_data_paths(DATA_ROOT)

    # 2. Разбиение
    (train_images, train_masks, train_labels), \
    (val_images, val_masks, val_labels), \
    (test_images, test_masks, test_labels) = split_data(
        image_paths, mask_paths, class_labels
    )
    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

    # 3. Аугментации (с константами)
    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=AUG_HORIZONTAL_FLIP_PROB),
        A.VerticalFlip(p=AUG_VERTICAL_FLIP_PROB),
        A.RandomBrightnessContrast(p=AUG_BRIGHTNESS_CONTRAST_PROB),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])

    # 4. Датасеты и загрузчики
    train_dataset = BUSIDataset(train_images, train_masks, train_labels, transform=train_transform)
    val_dataset   = BUSIDataset(val_images, val_masks, val_labels, transform=val_transform)
    test_dataset  = BUSIDataset(test_images, test_masks, test_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Вывод статистики по классам и маскам
    unique, counts = np.unique(train_labels, return_counts=True)
    print(f"\nDataset class distribution:")
    for u, c in zip(unique, counts):
        class_name = {0: 'normal', 1: 'benign', 2: 'malignant'}.get(u, 'unknown')
        print(f"  Class {u} ({class_name}): {c} samples")
    
    # Проверка масок
    print("\nChecking masks...")
    non_empty_masks = 0
    for i in range(len(train_dataset)):
        _, mask = train_dataset[i]
        if mask.max() > 0:
            non_empty_masks += 1
    print(f"Non-empty masks in train: {non_empty_masks}/{len(train_dataset)}")

    # 5. Модель, оптимизатор, функция потерь
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # L2 регуляризация
    criterion = FocalDiceLoss(alpha=1.0, gamma=2.0)  # Focal + Dice Loss

    # 6. Обучение
    print("\nStarting training...")
    train_losses, val_ious = train(
        model, train_loader, val_loader, optimizer, criterion, device,
        epochs=EPOCHS, num_classes=NUM_CLASSES, save_best_path=SAVE_MODEL_PATH
    )

    # 7. Графики
    plot_training_curves(train_losses, val_ious)

    # 8. Финальное тестирование
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(SAVE_MODEL_PATH, weights_only=True))
    test_iou, test_dice = evaluate(model, test_loader, NUM_CLASSES, device)
    print(f"Test mIoU: {test_iou:.4f}, Test Dice: {test_dice:.4f}")

    # 9. Визуализация предсказаний
    print("\nVisualizing predictions...")
    visualize_predictions(model, val_loader, device)


if __name__ == "__main__":
    main()
