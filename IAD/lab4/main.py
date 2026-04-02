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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision.models as models

warnings.filterwarnings("ignore")
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"


# ====================== КОНСТАНТЫ ======================
DATA_ROOT = "Dataset_BUSI_with_GT"
IMAGE_SIZE = 128
NUM_CLASSES = 3
IN_CHANNELS = 3

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 3e-4
NUM_WORKERS = 0

AUG_HORIZONTAL_FLIP_PROB = 0.5
AUG_VERTICAL_FLIP_PROB = 0.5
AUG_BRIGHTNESS_CONTRAST_PROB = 0.3

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

SAVE_MODEL_PATH = "best_unet.pth"
PLOT_CURVES_PATH = "training_curves.png"
VIS_PREDICTIONS_PATH = "predictions.png"
WORST_ERRORS_PATH = "worst_errors.png"
NUM_VIS_SAMPLES = 6
NUM_WORST = 6


# ====================== BUSIDataset (возвращает numpy — как раньше) ======================
class BUSIDataset(Dataset):
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
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Загрузка маски
        mask_pil = Image.open(self.mask_paths[idx])
        mask = np.array(mask_pil)

        if mask.dtype == bool:
            mask = mask.astype(np.int64)
        elif mask.max() > 1.0:
            mask = (mask > 128).astype(np.int64)
        else:
            mask = (mask > 0.5).astype(np.int64)

        class_id = self.class_labels[idx]
        if class_id > 0:
            mask = mask * class_id
        else:
            mask = np.zeros_like(mask)

        if mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            elif mask.shape[-1] == 3:
                mask = mask[:, :, 0]
            else:
                mask = mask[0]

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return image, mask

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, mask = self.subset[idx]

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            mask = torch.from_numpy(mask).long()

        return image, mask


def get_data_paths(root_dir):
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
            else:
                base = basename

            if base not in pairs:
                pairs[base] = {'image': None, 'mask': None}

            if '_mask' in basename:
                pairs[base]['mask'] = f
            else:
                pairs[base]['image'] = f

        for base, files in pairs.items():
            if files.get('image') and files.get('mask'):
                image_paths.append(files['image'])
                mask_paths.append(files['mask'])
                class_labels.append(label)

    return image_paths, mask_paths, class_labels


class UNet(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES):
        super().__init__()
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._block(256, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)

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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        x = self.bottleneck(self.pool(e3))

        x = self.upconv3(x);
        x = torch.cat([x, e3], dim=1);
        x = self.dec3(x)
        x = self.upconv2(x);
        x = torch.cat([x, e2], dim=1);
        x = self.dec2(x)
        x = self.upconv1(x);
        x = torch.cat([x, e1], dim=1);
        x = self.dec1(x)
        return self.out_conv(x)


class PretrainedUNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.bottleneck = resnet.layer4

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)

        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        d = self.up4(b)
        d = torch.cat([d, e4], dim=1)
        d = self.dec4(d)
        d = self.up3(d)
        d = torch.cat([d, e3], dim=1)
        d = self.dec3(d)
        d = self.up2(d)
        d = torch.cat([d, e2], dim=1)
        d = self.dec2(d)
        d = self.up1(d)
        d = torch.cat([d, e1], dim=1)
        d = self.dec1(d)
        d = self.final_up(d)          # ← добавлено
        return self.out_conv(d)

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        # Приводим метки к типу long
        targets = targets.long()

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        focal_loss = focal_loss.mean()

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice_score[1:].mean()

        return focal_loss + dice_loss


def iou_score(pred_mask, true_mask, num_classes):
    ious = []
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()
    for c in range(1, num_classes):
        pred_c = (pred_mask == c)
        true_c = (true_mask == c)
        intersection = (pred_c & true_c).sum().float()
        union = (pred_c | true_c).sum().float()
        ious.append((intersection / union).item() if union > 0 else 1.0)
    return np.mean(ious) if ious else 0.0


def per_class_iou(model, loader, num_classes, device):
    model.eval()
    class_ious = [[] for _ in range(num_classes)]
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Per-class IoU"):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            for i in range(images.size(0)):
                pred = preds[i].flatten()
                true = masks[i].flatten()
                for c in range(num_classes):
                    pred_c = (pred == c)
                    true_c = (true == c)
                    inter = (pred_c & true_c).sum().float()
                    union = (pred_c | true_c).sum().float()
                    iou = (inter / union).item() if union > 0 else 1.0
                    class_ious[c].append(iou)
    return [np.mean(ious) for ious in class_ious]


def dice_score(pred_mask, true_mask, num_classes):
    dices = []
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()
    for c in range(1, num_classes):
        pred_c = (pred_mask == c)
        true_c = (true_mask == c)
        intersection = (pred_c & true_c).sum().float()
        dice = (2. * intersection) / (pred_c.sum().float() + true_c.sum().float() + 1e-8)
        dices.append(dice.item())
    return np.mean(dices) if dices else 0.0


def evaluate(model, loader, num_classes, device):
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


def visualize_predictions(model, loader, device, num_samples=NUM_VIS_SAMPLES, save_path=VIS_PREDICTIONS_PATH):
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
        img_np = images_list[i].permute(1, 2, 0).numpy()
        img_np = img_np * np.array(NORM_STD) + np.array(NORM_MEAN)
        img_np = np.clip(img_np, 0, 1)[:, :, 0]

        axes[i, 0].imshow(img_np, cmap='gray')
        axes[i, 0].set_title("Image")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(masks_list[i].numpy(), cmap='tab10', vmin=0, vmax=NUM_CLASSES - 1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(preds_list[i].numpy(), cmap='tab10', vmin=0, vmax=NUM_CLASSES - 1)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def visualize_worst_examples(model, loader, device, num_worst=NUM_WORST, save_path=WORST_ERRORS_PATH):
    print(f"\nПоиск {num_worst} самых плохих изображений (самый низкий IoU)...")
    model.eval()
    samples = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            for i in range(images.size(0)):
                img = images[i].cpu()
                true_m = masks[i].cpu()
                pred_m = preds[i].cpu()
                iou = iou_score(pred_m, true_m, NUM_CLASSES)
                samples.append((img, true_m, pred_m, iou))
            if len(samples) >= 100:
                break

    samples.sort(key=lambda x: x[3])
    worst = samples[:num_worst]

    fig, axes = plt.subplots(len(worst), 4, figsize=(20, 4 * len(worst)))
    for row, (img_t, true_t, pred_t, iou) in enumerate(worst):
        img_np = img_t.permute(1, 2, 0).numpy()
        img_np = img_np * np.array(NORM_STD) + np.array(NORM_MEAN)
        img_np = np.clip(img_np, 0, 1)[:, :, 0]

        axes[row, 0].imshow(img_np, cmap='gray')
        axes[row, 0].set_title(f"Image\nIoU = {iou:.4f}")
        axes[row, 0].axis('off')

        axes[row, 1].imshow(true_t.numpy(), cmap='tab10', vmin=0, vmax=NUM_CLASSES - 1)
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis('off')

        axes[row, 2].imshow(pred_t.numpy(), cmap='tab10', vmin=0, vmax=NUM_CLASSES - 1)
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis('off')

        error_map = (pred_t != true_t).float().numpy()
        axes[row, 3].imshow(error_map, cmap='Reds')
        axes[row, 3].set_title("Error Map\n(red = mistake)")
        axes[row, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Самые плохие изображения + карты ошибок сохранены в {save_path}")


def plot_training_curves(train_losses, val_ious, save_path=PLOT_CURVES_PATH):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.set_xlabel('Epoch');
    ax1.set_ylabel('Loss');
    ax1.legend()
    ax2.plot(val_ious, label='Val mIoU')
    ax2.set_xlabel('Epoch');
    ax2.set_ylabel('mIoU');
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def train(model, train_loader, val_loader, optimizer, criterion, device, epochs, num_classes, save_best_path):
    train_losses = []
    val_ious = []
    best_val_iou = 0.0
    convergence_epoch = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Training epoch {epoch}"):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)

        val_iou, val_dice = evaluate(model, val_loader, num_classes, device)
        val_ious.append(val_iou)

        scheduler.step(val_iou)

        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Val mIoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            convergence_epoch = epoch
            torch.save(model.state_dict(), save_best_path)
            print(f"  --> Saved best model (mIoU = {val_iou:.4f})")

    return train_losses, val_ious, best_val_iou, convergence_epoch


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    print("\n" + "=" * 60)
    print("Выберите модель:")
    print("1 — U-Net с нуля")
    print("2 — U-Net + pretrained ResNet18 encoder")
    choice = input("Ваш выбор (1/2): ").strip()
    
    # === Загрузка всех данных ===
    image_paths, mask_paths, class_labels = get_data_paths(DATA_ROOT)

    # === Создаём полный датасет (transform=None) ===
    full_dataset = BUSIDataset(image_paths, mask_paths, class_labels, transform=None)

    # === Разбиение===
    total = len(full_dataset)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)
    test_size = total - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    print(f"Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_subset)}")

    # === Трансформации ===
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

    # === Оборачиваем в TransformedSubset ===
    train_dataset = TransformedSubset(train_subset, train_transform)
    val_dataset = TransformedSubset(val_subset, val_transform)
    test_dataset = TransformedSubset(test_subset, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    if choice == "1":
        model = UNet().to(device)
        model_name = "U-Net с нуля"
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    elif choice == "2":
        model = PretrainedUNet().to(device)
        model_name = "U-Net + ResNet18 (pretrained encoder)"
        backbone_params = [p for n, p in model.named_parameters() if
                           any(x in n for x in ['enc', 'bottleneck', 'layer'])]
        decoder_params = [p for n, p in model.named_parameters() if
                          not any(x in n for x in ['enc', 'bottleneck', 'layer'])]
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': 1e-4},
            {'params': decoder_params, 'lr': 3e-4}
        ], weight_decay=1e-5)
    else:
        print("Неверный выбор!")
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== {model_name} ===")
    print(f"Всего параметров:      {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")

    criterion = FocalDiceLoss()

    print("\nЗапуск обучения...")
    train_losses, val_ious, best_val_iou, convergence_epoch = train(
        model, train_loader, val_loader, optimizer, criterion, device,
        EPOCHS, NUM_CLASSES, SAVE_MODEL_PATH)

    plot_training_curves(train_losses, val_ious)

    model.load_state_dict(torch.load(SAVE_MODEL_PATH, weights_only=True))

    test_iou, test_dice = evaluate(model, test_loader, NUM_CLASSES, device)
    test_per_class_iou = per_class_iou(model, test_loader, NUM_CLASSES, device)

    print("\n" + "=" * 80)
    print(f"РЕЗУЛЬТАТЫ ДЛЯ {model_name.upper()}")
    print(f"Эпох до сходимости:      {convergence_epoch}")
    print(f"Лучший Val mIoU:         {best_val_iou:.4f}")
    print(f"Test mIoU (опухоли):     {test_iou:.4f}")
    print(f"Test Dice (опухоли):     {test_dice:.4f}")
    print(f"Всего параметров:        {total_params:,}")
    print(f"Обучаемых параметров:    {trainable_params:,}")
    print("\nIoU по КАЖДОМУ классу (Test set):")
    class_names = ['Background (0)', 'Benign (1)', 'Malignant (2)']
    for c, iou_val in enumerate(test_per_class_iou):
        print(f"   {class_names[c]:20} → {iou_val:.4f}")
    print("=" * 80)

    visualize_predictions(model, val_loader, device)

    visualize_worst_examples(model, test_loader, device, num_worst=NUM_WORST, save_path=WORST_ERRORS_PATH)

if __name__ == "__main__":
    main()