import torch
import time
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
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DATA_ROOT = './data/data'
IMG_SIZE = 128
BATCH_SIZE = 64
NUM_EPOCHS = 2
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
SAVE_EXAMPLES = True
EXAMPLES_DIR = "./transformed_examples"
NUM_EXAMPLES_TO_SAVE = 3
RANDOM_SEED = 42

WEATHERCNN_CONV1_OUT = 32
WEATHERCNN_CONV2_OUT = 64
WEATHERCNN_CONV3_OUT = 128
WEATHERCNN_CONV4_OUT = 256
WEATHERCNN_CONV5_OUT = 512
WEATHERCNN_KERNEL_SIZE = 3
WEATHERCNN_PADDING = 1
WEATHERCNN_FC_HIDDEN = 256
WEATHERCNN_DROPOUT = 0.4

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
LR_FINETUNE_BACKBONE = 1e-4
LR_FINETUNE_HEAD = 1e-3
LR_FROZEN = 0.001
NUM_WORKERS = 4
PIN_MEMORY = True
VIZ_ALPHA = 0.5
VIZ_DPI = 300
VIZ_FIGSIZE_MATRIX = (12, 12)
VIZ_FILTERS_GRID_SIZE = 8
VIZ_ACTIVATIONS_CHANNELS = 8
VIZ_GRADCAM_CORRECT_COUNT = 4
VIZ_GRADCAM_WRONG_COUNT = 4
VIZ_ERROR_EXAMPLES_PER_PAIR = 3
IMG_SIZE_TL = 224

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
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform: img = self.transform(img)
        return img, label

def get_transforms(img_size=IMG_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
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
    return random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(RANDOM_SEED))

def create_dataloaders(train_subset, val_subset, test_subset, train_transform, val_transform):
    train_dataset = TransformedSubset(train_subset, train_transform)
    val_dataset   = TransformedSubset(val_subset,   val_transform)
    test_dataset  = TransformedSubset(test_subset,  val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    return train_loader, val_loader, test_loader

class WeatherCNN(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, WEATHERCNN_CONV1_OUT, kernel_size=WEATHERCNN_KERNEL_SIZE, padding=WEATHERCNN_PADDING),
            nn.BatchNorm2d(WEATHERCNN_CONV1_OUT), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(WEATHERCNN_CONV1_OUT, WEATHERCNN_CONV2_OUT, kernel_size=WEATHERCNN_KERNEL_SIZE, padding=WEATHERCNN_PADDING),
            nn.BatchNorm2d(WEATHERCNN_CONV2_OUT), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(WEATHERCNN_CONV2_OUT, WEATHERCNN_CONV3_OUT, kernel_size=WEATHERCNN_KERNEL_SIZE, padding=WEATHERCNN_PADDING),
            nn.BatchNorm2d(WEATHERCNN_CONV3_OUT), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(WEATHERCNN_CONV3_OUT, WEATHERCNN_CONV4_OUT, kernel_size=WEATHERCNN_KERNEL_SIZE, padding=WEATHERCNN_PADDING),
            nn.BatchNorm2d(WEATHERCNN_CONV4_OUT), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(WEATHERCNN_CONV4_OUT, WEATHERCNN_CONV5_OUT, kernel_size=WEATHERCNN_KERNEL_SIZE, padding=WEATHERCNN_PADDING),
            nn.BatchNorm2d(WEATHERCNN_CONV5_OUT), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(WEATHERCNN_CONV5_OUT, WEATHERCNN_FC_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(p=WEATHERCNN_DROPOUT),
            nn.Linear(WEATHERCNN_FC_HIDDEN, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None):
    train_losses = []
    val_accuracies = []
    epoch_times = []
    best_val_acc = 0.0
    convergence_epoch = 0

    print("Начало обучения...")
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_times.append(epoch_duration)

        train_losses.append(train_loss)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            convergence_epoch = epoch + 1

        print(f"Epoch [{epoch + 1}/{num_epochs}] - {epoch_duration:.2f}s | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if scheduler is not None:
            scheduler.step(train_loss)

    avg_time = sum(epoch_times) / len(epoch_times)
    print(f"\nСреднее время на эпоху: {avg_time:.2f} сек")
    print(f"Сходимость (лучшая эпоха): {convergence_epoch} (Acc: {best_val_acc:.2f}%)")
    return train_losses, val_accuracies, avg_time, convergence_epoch, best_val_acc

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

def prepare_test_examples(full_dataset, test_subset):
    test_transform = get_transforms(IMG_SIZE_TL)[1]
    test_pil_list = []
    for i in range(len(test_subset)):
        orig_idx = test_subset.indices[i]
        pil_img, label = full_dataset[orig_idx]
        tensor = test_transform(pil_img)
        test_pil_list.append((pil_img, label, tensor))
    return test_pil_list

def compute_predictions(model, test_pil_list, device):
    all_preds = []
    all_labels = []
    correct_examples = []
    incorrect_examples = []
    model.eval()
    with torch.no_grad():
        for pil, true_label, tensor in test_pil_list:
            tensor = tensor.unsqueeze(0).to(device)
            output = model(tensor)
            pred = output.argmax(dim=1).item()
            all_preds.append(pred)
            all_labels.append(true_label)
            if pred == true_label:
                correct_examples.append((pil, true_label, pred, tensor.squeeze(0)))
            else:
                incorrect_examples.append((pil, true_label, pred, tensor.squeeze(0)))
    return all_preds, all_labels, correct_examples, incorrect_examples

class GradCAM:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, out):
        self.activations = out.detach()

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, img_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(img_tensor.unsqueeze(0).to(self.device))
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        output[0, target_class].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear')[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy()

def plot_gradcam(example, title_prefix, filename, grad_cam, full_dataset, save_dir):
    pil, true_label, pred_label, tensor = example
    cam = grad_cam(tensor)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(pil)
    plt.title(f"Original\nTrue: {full_dataset.classes[true_label]}\nPred: {full_dataset.classes[pred_label]}")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(pil)
    plt.imshow(cam, cmap='jet', alpha=VIZ_ALPHA)
    plt.title("Grad-CAM")
    plt.axis('off')
    plt.suptitle(title_prefix)
    plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=VIZ_DPI, bbox_inches='tight')
    plt.show()

def task2_part_a(model, save_dir="task2_results"):
    os.makedirs(save_dir, exist_ok=True)
    first_conv = model.conv1
    filters = first_conv.weight.detach().cpu()
    filters_norm = (filters - filters.min()) / (filters.max() - filters.min() + 1e-8)

    fig, axes = plt.subplots(VIZ_FILTERS_GRID_SIZE, VIZ_FILTERS_GRID_SIZE, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < filters.shape[0]:
            img = filters_norm[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')
    plt.suptitle("First Layer Filters (conv1) — Model C (fine-tuning)")
    plt.savefig(os.path.join(save_dir, "A_filters.png"), dpi=VIZ_DPI, bbox_inches='tight')
    plt.show()
    print("PART A completed")

def task2_part_b(model, test_pil_list, full_dataset, save_dir="task2_results"):
    os.makedirs(save_dir, exist_ok=True)
    activations = {}

    def hook_fn(name):
        def hook(module, inp, out):
            activations[name] = out.detach().cpu()
        return hook

    model.layer1.register_forward_hook(hook_fn('layer1'))
    model.layer4.register_forward_hook(hook_fn('layer4'))

    samples = random.sample(test_pil_list, min(3, len(test_pil_list)))
    for idx, (pil, true, tensor) in enumerate(samples):
        with torch.no_grad():
            _ = model(tensor.unsqueeze(0).to(DEVICE))

        fig, axes = plt.subplots(2, VIZ_ACTIVATIONS_CHANNELS, figsize=(16, 6))
        fig.suptitle(f"Example {idx + 1} — True: {full_dataset.classes[true]}")

        acts1 = activations['layer1'][0][:VIZ_ACTIVATIONS_CHANNELS]
        for j, ax in enumerate(axes[0]):
            ax.imshow(acts1[j].numpy(), cmap='viridis')
            ax.axis('off')
        axes[0, 0].set_title('Layer1 (low-level features)')

        acts4 = activations['layer4'][0][:VIZ_ACTIVATIONS_CHANNELS]
        for j, ax in enumerate(axes[1]):
            ax.imshow(acts4[j].numpy(), cmap='viridis')
            ax.axis('off')
        axes[1, 0].set_title('Layer4 (high-level features)')

        plt.savefig(os.path.join(save_dir, f"B_activations_{idx + 1}.png"), dpi=200)
        plt.show()
    print("PART B completed")

def task2_part_c(model, correct_examples, incorrect_examples, full_dataset, save_dir="task2_results"):
    os.makedirs(save_dir, exist_ok=True)
    grad_cam = GradCAM(model, model.layer4, DEVICE)

    for i, ex in enumerate(random.sample(correct_examples, min(VIZ_GRADCAM_CORRECT_COUNT, len(correct_examples)))):
        plot_gradcam(ex, f"CORRECT classification {i + 1}", f"C_correct_{i + 1}", grad_cam, full_dataset, save_dir)
    for i, ex in enumerate(random.sample(incorrect_examples, min(VIZ_GRADCAM_WRONG_COUNT, len(incorrect_examples)))):
        plot_gradcam(ex, f"ERROR {i + 1}", f"C_wrong_{i + 1}", grad_cam, full_dataset, save_dir)
    print("PART C completed")

def task2_part_d(all_labels, all_preds, full_dataset, incorrect_examples, model, save_dir="task2_results"):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=VIZ_FIGSIZE_MATRIX)
    disp = ConfusionMatrixDisplay(cm, display_labels=full_dataset.classes)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix — Model C (fine-tuning)")
    plt.savefig(os.path.join(save_dir, "D_confusion_matrix.png"), dpi=VIZ_DPI, bbox_inches='tight')
    plt.show()

    errors = [(cm[i, j], i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
    errors.sort(reverse=True)
    top3 = errors[:3]

    print("\nTOP-3 most frequent errors:")
    for count, true_i, pred_j in top3:
        print(f"   {full_dataset.classes[true_i]} → {full_dataset.classes[pred_j]}  ({count} times)")

    grad_cam = GradCAM(model, model.layer4, DEVICE)
    for rank, (count, true_i, pred_j) in enumerate(top3, 1):
        print(f"\nTOP-{rank} error: {full_dataset.classes[true_i]} → {full_dataset.classes[pred_j]}")
        mistakes = [ex for ex in incorrect_examples if ex[1] == true_i and ex[2] == pred_j]
        for i, ex in enumerate(mistakes[:VIZ_ERROR_EXAMPLES_PER_PAIR]):
            plot_gradcam(ex, f"Error example TOP-{rank} ({i + 1})", f"D_error_{rank}_{i + 1}", grad_cam, full_dataset, save_dir)
    print("PART D completed")

def run_task2(model, full_dataset, test_subset, device):
    global DEVICE
    DEVICE = device

    print("\n" + "=" * 70)
    print("RUNNING TASK 2 (Parts A–D)")

    test_pil_list = prepare_test_examples(full_dataset, test_subset)
    all_preds, all_labels, correct_examples, incorrect_examples = compute_predictions(model, test_pil_list, device)

    save_dir = "task2_results"
    os.makedirs(save_dir, exist_ok=True)

    task2_part_a(model, save_dir)
    task2_part_b(model, test_pil_list, full_dataset, save_dir)
    task2_part_c(model, correct_examples, incorrect_examples, full_dataset, save_dir)
    task2_part_d(all_labels, all_preds, full_dataset, incorrect_examples, model, save_dir)

    print("\nTASK 2 COMPLETED SUCCESSFULLY!")
    print(f"All files saved in folder **{save_dir}/**")

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    print("\n" + "="*60)
    print("Выберите модель для обучения:")
    print("1 — Модель A (с нуля — WeatherCNN 128×128, как в твоём оригинальном коде)")
    print("2 — Модель B (ResNet18 с замороженными слоями)")
    print("3 — Модель C (ResNet18 полный fine-tuning)")
    choice = input("Ваш выбор (1/2/3): ").strip()

    full_dataset = load_full_dataset(DATA_ROOT)
    num_classes = len(full_dataset.classes)
    print(f"Найдено классов: {num_classes} → {full_dataset.classes}")

    train_subset, val_subset, test_subset = split_datasets(full_dataset)

    if choice == "1":
        print("\n=== ЗАПУСК МОДЕЛИ A (оригинальный код) ===")
        train_transform, val_transform = get_transforms(IMG_SIZE)
        if SAVE_EXAMPLES:
            save_transformed_examples(full_dataset, train_transform, val_transform)

        train_loader, val_loader, test_loader = create_dataloaders(train_subset, val_subset, test_subset, train_transform, val_transform)

        model = WeatherCNN(num_classes).to(device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Обучаемых:{trainable_params:,} / Всего:{sum(p.numel() for p in model.parameters()):,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

        train_losses, val_accuracies, avg_time, conv_epoch, best_val = train_model(
            model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device, scheduler
        )

        test_acc = evaluate(model, test_loader, device)
        plot_metrics(train_losses, val_accuracies)

        torch.save(model.state_dict(), 'weather_cnn_final.pth')
        print(f"Test Accuracy: {test_acc:.2f}%")

    elif choice in ["2", "3"]:
        import torchvision.models as models
        print(f"\n=== ЗАПУСК МОДЕЛИ {'B' if choice=='2' else 'C'} ===")
        train_transform, val_transform = get_transforms(IMG_SIZE_TL)
        train_loader, val_loader, test_loader = create_dataloaders(train_subset, val_subset, test_subset, train_transform, val_transform)

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
        model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)

        if choice == "2":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_FROZEN,
                weight_decay=WEIGHT_DECAY
            )
            model_name = "ResNet18 (замороженные)"
        else:
            backbone_params = [p for name, p in model.named_parameters() if 'fc' not in name]
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': LR_FINETUNE_BACKBONE},
                {'params': model.fc.parameters(), 'lr': LR_FINETUNE_HEAD}
            ], weight_decay=WEIGHT_DECAY)
            model_name = "ResNet18 (fine-tuning)"

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Обучаемых параметров: {trainable_params:,}")

        criterion = nn.CrossEntropyLoss()
        train_losses, val_accuracies, avg_time, conv_epoch, best_val = train_model(
            model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device, scheduler=None
        )
        test_acc = evaluate(model, test_loader, device)
        plot_metrics(train_losses, val_accuracies)

        run_task2(model, full_dataset, test_subset, device)

        torch.save(model.state_dict(), f'resnet18_{"frozen" if choice=="2" else "finetune"}.pth')
        print(f"Test Accuracy: {test_acc:.2f}%")

    else:
        print("Неверный выбор!")
        exit()

    print("\n" + "="*70)
    print(f"Модель: {'A (с нуля)' if choice=='1' else 'B (замороженные)' if choice=='2' else 'C (fine-tuning)'}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    print(f"Время на эпоху: {avg_time:.2f} сек")
    print(f"Эпох до сходимости: {conv_epoch}")
    print(f"Val accuracy: {best_val:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print("="*70)