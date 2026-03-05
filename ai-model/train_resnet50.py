"""
Training script for SmartMineResNet50.

Dataset structure expected:
    dataset/
        train/
            safe/
            unsafe/
        val/
            safe/
            unsafe/

Run:
    python train_resnet50.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.resnet50_model import SmartMineResNet50

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Datasets & Loaders ────────────────────────────────────────────────────────
train_dataset = datasets.ImageFolder("dataset/train", transform=train_transform)
val_dataset   = datasets.ImageFolder("dataset/val",   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)

num_classes = len(train_dataset.classes)
print(f"Classes ({num_classes}): {train_dataset.classes}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = SmartMineResNet50(num_classes=num_classes)
model.to(device)

# ── Loss & Optimiser ──────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ── Training Loop ─────────────────────────────────────────────────────────────
EPOCHS = 25
best_val_acc = 0.0

for epoch in range(EPOCHS):

    # --- Train ---
    model.train()
    running_loss = 0.0
    correct = total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = torch.max(outputs, 1)
        total        += labels.size(0)
        correct      += (predicted == labels).sum().item()

    train_acc = 100.0 * correct / total

    # --- Validate ---
    model.eval()
    val_correct = val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100.0 * val_correct / val_total
    scheduler.step()

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}]  "
          f"Loss: {running_loss:.4f}  "
          f"Train Acc: {train_acc:.2f}%  "
          f"Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/resnet50_smartmine.pth")
        print(f"  ✅ Best model saved (val_acc={val_acc:.2f}%)")

print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.2f}%")
