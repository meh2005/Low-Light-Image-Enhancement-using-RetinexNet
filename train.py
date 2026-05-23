import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from utils.image_loader import LowLightDataset
from models.retinex_model import RetinexNet

# --- Paths ---
train_low    = "dataset/train/low"
train_normal = "dataset/train/normal"

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Dataset & Loader ---
dataset = LowLightDataset(train_low, train_normal)
loader  = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# --- Model ---
model = RetinexNet().to(device)

# --- Loss, Optimizer, Scheduler ---
loss_function = nn.MSELoss()
optimizer     = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler     = StepLR(optimizer, step_size=20, gamma=0.1)  # LR ÷10 at epoch 20

# --- Training ---
epochs = 40

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for low, normal in loader:
        low    = low.to(device)
        normal = normal.to(device)

        output = model(low)
        loss   = loss_function(output, normal)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    scheduler.step()

    print(f"Epoch [{epoch+1:02d}/{epochs}]  Loss: {avg_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")

    # Early warning: if loss is > 2 something is wrong
    if avg_loss > 2.0:
        print("WARNING: Loss > 2.0 — check normalization and dataset pairing!")

torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")