import torch
import cv2
import numpy as np
import os
from models.retinex_model import RetinexNet

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
model = RetinexNet().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# --- Paths ---
test_folder   = "dataset/test/low"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(test_folder):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    path = os.path.join(test_folder, img_name)
    img  = cv2.imread(path)

    if img is None:
        print(f"Skipping unreadable file: {path}")
        continue

    # Fix: BGR → RGB before processing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))

    img_tensor = (img.astype('float32') / 255.0)
    img_tensor = torch.tensor(img_tensor).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    # Convert back to uint8 image
    result = output.squeeze().permute(1, 2, 0).cpu().numpy()
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

    # Fix: RGB → BGR before saving with OpenCV
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(output_folder, img_name), result)
    print(f"Saved: {img_name}")

print("Done. Check the output/ folder.")