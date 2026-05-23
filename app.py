"""
RetinexNet Enhancement Server
Run: python app.py
Then open: http://localhost:5000
"""

import os
import io
import base64
import json
import torch
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from models.retinex_model import RetinexNet

app = Flask(__name__, static_folder="static")

# ── Load model once at startup ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[RetinexNet] Using device: {DEVICE}")

model = RetinexNet().to(DEVICE)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.eval()
print("[RetinexNet] Model loaded.")

# Folders
LOW_DIR    = os.path.join("dataset", "test", "low")
NORMAL_DIR = os.path.join("dataset", "test", "normal")


def enhance_image_array(bgr_img: np.ndarray) -> np.ndarray:
    """Takes a BGR uint8 numpy array, returns enhanced BGR uint8 array."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (256, 256))
    t   = torch.tensor(rgb.astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t)
    result = out.squeeze().permute(1, 2, 0).cpu().numpy()
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def img_to_b64(bgr_img: np.ndarray) -> str:
    """Encode BGR numpy array to base64 PNG string."""
    ok, buf = cv2.imencode(".png", bgr_img)
    return base64.b64encode(buf).decode("utf-8")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/gallery")
def gallery():
    """Return list of test images with their base64 thumbnails."""
    items = []
    if not os.path.isdir(LOW_DIR):
        return jsonify({"error": f"Folder not found: {LOW_DIR}"}), 404

    for fname in sorted(os.listdir(LOW_DIR)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(LOW_DIR, fname)
        img  = cv2.imread(path)
        if img is None:
            continue
        thumb = cv2.resize(img, (120, 90))
        items.append({
            "name": fname,
            "thumb": img_to_b64(thumb)
        })
    return jsonify(items)


@app.route("/api/enhance/gallery/<filename>")
def enhance_gallery(filename):
    """Enhance a specific file from the test/low folder."""
    low_path = os.path.join(LOW_DIR, filename)
    if not os.path.isfile(low_path):
        return jsonify({"error": "File not found"}), 404

    low_img      = cv2.imread(low_path)
    enhanced_img = enhance_image_array(low_img)

    # Resize source to 256 for consistent display
    low_resized = cv2.resize(low_img, (256, 256))

    # Check if ground-truth normal exists
    normal_b64 = None
    normal_path = os.path.join(NORMAL_DIR, filename)
    if os.path.isfile(normal_path):
        normal_img   = cv2.imread(normal_path)
        normal_img   = cv2.resize(normal_img, (256, 256))
        normal_b64   = img_to_b64(normal_img)

    return jsonify({
        "original":  img_to_b64(low_resized),
        "enhanced":  img_to_b64(enhanced_img),
        "reference": normal_b64
    })


@app.route("/api/enhance/upload", methods=["POST"])
def enhance_upload():
    """Enhance a user-uploaded image (sent as base64 JSON)."""
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    # Decode base64 → numpy
    img_bytes = base64.b64decode(data["image"])
    nparr     = np.frombuffer(img_bytes, np.uint8)
    img       = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    enhanced = enhance_image_array(img)
    original = cv2.resize(img, (256, 256))

    return jsonify({
        "original": img_to_b64(original),
        "enhanced": img_to_b64(enhanced)
    })


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    print("\n  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
