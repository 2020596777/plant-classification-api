import os
import io
import torch
from PIL import Image
from torchvision import transforms
from model.encoder import ConvEmbedding
from utils.features import extract_color_histogram, extract_texture_features

# Configuration
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "dataset"
TEST_IMAGE_FOLDER = "test_image"
CHECKPOINT_PATH = "checkpoints/best_protonet_encoder.pth"

# Load model
print("📦 Loading model...")
model = ConvEmbedding().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()
print("✅ Model loaded.")

# Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Prepare support embeddings
def _prepare_support_embeddings():
    print("📚 Preparing support embeddings...")
    classes = sorted(os.listdir(DATA_PATH))
    sup_images, sup_labels = [], []

    for i, cls in enumerate(classes):
        cls_dir = os.path.join(DATA_PATH, cls)
        img_name = sorted(os.listdir(cls_dir))[0]
        img_path = os.path.join(cls_dir, img_name)

        img = Image.open(img_path).convert('RGB')
        img_t = transform(img)
        sup_images.append(img_t)
        sup_labels.append(i)

    sup_x = torch.stack(sup_images).to(DEVICE)
    sup_color = extract_color_histogram(sup_x).to(DEVICE)
    sup_texture = extract_texture_features(sup_x).to(DEVICE)

    with torch.no_grad():
        emb_sup = model(sup_x, sup_color, sup_texture)

    print("✅ Support embeddings ready.")
    return emb_sup, classes

# Load once
EMB_SUP, CLASSES = _prepare_support_embeddings()

# Predict from bytes
def predict_image_bytes(image_bytes: bytes) -> str:
    try:
        print("🖼️ Predicting image from bytes...")
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(DEVICE)

        color_feat = extract_color_histogram(img_t).to(DEVICE)
        texture_feat = extract_texture_features(img_t).to(DEVICE)

        print(f"🧪 Color feature shape: {color_feat.shape}")
        print(f"🧪 Texture feature shape: {texture_feat.shape}")

        with torch.no_grad():
            emb_qry = model(img_t, color_feat, texture_feat)
            print(f"🔍 Query embedding shape: {emb_qry.shape}")

            dists = torch.cdist(emb_qry, EMB_SUP)
            print(f"📏 Distances: {dists}")

            pred_idx = torch.argmin(dists).item()
            print(f"✅ Prediction index: {pred_idx}, Class: {CLASSES[pred_idx]}")

        return CLASSES[pred_idx]

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return "Unknown"

# Predict from image path
def predict(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return predict_image_bytes(f.read())

# CLI Testing
if __name__ == "__main__":
    for img_name in os.listdir(TEST_IMAGE_FOLDER):
        img_path = os.path.join(TEST_IMAGE_FOLDER, img_name)
        pred = predict(img_path)
        print(f"Image '{img_name}' predicted as: {pred}")