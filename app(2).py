import os
import io

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from model import load_model

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "model_weights", "cnn_vit_hybrid_5class_merged.pth")
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_DESCRIPTIONS = {
    "plastic": "Bottles, containers, packaging, and other plastic items.",
    "metal": "Cans, foil, and other metal items.",
    "glass": "Bottles, jars, and other glass items.",
    "paper_cardboard": "Paper, cardboard, and boxes.",
    "trash": "Items that don't belong to a recyclable category above.",
}

st.set_page_config(page_title="Waste Classifier", page_icon="\u267b", layout="centered")


@st.cache_resource
def get_model():
    if not os.path.exists(CHECKPOINT_PATH):
        return None, None, None
    model, classes, config = load_model(CHECKPOINT_PATH, device=DEVICE)
    return model, classes, config


def preprocess(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def predict(model, classes, image: Image.Image):
    tensor = preprocess(image).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    ranked = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    return ranked


st.title("Waste classifier")
st.write(
    "Upload a photo of a single waste item. The model sorts it into one of "
    "five categories using a fused ResNet50 + DeiT-Small architecture."
)

model, classes, config = get_model()

if model is None:
    st.error(
        f"No model checkpoint found at `{CHECKPOINT_PATH}`. "
        "Add your trained .pth file there, or edit CHECKPOINT_PATH in app.py "
        "to point at your own checkpoint. See README.md for details."
    )
    st.stop()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    ranked = predict(model, classes, image)
    top_class, top_prob = ranked[0]

    with col2:
        st.subheader(top_class.replace("_", " ").title())
        st.write(f"Confidence: {top_prob * 100:.1f}%")
        st.caption(CLASS_DESCRIPTIONS.get(top_class, ""))

    st.write("All class probabilities")
    for class_name, prob in ranked:
        label = class_name.replace("_", " ").title()
        st.write(f"{label} — {prob * 100:.1f}%")
        st.progress(min(float(prob), 1.0))
else:
    st.info("Upload an image to get a prediction.")

with st.expander("Model details"):
    st.write(f"Classes: {', '.join(classes)}")
    if config:
        st.write(f"CNN backbone: {config.get('cnn_model', 'resnet50')}")
        st.write(f"ViT backbone: {config.get('vit_model', 'deit_small_patch16_224')}")
        st.write(f"Input image size: {config.get('image_size', IMAGE_SIZE)}px")
    st.write(f"Running on: {DEVICE}")
