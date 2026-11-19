"""Preserved and repaired Streamlit application (originally app_422c9ef3.py).

Enhancements over original app.py:
  - Enforces class_to_idx == {"fake": 0, "real": 1} from checkpoint
  - Supports WEBP alongside JPG/JPEG/PNG
  - Adds configurable low-confidence warning cutoff (default 0.65)
  - Preserves Grad-CAM visualization on model.features[-1]
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

st.set_page_config(
    page_title="Aegis Forensics — Deepfake Image Detection",
    layout="centered",
)

st.title("Aegis Forensics — Deepfake Image Detection")
st.write("Upload an image to detect whether it is FAKE or REAL.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
EXPECTED_MAPPING = {"fake": 0, "real": 1}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    if checkpoint.get("class_to_idx") != EXPECTED_MAPPING:
        raise ValueError(
            f"Unexpected class_to_idx {checkpoint.get('class_to_idx')}; expected {EXPECTED_MAPPING}."
        )

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


try:
    model = load_model()
    st.success(f"Trained model loaded from {MODEL_PATH.name} ({device})")
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()


def generate_gradcam(image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
    gradients = []
    activations = []
    target_layer = model.features[-1]

    def forward_hook(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    hook = target_layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        grad_input = image_tensor.detach().clone().requires_grad_(True)
        output = model(grad_input)
        score = output[:, class_idx]
        score.backward()
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations[0]).sum(dim=1))
        cam = F.interpolate(
            cam.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False
        )
        cam_np = cam.squeeze().detach().cpu().numpy()
        return (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
    finally:
        hook.remove()


threshold = st.slider("Low-Confidence Warning Cutoff", 0.50, 0.95, 0.65, 0.05)
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=320)
    image_tensor = transform(image).unsqueeze(0).to(device)

    if st.button("Detect Deepfake"):
        start = time.perf_counter()
        with torch.inference_mode():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
        elapsed = time.perf_counter() - start

        fake_prob = float(probabilities[0].item())
        real_prob = float(probabilities[1].item())
        confidence = max(fake_prob, real_prob)
        predicted_class = 0 if fake_prob > real_prob else 1
        verdict = "FAKE" if fake_prob > real_prob else "REAL"

        cam = generate_gradcam(image_tensor, predicted_class)

        st.subheader(f"Verdict: {verdict}")
        st.write(f"Confidence: {confidence * 100:.2f}% | Inference time: {elapsed:.3f}s")
        if confidence < threshold:
            st.warning("Low confidence — prediction is uncertain.")
        st.write(f"FAKE (Class 0): {fake_prob * 100:.2f}% | REAL (Class 1): {real_prob * 100:.2f}%")

        original_array = np.array(image.resize((224, 224)))
        heatmap = (plt.get_cmap("jet")(cam)[:, :, :3] * 255).astype(np.uint8)
        overlay = (0.5 * original_array + 0.5 * heatmap).astype(np.uint8)
        st.image(overlay, caption="Grad-CAM Explanation Overlay", width=350)
        st.warning(
            "Probabilistic estimate from EfficientNet-B0—not definitive proof of authenticity."
        )
