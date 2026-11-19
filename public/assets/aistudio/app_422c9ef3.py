
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models, transforms
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
st.set_page_config(
    page_title="Deepfake Image Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("AI-Powered Deepfake Image Detection")
st.write("Upload a human face image to detect whether it is REAL or FAKE.")

# Model paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
])

# Load model
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model

try:
    model = load_model()
    st.success("Trained model loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# Grad-CAM
def generate_gradcam(image_tensor, class_idx):
    gradients = []
    activations = []

    target_layer = model.features[-1]

    def forward_hook(module, inputs, output):
        activations.append(output)
        output.register_hook(
            lambda grad: gradients.append(grad)
        )

    hook = target_layer.register_forward_hook(forward_hook)

    model.zero_grad(set_to_none=True)

    output = model(image_tensor)
    score = output[:, class_idx]
    score.backward()

    hook.remove()

    weights = gradients[0].mean(
        dim=(2, 3), keepdim=True
    )

    cam = (weights * activations[0]).sum(dim=1)
    cam = F.relu(cam)

    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    return cam


# Upload image
uploaded_file = st.file_uploader(
    "Choose a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    image_tensor = transform(image).unsqueeze(0).to(device)

    if st.button("Detect Deepfake"):

        model.eval()

        with torch.enable_grad():
            output = model(image_tensor)

            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(
                probabilities, dim=1
            )

            st.write("Raw model output:", output.detach().cpu().numpy())
            st.write("FAKE probability:", probabilities[0][0].item())
            st.write("REAL probability:", probabilities[0][1].item())

            st.write("FAKE probability:", probabilities[0][0].item() * 100)
            st.write("REAL probability:", probabilities[0][1].item() * 100)

            class_names = ["fake", "real"]
            predicted_class = prediction.item()

            cam = generate_gradcam(
                image_tensor,
                predicted_class
            )

        st.subheader("Detection Result")

        if predicted_class == 0:
            st.error("FAKE IMAGE DETECTED")
        else:
            st.success("REAL IMAGE DETECTED")

        st.write(
            f"Confidence: {confidence.item() * 100:.2f}%"
        )

        # Grad-CAM visualization
        original = image.resize((224, 224))
        original_array = np.array(original)

        heatmap = plt.get_cmap("jet")(cam)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)

        overlay = (
            0.5 * original_array + 0.5 * heatmap
        ).astype(np.uint8)

        st.subheader("Grad-CAM Explanation")
        st.image(
            overlay,
            caption="Areas influencing the model prediction",
            width=350
        )

        st.warning(
            "This AI prediction is not definitive proof. "
            "Results may be incorrect, especially for unfamiliar images."
        )
