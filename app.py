import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd

from model import CNN   

st.set_page_config(
    page_title="Weather Vision",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Constants
CLASS_NAMES = [
    "dew", "fogsmog", "frost", "glaze", "hail", "lightning",
    "rain", "rainbow", "rime", "sandstorm", "snow"
]

CLASS_EMOJI = {
    "dew": "💧", "fogsmog": "🌫️", "frost": "❄️", "glaze": "🧊",
    "hail": "🌨️", "lightning": "⚡", "rain": "🌧️", "rainbow": "🌈",
    "rime": "🥶", "sandstorm": "🏜️", "snow": "☃️",
}

MODEL_PATH = "weather_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom CSS 
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    h1, h2, h3, p, label, .stMarkdown {
        color: #f1f5f9 !important;
    }
    .hero {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #94a3b8 !important;
        font-size: 1.05rem;
    }
    .result-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        backdrop-filter: blur(6px);
    }
    .result-card .emoji {
        font-size: 3.5rem;
        line-height: 1;
    }
    .result-card .label {
        font-size: 1.8rem;
        font-weight: 700;
        text-transform: capitalize;
        margin: 0.3rem 0;
        color: #38bdf8 !important;
    }
    .result-card .confidence {
        color: #94a3b8 !important;
        font-size: 1rem;
    }
    div[data-testid="stFileUploader"] section {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #38bdf8, #6366f1);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Model loading 
@st.cache_resource
def load_model():
    model = CNN(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def predict(image: Image.Image, model):
    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    return probs


# Sidebar
with st.sidebar:
    st.markdown("### 🌦️ Weather Vision")
    st.markdown(
        "A CNN-based image classifier that recognizes **11 weather conditions** "
        "from a single photo, built with a custom 3-layer convolutional network."
    )
    st.markdown("---")
    st.markdown("**Classes recognized:**")
    cols = st.columns(2)
    for i, name in enumerate(CLASS_NAMES):
        cols[i % 2].markdown(f"{CLASS_EMOJI[name]} {name.capitalize()}")
    st.markdown("---")
    st.caption("Model: Custom CNN (3 conv layers)  ")


# Main content
st.markdown("""
<div class="hero">
    <h1>Weather Image Classifier</h1>
    <p>Upload a photo and let the model tell you what weather it's showing</p>
</div>
""", unsafe_allow_html=True)

st.write("")

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("#### 📤 Upload an image")
    uploaded_file = st.file_uploader(
        "Drop a weather photo here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Uploaded image")

with right:
    st.markdown("#### 🔍 Prediction")

    if uploaded_file is None:
        st.info("Upload an image on the left to see the prediction here.")
    else:
        try:
            model = load_model()
        except FileNotFoundError:
            st.error(
                f"Couldn't find `{MODEL_PATH}`. Make sure the trained model "
                f"file is in the same folder as this app."
            )
            st.stop()

        with st.spinner("Analyzing image..."):
            probs = predict(image, model)

        top_idx = probs.argmax()
        top_class = CLASS_NAMES[top_idx]
        top_conf = probs[top_idx] * 100

        st.markdown(f"""
        <div class="result-card">
            <div class="emoji">{CLASS_EMOJI[top_class]}</div>
            <div class="label">{top_class}</div>
            <div class="confidence">{top_conf:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown("**All class probabilities**")

        df = pd.DataFrame({
            "Class": [f"{CLASS_EMOJI[c]} {c.capitalize()}" for c in CLASS_NAMES],
            "Probability": probs * 100
        }).sort_values("Probability", ascending=True)

        st.bar_chart(df.set_index("Class"), horizontal=True, height=380)

st.write("")
st.markdown("---")

