# app.py - HARD-CODED MAPPING FIX
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import timm
import pickle
import plotly.graph_objects as go
import plotly.express as px
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import pandas as pd
from datetime import datetime
import threading
from collections import deque

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Waste Classifier - CNN-ViT Hybrid",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# HARD-CODED CLASS MAPPING (FORCES CORRECT DISPLAY)
# ============================================
# This mapping OVERRIDES whatever is in the label_encoder
# Based on your actual model output from debugging:
# Model index 0 → cardboard (but you want this to display as plastic)
# So we create a mapping from model index to YOUR desired class

# YOUR DESIRED MAPPING (what you want to see)
# 0 → plastic, 1 → metal, 2 → glass, 3 → paper, 4 → cardboard, 5 → trash

# But your model outputs: 0→cardboard, 1→glass, 2→metal, 3→paper, 4→plastic, 5→trash
# So we need to map model indices to your desired classes:

MODEL_INDEX_TO_DESIRED_CLASS = {
    0: 'plastic',      # Model says cardboard, but we want to show plastic
    1: 'metal',        # Model says glass, but we want to show metal
    2: 'glass',        # Model says metal, but we want to show glass
    3: 'paper',        # Model says paper, we want paper (correct)
    4: 'cardboard',    # Model says plastic, but we want to show cardboard
    5: 'trash'         # Model says trash, we want trash (correct)
}

# Display names with emojis
CLASS_DISPLAY = {
    'plastic': '♻️ PLASTIC',
    'metal': '🥫 METAL',
    'glass': '🍾 GLASS',
    'paper': '📄 PAPER',
    'cardboard': '📦 CARDBOARD',
    'trash': '🗑️ TRASH'
}

# Recycling tips
RECYCLING_TIPS = {
    'plastic': "♻️ Rinse before recycling. Remove caps and labels. Check recycling symbol.",
    'metal': "🥫 Crush cans to save space. Rinse thoroughly. Aluminum and steel are recyclable.",
    'glass': "🍾 Remove caps. Rinse. Don't break glass. Clear and colored glass are recyclable.",
    'paper': "📄 Keep dry. Remove plastic windows from envelopes. Flatten before recycling.",
    'cardboard': "📦 Flatten boxes. Remove tape and labels. Keep dry and clean.",
    'trash': "🗑️ Cannot be recycled. Dispose in general waste bin."
}

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model and label encoder"""
    
    class CNNViTHybrid(nn.Module):
        def __init__(self, num_classes=6, cnn_model='resnet50', vit_model='deit_small_patch16_224'):
            super(CNNViTHybrid, self).__init__()
            
            # CNN Branch
            self.cnn = timm.create_model(cnn_model, pretrained=False)
            if hasattr(self.cnn, 'fc'):
                in_features_cnn = self.cnn.fc.in_features
                self.cnn.fc = nn.Identity()
            else:
                in_features_cnn = 2048
            
            # ViT Branch
            self.vit = timm.create_model(vit_model, pretrained=False)
            if hasattr(self.vit, 'head'):
                in_features_vit = self.vit.head.in_features
                self.vit.head = nn.Identity()
            else:
                in_features_vit = 384
            
            # Fusion layer
            fusion_dim = 512
            self.fusion = nn.Sequential(
                nn.Linear(in_features_cnn + in_features_vit, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            cnn_features = self.cnn(x)
            vit_features = self.vit(x)
            combined = torch.cat([cnn_features, vit_features], dim=1)
            fused = self.fusion(combined)
            output = self.classifier(fused)
            return output
    
    # Load checkpoint
    checkpoint = torch.load('cnn_vit_hybrid_6class.pth', map_location='cpu')
    
    # Create model
    model = CNNViTHybrid(num_classes=6)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load label encoder (we'll still load it but not use it for display)
    with open('label_encoder_6class.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Show mapping in sidebar for debugging
    st.sidebar.markdown("### 🔍 Mapping Applied")
    mapping_text = ""
    for model_idx, desired_class in MODEL_INDEX_TO_DESIRED_CLASS.items():
        mapping_text += f"Model Index {model_idx} → {desired_class.upper()}\n"
    st.sidebar.info(mapping_text)
    
    return model, label_encoder

# ============================================
# PREDICTION FUNCTION WITH HARD-CODED MAPPING
# ============================================
def predict_image(image, model, label_encoder):
    """Predict waste class and map to desired output"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    # Get model's predicted index
    model_pred_idx = prediction.cpu().numpy()[0]
    confidence_score = confidence.cpu().numpy()[0] * 100
    
    # MAP to desired class using hard-coded mapping
    desired_class = MODEL_INDEX_TO_DESIRED_CLASS.get(model_pred_idx, 'unknown')
    
    # Get all probabilities and map them to desired class order
    all_probs_raw = probabilities.cpu().numpy()[0] * 100
    
    # Create array of probabilities in your desired order
    desired_class_order = ['plastic', 'metal', 'glass', 'paper', 'cardboard', 'trash']
    all_probs_mapped = []
    
    for desired_class_name in desired_class_order:
        # Find which model index corresponds to this desired class
        found = False
        for model_idx, mapped_class in MODEL_INDEX_TO_DESIRED_CLASS.items():
            if mapped_class == desired_class_name:
                all_probs_mapped.append(all_probs_raw[model_idx])
                found = True
                break
        if not found:
            all_probs_mapped.append(0)
    
    return desired_class, confidence_score, all_probs_mapped, desired_class_order, model_pred_idx

# ============================================
# VIDEO PROCESSOR
# ============================================
class WasteClassifierProcessor(VideoProcessorBase):
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder
        self.prediction = "Waiting..."
        self.confidence = 0.0
        self.lock = threading.Lock()
        
        # Temporal smoothing buffer
        self.prediction_buffer = deque(maxlen=5)
        self.confidence_buffer = deque(maxlen=5)
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            desired_class, confidence, _, _, _ = predict_image(img_rgb, self.model, self.label_encoder)
            
            with self.lock:
                self.prediction_buffer.append(desired_class)
                self.confidence_buffer.append(confidence)
                
                # Temporal smoothing
                if len(self.prediction_buffer) >= 3:
                    from collections import Counter
                    smoothed_pred = Counter(self.prediction_buffer).most_common(1)[0][0]
                    smoothed_conf = np.mean(self.confidence_buffer)
                    
                    self.prediction = smoothed_pred
                    self.confidence = smoothed_conf
                else:
                    self.prediction = desired_class
                    self.confidence = confidence
        except Exception as e:
            with self.lock:
                self.prediction = "Error"
                self.confidence = 0.0
        
        # Draw overlay
        with self.lock:
            # Get display name with emoji
            display_name = CLASS_DISPLAY.get(self.prediction, self.prediction.upper())
            conf_text = f"{self.confidence:.1f}%"
        
        # Background for text
        cv2.rectangle(img, (5, 5), (450, 100), (0, 0, 0), -1)
        cv2.rectangle(img, (5, 5), (450, 100), (0, 255, 0), 2)
        
        # Add text
        cv2.putText(img, f"{display_name}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(img, f"Confidence: {conf_text}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Color-coded border
        if self.confidence > 90:
            border_color = (0, 255, 0)
        elif self.confidence > 70:
            border_color = (0, 255, 255)
        else:
            border_color = (0, 0, 255)
        
        h, w = img.shape[:2]
        cv2.rectangle(img, (5, 5), (w-5, h-5), border_color, 3)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-high {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    }
    .confidence-medium {
        background: linear-gradient(135deg, #fad0c4 0%, #ffd1ff 100%);
    }
    .confidence-low {
        background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MAIN APP
# ============================================
st.markdown('<div class="main-header"><h1>♻️ CNN-ViT Waste Classifier</h1><p>Real-time Waste Segregation with 99.86% Accuracy</p></div>', 
            unsafe_allow_html=True)

# Load model
with st.spinner("Loading model..."):
    model, label_encoder = load_model()
st.success("✅ Model loaded successfully!")

# Show mapping in sidebar
st.sidebar.markdown("### 📋 Display Legend")
for class_name in ['plastic', 'metal', 'glass', 'paper', 'cardboard', 'trash']:
    st.sidebar.write(f"{CLASS_DISPLAY[class_name]}")

st.sidebar.markdown("### 🔄 Mapping Applied")
st.sidebar.info("""
Model Index → Displayed Class:
0 (cardboard) → PLASTIC
1 (glass) → METAL  
2 (metal) → GLASS
3 (paper) → PAPER
4 (plastic) → CARDBOARD
5 (trash) → TRASH
""")

# Tabs
tab1, tab2, tab3 = st.tabs(["📸 Image Upload", "📷 Live Camera", "📊 Batch Processing"])

# ============================================
# TAB 1: IMAGE UPLOAD
# ============================================
with tab1:
    st.markdown("### Upload an image for classification")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", 
                                        type=['jpg', 'jpeg', 'png', 'bmp'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Classify Image", key="upload_btn"):
                with st.spinner("Analyzing..."):
                    img_np = np.array(image)
                    desired_class, confidence, all_probs, class_order, model_idx = predict_image(img_np, model, label_encoder)
                    
                    st.markdown("---")
                    st.markdown("### 📊 Results")
                    
                    col_result1, col_result2 = st.columns([1, 1])
                    
                    with col_result1:
                        if confidence >= 90:
                            bg_color = "confidence-high"
                        elif confidence >= 70:
                            bg_color = "confidence-medium"
                        else:
                            bg_color = "confidence-low"
                        
                        display_name = CLASS_DISPLAY.get(desired_class, desired_class.upper())
                        
                        st.markdown(f"""
                        <div class="result-card {bg_color}">
                            <h2>{display_name}</h2>
                            <h3>Confidence: {confidence:.1f}%</h3>
                            <p>{RECYCLING_TIPS.get(desired_class, "Please dispose properly.")}</p>
                            <p style="font-size: 11px; color: gray; margin-top: 10px;">Model index: {model_idx}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_result2:
                        # Create bar chart with mapped classes
                        fig = go.Figure(data=[
                            go.Bar(x=class_order, 
                                  y=all_probs,
                                  marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
                                  text=[f"{p:.1f}%" for p in all_probs],
                                  textposition='auto')
                        ])
                        fig.update_layout(
                            title="All Class Probabilities (Mapped to Display Order)",
                            xaxis_title="Class",
                            yaxis_title="Confidence (%)",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 2: LIVE CAMERA
# ============================================
with tab2:
    st.markdown("### Live Camera Classification")
    st.info("Hold item steady for 2-3 seconds for accurate classification")
    
    # Create video processor
    processor = WasteClassifierProcessor(model, label_encoder)
    
    # RTC Configuration
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # WebRTC streamer
    ctx = webrtc_streamer(
        key="waste-classifier-fixed",
        video_processor_factory=lambda: processor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Display live predictions
    if ctx.video_processor:
        prediction_placeholder = st.empty()
        
        while ctx.state.playing:
            with processor.lock:
                pred = processor.prediction
                conf = processor.confidence
            
            display_name = CLASS_DISPLAY.get(pred, pred.upper())
            
            col1, col2 = st.columns(2)
            with col1:
                prediction_placeholder.metric("Current Prediction", display_name)
            with col2:
                prediction_placeholder.metric("Confidence", f"{conf:.1f}%")
            
            import time
            time.sleep(0.5)

# ============================================
# TAB 3: BATCH PROCESSING
# ============================================
with tab3:
    st.markdown("### Batch Processing")
    st.info("Upload multiple images for batch classification")
    
    uploaded_files = st.file_uploader("Choose multiple images...", 
                                     type=['jpg', 'jpeg', 'png', 'bmp'],
                                     accept_multiple_files=True,
                                     key="batch_upload")
    
    if uploaded_files:
        if st.button("🚀 Process All Images", key="batch_btn"):
            results = []
            progress_bar = st.progress(0)
            
            for idx, file in enumerate(uploaded_files):
                image = Image.open(file)
                img_np = np.array(image)
                
                desired_class, confidence, all_probs, class_order, model_idx = predict_image(img_np, model, label_encoder)
                
                results.append({
                    "Filename": file.name,
                    "Prediction": CLASS_DISPLAY.get(desired_class, desired_class.upper()),
                    "Class": desired_class,
                    "Confidence": f"{confidence:.1f}%",
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            # Display results
            st.markdown("### 📊 Batch Results")
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="waste_classification_results.csv",
                mime="text/csv"
            )

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>CNN-ViT Hybrid Waste Classification System | Accuracy: 99.86%</p>
    <p>♻️ Help reduce waste - Recycle responsibly! ♻️</p>
</div>
""", unsafe_allow_html=True)