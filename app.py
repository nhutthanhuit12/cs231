import streamlit as st
import numpy as np
import cv2
import joblib
import json
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image


st.set_page_config(
    page_title="Phân loại loài hoa",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .prediction-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .warning-card {
        background-color: #fff3cd;
        color: #856404;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ffeeba;
        text-align: center;
    }
    h1 { color: #ff4b4b; }
    /* Tùy chỉnh thanh progress bar */
    .stProgress > div > div > div > div {
        background-color: #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
@st.cache_resource
def load_resources():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "archive", "cat_to_name.json")
    model_path = os.path.join(current_dir, "best_resnet_head.keras")
    le_path = os.path.join(current_dir, "label_encoder.pkl")
    
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    try:
        classifier_model = load_model(model_path)
        le = joblib.load(le_path)
    except Exception as e:
        return None, None, None, None, str(e)

    cat_to_name = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            cat_to_name = json.load(f)
    
    return base_model, classifier_model, le, cat_to_name, None

with st.spinner():
    base_model, classifier, le, cat_to_name, error_msg = load_resources()

if base_model is None or classifier is None:
    st.error(f" Lỗi: Không tìm thấy file Model hoặc Label Encoder!\nChi tiết: {error_msg}")
    st.stop()

# ==========================================
with st.sidebar:
    st.title("Phân loại loài hoa")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📸 Tải ảnh lên:", type=["jpg", "png", "jpeg"])
    
    st.markdown("---")

# ==========================================
if uploaded_file is not None:
    with st.spinner():
        try:
            image = Image.open(uploaded_file)
            img_array = np.array(image.convert('RGB'))
            img_resized = cv2.resize(img_array, (224, 224))
            
            img_batch = np.expand_dims(img_resized, axis=0).astype(np.float32)
            img_preprocessed = preprocess_input(img_batch)
            
            features = base_model.predict(img_preprocessed, verbose=0)
            prediction = classifier.predict(features)
            
            confidence = np.max(prediction) * 100
            top_1_idx = np.argmax(prediction)
            
            class_id = le.inverse_transform([top_1_idx])[0]
            real_name = cat_to_name.get(str(class_id), f"Class {class_id}")
            
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi xử lý ảnh: {e}")
            st.stop()

    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### Ảnh")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### Kết quả")
        
        if confidence < 50:
            st.markdown(f"""
            <div class="warning-card">
                <h2>Đây không phải là hoa</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card">
                <h2 style='margin:0; color:#28a745;'>{real_name.title()}</h2>
                <p style='font-size: 60px; margin: 10px 0;'></p>
                <p style='color: #555;'>Độ chính xác: <b>{confidence:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Hiển thị Top 3 chi tiết
            st.markdown("#### Top 3 loài hoa nhận dạng:")
            
            for idx in top_3_indices:
                c_id = le.inverse_transform([idx])[0]
                c_name = cat_to_name.get(str(c_id), str(c_id))
                prob = prediction[0][idx] * 100
                
                row_name, row_bar, row_val = st.columns([3, 5, 1.5])
                with row_name: 
                    st.write(f"**{c_name.title()}**")
                with row_bar: 
                    st.progress(int(prob))
                with row_val: 
                    st.write(f"{prob:.1f}%")
