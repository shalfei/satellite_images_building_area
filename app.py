#streamlit run app.py -- --model_path weights.pt
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2

from PIL import Image, ImageOps
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tqdm import tqdm
import os

import argparse
import torch

from models import *
from yolox import *
from det_processing_func import *

st.set_page_config(page_title="Building AI Scan", layout="wide", page_icon="🛰️")



# --- 🎨 CSS СТИЛИЗАЦИЯ ---
st.markdown("""
    <style>
    /* 1. Основной фон и шрифт */
    .stApp {
        background-color: #F5F7F9; /* Очень светло-серый, приятный для глаз */
    }
    
    /* 2. Заголовки */

    h2, h3 {
        color: #37474F;
    }

    /* 3. Кастомизация Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }

    /* 4. Стиль для карточек метрик (GSD, Площадь) */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); /* Легкая тень */
        text-align: center;
    }
    [data-testid="stMetric"]:hover {
        transform: scale(1.02); /* Эффект увеличения при наведении */
    }
    
    /* Цвет цифр в метриках */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #0c343d; 
        font-weight: bold;
    }

    /* 5. Изображения (скругление углов) */
    img {
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    
    /* 6. Кнопка загрузки (File Uploader) */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2E7D32;
        border-radius: 10px;
        padding: 20px;
        background-color: #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)

SEG_MODEL_PATH = 'weights/aspp_unet_resnet34_best_val_iou.pt'
DET_MODEL_PATH = 'weights/yolox_L_best_mAP.pt'

@st.cache_resource
def load_my_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetASPPResNet34().to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_model'])
    model.eval()
    return model, device

@st.cache_resource
def load_detector(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOX().to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_model'])
    model.eval()
    return model


# --- 2. ЛОГИКА ОБРАБОТКИ (наша analyze_single_picture) ---
def process_image(img_pil, model, device):
    w_orig, h_orig = img_pil.size
    max_side = max(w_orig, h_orig)
    
    # Padding до квадрата
    padding = (0, 0, max_side - w_orig, max_side - h_orig)
    img_padded = ImageOps.expand(img_pil, padding, fill=0)
    
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_padded).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, pred_gsd = model(input_tensor)
        proba_map = torch.sigmoid(logits).cpu().squeeze().numpy()
        gsd_val = pred_gsd.item()

    # Обратный ресайз и обрезка
    full_mask = cv2.resize(proba_map, (max_side, max_side), interpolation=cv2.INTER_NEAREST)
    final_mask = full_mask[0:h_orig, 0:w_orig]
    binary_mask = (final_mask > 0.5).astype(np.uint8)
    
    # Площадь
    total_area = np.sum(binary_mask) * (gsd_val ** 2)
    
    return binary_mask, total_area, gsd_val

# --- ИНТЕРФЕЙС STREAMLIT ---
st.set_page_config(page_title="Building AI Scan", layout="wide")
st.title("🛰️ Анализ площади застройки")
st.markdown("**Инструмент для расчета площади застройки по спутниковому или аэрофотоснимку**")

uploaded_file = st.sidebar.file_uploader("Загрузите спутниковый снимок", type=["jpg", "jpeg", "png", "tif"])
model_path = SEG_MODEL_PATH

if uploaded_file:
    model, device = load_my_model(model_path)
    detector = load_detector(DET_MODEL_PATH)

    image = Image.open(uploaded_file).convert("RGB")    
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.image(image, caption="Исходное изображение", width='stretch')
    
    with st.spinner("Нейросеть считает метры..."):
        mask, area, gsd = process_image(image, model, device)

        
    with col2:
        # Красивое наложение маски
        img_np = np.array(image)
        overlay = img_np.copy()
        overlay[mask > 0] = [255, 0, 255]
        blended = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)
        st.image(blended, caption="Результат сегментации", width='stretch')

    with col3:
        mask_to_show = (mask * 255).astype(np.uint8)
        img_mask = Image.fromarray(mask_to_show)
        st.image(img_mask, caption="Маска сегментации", width='stretch')

    # Метрики
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Общая площадь", f"{area:.1f} м²")
    m_col2.metric("GSD (масштаб)", f"{gsd:.2f} м/пкс")

    st.divider()

# - - - ДЕТЕКЦИЯ
    st.header("🔍️ Поиск и подсчет строений")
    final_global_bboxes = visualize_detection_inference_streamlit(model = detector,
        pil_image_input = image,
        device = device,
        conf_threshold=0.2,
        iou_threshold=0.45,
        patch_size=1024,
        overlap_ratio=0.2,
        image_name="Uploaded Image")
