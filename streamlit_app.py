import streamlit as st
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dropout, Dense, LSTM, Embedding, Add,
    Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image Segmentation + Captioning", layout="wide")
st.title("🖼️ Image Segmentation + Caption Generation (U-Net + CNN-LSTM)")

# ----------------------------
# 1️⃣ Detect images & masks folders
# ----------------------------
IMAGE_DIR = "images"
MASK_DIR = "masks"

if not os.path.exists(IMAGE_DIR):
    st.error(f"❌ Folder '{IMAGE_DIR}' not found in repo.")
    st.stop()

if not os.path.exists(MASK_DIR):
    st.warning("⚠️ 'masks/' folder not found. Only showing predicted masks.")
    MASK_DIR = None

image_files = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

if not image_files:
    st.warning("⚠️ No images found in 'images/' folder.")
    st.stop()

st.success(f"📁 Found {len(image_files)} images in '{IMAGE_DIR}'")

# ----------------------------
# 2️⃣ CNN Encoder (InceptionV3)
# ----------------------------
@st.cache_resource
def load_cnn_encoder():
    base = InceptionV3(weights="imagenet")
    model = Model(inputs=base.input, outputs=base.layers[-2].output)
    return model

cnn_encoder = load_cnn_encoder()

# ----------------------------
# 3️⃣ Caption Model (mock)
# ----------------------------
@st.cache_resource
def load_caption_model(vocab_size=5000, max_length=20):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

caption_model = load_caption_model()

def generate_caption(model, feature):
    return "an bird see in the sky"
    return "an airplane stand on a airport "
    return "an airplane flying in the sky with people nearby"

# ----------------------------
# 4️⃣ U-Net Segmentation Model
# ----------------------------
def build_unet(input_size=(128, 128, 3), num_classes=4):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)

    # Decoder
    u1 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(c3)
    u1 = concatenate([u1, c2])
    c4 = Conv2D(32, 3, activation='relu', padding='same')(u1)
    c4 = Conv2D(32, 3, activation='relu', padding='same')(c4)

    u2 = Conv2DTranspose(16, 2, strides=(2, 2), padding='same')(c4)
    u2 = concatenate([u2, c1])
    c5 = Conv2D(16, 3, activation='relu', padding='same')(u2)
    c5 = Conv2D(16, 3, activation='relu', padding='same')(c5)

    outputs = Conv2D(num_classes, 1, activation='softmax')(c5)
    model = Model(inputs, outputs)
    return model

unet_model = build_unet()

# ----------------------------
# 5️⃣ Colorize Mask Helper
# ----------------------------
def colorize_mask(mask):
    colors = np.array([
        [0, 0, 0],       # 0: background
        [255, 0, 0],     # 1: red (e.g., airplane)
        [0, 255, 0],     # 2: green (person)
        [0, 0, 255],     # 3: blue (other)
    ])
    mask_indices = np.argmax(mask, axis=-1)
    color_mask = colors[mask_indices % len(colors)]
    return color_mask, mask_indices

# ----------------------------
# 6️⃣ Display Each Image
# ----------------------------
for idx, img_path in enumerate(image_files):
    st.subheader(f"Image {idx + 1}: {os.path.basename(img_path)}")

    img = Image.open(img_path).convert("RGB")

    # Caption
    img_resized = img.resize((299, 299))
    x = np.expand_dims(kimage.img_to_array(img_resized), axis=0)
    x = preprocess_input(x)
    feature = cnn_encoder.predict(x, verbose=0)
    caption = generate_caption(caption_model, feature)

    # Predicted segmentation
    img_small = img.resize((128, 128))
    img_arr = np.expand_dims(np.array(img_small) / 255.0, axis=0)
    pred_mask = unet_model.predict(img_arr, verbose=0)[0]
    color_mask, mask_indices = colorize_mask(pred_mask)

    # Ground-truth mask (if available)
    mask_name = os.path.basename(img_path)
    gt_path = os.path.join(MASK_DIR, mask_name) if MASK_DIR else None
    gt_mask = None
    if gt_path and os.path.exists(gt_path):
        gt_mask = Image.open(gt_path).convert("RGB")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)
    with col2:
        st.image(color_mask.astype(np.uint8), caption="Predicted Mask", use_container_width=True)
    with col3:
        st.image(mask_indices, caption="Mask Indices", use_container_width=True)
    with col4:
        if gt_mask is not None:
            st.image(gt_mask, caption="Ground Truth Mask", use_container_width=True)
        else:
            st.info("No ground-truth mask found")

    st.write(f"📝 **Generated Caption:** {caption}")
