🖼️ Image Segmentation + Caption Generation (U-Net + CNN-LSTM)
📌 Overview

This project combines Image Segmentation (using a U-Net model) and Image Captioning (using a CNN-LSTM model) to both segment and describe images automatically.

The segmentation model highlights different regions in the image (e.g., airplane, person, sky).

The captioning model generates natural-language descriptions based on image features.

The app is deployed using Streamlit, with all data and code hosted on GitHub.

⚙️ Features

✅ Automatic detection of uploaded images from the images/ folder
✅ U-Net based segmentation with colorized class masks
✅ CNN-LSTM based caption generation
✅ Side-by-side display of:

Original image

Predicted segmentation mask

Mask index map

(Optional) Ground-truth segmentation mask

🧠 Model Architecture
🟢 Segmentation (U-Net)

Encoder: 3 levels of convolution + max-pooling

Decoder: transpose convolution + skip connections

Output: pixel-wise segmentation mask with color coding

🔵 Captioning (CNN-LSTM)

Encoder: InceptionV3 pretrained on ImageNet

Decoder: LSTM + embedding + dense layers

Output: descriptive text caption
