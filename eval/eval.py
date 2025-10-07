import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from unet import ResNetUNet
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)  # Python的随机种子
    np.random.seed(seed)  # NumPy的随机种子
    torch.manual_seed(seed)  # PyTorch的随机种子
    torch.cuda.manual_seed(seed)  # 如果你使用GPU
    torch.backends.cudnn.deterministic = True  # 确保每次使用相同的卷积算法
    torch.backends.cudnn.benchmark = False  # 禁用以提高确定性

# 设置种子
set_seed(42)  # 你可以将42替换为任何你选择的种子数

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNetUNet(n_classes=1).to(device)
# model.load_state_dict()
model.load_state_dict(torch.load('segimg/model/segimg.pth', map_location=device))
# Data loading
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

model.eval()
with torch.no_grad():
    # Load and preprocess image
    img = Image.open('segimg/data/img/2（齿痕 苔润）.jpg')
    transform_img = transform(img)
    transform_img = transform_img.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Inference
    pred = model(transform_img)

    # Apply sigmoid function to convert model output to probabilities
    pred = torch.sigmoid(pred)

    # Use threshold (0.5) to convert probabilities to binary image
    pred_mask = (pred > 0.5).float()  # Generate binary mask

    # Convert mask to image format
    pred_mask = pred_mask.squeeze().cpu().numpy()  # Remove batch dimension and convert to numpy array
    pred_mask = (pred_mask * 255).astype(np.uint8)  # Convert to 0-255 range

    # Save mask image
    mask_image = Image.fromarray(pred_mask)  # Create image from numpy array
    mask_image.save('segimg/data/mask_output.png')  # Save path
