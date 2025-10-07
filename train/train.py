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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图片和掩码的数据集
data_path = 'segimg/data/seg.json'

class CustomDataset(Dataset):
    def __init__(self, file_dir, transform=None):
        image_dir = []
        mask_dir = []
        with open(file_dir, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                image_dir.append(line[0])
                mask_dir.append(line[1])
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        print("图像地址、mask地址", self.image_dir[0], self.mask_dir[0])

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        
        image = Image.open(self.image_dir[idx]).convert('RGB')
        label = Image.open(self.mask_dir[idx]).convert('L') 

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = CustomDataset(file_dir=data_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = ResNetUNet(n_classes=1).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


model.train()
print("train........")
for epoch in range(10):  
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(loss)
torch.save(model.state_dict(), 'segimg/model/test/segimg.pth')  # 保存路径

# 评估
# model.eval()
# with torch.no_grad():
    
#     img = Image.open('segimg/data/img/2（齿痕 苔润）_aug_1_aug_2_aug_3.jpg')
#     transform_img = transform(img)
#     transform_img = transform_img.unsqueeze(0).to(device)  # Add batch dimension and move to device

#     # Inference
#     pred = model(transform_img)

#     # Apply sigmoid function to convert model output to probabilities
#     pred = torch.sigmoid(pred)

#     # Use threshold (0.5) to convert probabilities to binary image
#     pred_mask = (pred > 0.5).float()  # Generate binary mask

#     # Convert mask to image format
#     pred_mask = pred_mask.squeeze().cpu().numpy()  # Remove batch dimension and convert to numpy array
#     pred_mask = (pred_mask * 255).astype(np.uint8)  # Convert to 0-255 range

#     # Save mask image
#     mask_image = Image.fromarray(pred_mask)  # Create image from numpy array
#     mask_image.save('mask_output.png')  # Save path
