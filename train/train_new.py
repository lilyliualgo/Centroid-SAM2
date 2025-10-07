import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import logging

logging.basicConfig(filename='segimg/model/training_log_20250217.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

num_epochs = 50
device = "cuda:4" if torch.cuda.is_available() else "cpu"

class TongueDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx].replace(".JPG", ".jpg"))#.replace(".jpg", ".png"))

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # 转为 tensor 格式并二值化掩码
        mask = (np.array(mask) > 0).astype(np.float32)
        mask = torch.from_numpy(mask)#.unsqueeze(0)

        return image, mask

# 创建数据变换（调整尺寸和归一化）
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# 加载数据集
image_dir = "segimg/data/_20241203/samples_for_seg_20250217-舌底/img"
mask_dir = "segimg/data/_20241203/samples_for_seg_20250217-舌底/mask_img"
dataset = TongueDataset(image_dir, mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

import torchvision.models.segmentation as segmentation

# 加载预训练模型并调整输出层
model = segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))  # 将输出通道数设为1
model = model.to(device)

import torch.optim as optim
import torch.nn as nn

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


from tqdm import tqdm  # 导入tqdm
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # 用tqdm包装dataloader，显示批次的进度
    with tqdm(dataloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")  # 设置描述信息

        for images, masks in tepoch:
            images, masks = images.to(device), masks.to(device)

            # 前向传播
            outputs = model(images)['out']
            loss = criterion(outputs, masks)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 更新tqdm进度条的后缀信息
            tepoch.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    log_message = f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}"
    logging.info(log_message)

# 保存模型
torch.save(model.state_dict(), "segimg/model/segimg_0217_epoch50_shedi.pth") #1203是目前最好的