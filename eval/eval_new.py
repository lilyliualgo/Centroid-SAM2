import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.models.segmentation as segmentation
from PIL import ImageDraw

# 加载模型
def load_model(model_path, device='cuda:2' if torch.cuda.is_available() else 'cpu'):
    model = segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=True)  # 启用辅助分类器
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))  # 设定输出通道数为1
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)  # 加载保存的模型权重
    model.eval()  # 设置为评估模式
    model.to(device)  # 将模型移动到指定设备
    return model

# 预处理单张图像
def preprocess_image(image_path, image_size=(512, 512)):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        # transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # 增加批次维度

# 模型推理
def predict_single_image(model, image_tensor, device='cuda:2' if torch.cuda.is_available() else 'cpu'):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():  # 关闭梯度计算
        output = model(image_tensor)['out']  # 进行推理
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()  # 计算sigmoid并转为numpy
    return (prediction > 0.5).astype(np.uint8)  # 二值化输出

import numpy as np
import cv2

def find_largest_contiguous_area(binary_mask):
    # 查找所有连通组件
    num_labels, labels_im = cv2.connectedComponents(binary_mask.astype(np.uint8))

    # 如果没有找到连通区域，返回 None
    if num_labels <= 1:
        return None, None

    # 统计每个区域的大小
    sizes = np.bincount(labels_im.ravel())
    largest_label = sizes[1:].argmax() + 1  # 1是从1开始的，因此加1

    # 创建一个只包含最大连通区域的掩码
    largest_area_mask = np.zeros_like(binary_mask)
    largest_area_mask[labels_im == largest_label] = 1

    # 计算最大区域的中心点
    M = cv2.moments(largest_area_mask)
    if M['m00'] == 0:  # 避免除以0
        center = None
    else:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        center = (cX, cY)

    return center

def mark_center_on_image(predicted_mask, center, image_size):
    # 将二值化的结果转换为RGB图像
    result_image = Image.fromarray(predicted_mask * 255).convert('RGB')
    
    # 获取中心点坐标
    x, y = center

    # 在结果图像上绘制红色点
    draw = ImageDraw.Draw(result_image)
    radius = 5  # 红点的半径
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))

    return result_image

# 使用示例
if __name__ == "__main__":
    model_path = "Swin-Unet-main/checkpoints/swin_tiny_patch4_window7_224.pth"  # 替换为你的模型权重路径
    image_path = "data/origin/企业微信截图_17298388461790.png"  # 替换为你的测试图像路径
    model = load_model(model_path)
    img = Image.open(image_path).convert("RGB")

    image_tensor = preprocess_image(image_path,img.size)
    predicted_mask = predict_single_image(model, image_tensor)

    center = find_largest_contiguous_area(predicted_mask)
    print("======================",center)


    # 保存结果或进一步处理
    result_image = Image.fromarray(predicted_mask * 255).convert("RGB")  # 转换为可视化图像
    # 在结果图像上绘制红色点
    # 获取中心点坐标
    x, y = center
    draw = ImageDraw.Draw(result_image)
    radius = 5  # 红点的半径
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))
    result_image.save("data/real_mask_output1.png")
