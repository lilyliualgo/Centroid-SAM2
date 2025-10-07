import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os

# 原图和JSON文件的路径
image_dir = "data/_20241203/samples_for_seg_20250217-舌底/img"
mask_dir = "data/_20241203/samples_for_seg_20250217-舌底/mask"
mask_img_save_dir = "data/_20241203/samples_for_seg_20250217-舌底/seg_img"

file_list = os.listdir(image_dir)

for file in file_list:
    if "aug" not in file:
        img_path = os.path.join(image_dir, file)
        json_path = os.path.join(mask_dir, file.replace('jpg', 'json').replace('JPG', 'json'))
        mask_path = os.path.join(mask_img_save_dir, file.replace('jpg', 'png').replace('JPG', 'png'))  # 保存为PNG以保留透明度
        
        # 读取JSON文件
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 打开原图
        original_image = Image.open(img_path).convert("RGBA")  # 转换为RGBA模式，添加透明通道
        width, height = original_image.size

        # 创建与原图相同大小的透明掩码图像
        mask = Image.new('L', (width, height), 0)  # L模式为灰度图，初始化为全黑

        # 读取分割数据并生成掩码
        max_len = 0
        for i in range(len(data['objects'])):
            len_seg = len(data['objects'][i]['segmentation'])
            if len_seg > max_len:
                max_len = len_seg
                segmentation = data['objects'][i]['segmentation']

        # 确保坐标是整数
        polygon = [(int(x), int(y)) for x, y in segmentation]

        # 使用白色(255)填充掩码区域
        ImageDraw.Draw(mask).polygon(polygon, outline=255, fill=255)

        # 将掩码应用到原图上，只保留多边形内的区域
        mask = mask.point(lambda p: p > 0 and 255)  # 将掩码二值化
        mask = Image.merge("L", [mask])  # 合并为单通道掩码
        result = Image.composite(original_image, Image.new("RGBA", original_image.size, (0, 0, 0, 0)), mask)

        # 平滑边缘处理
        result = result.filter(ImageFilter.SMOOTH)

        # 将结果图像调整到 512x512 尺寸，保持宽高比例
        result.thumbnail((512, 512), Image.LANCZOS)  # 缩放图像以适应 512x512 尺寸，保持比例
        background = Image.new("RGBA", (512, 512), (0, 0, 0, 0))  # 创建一个 512x512 的透明背景
        offset = ((512 - result.width) // 2, (512 - result.height) // 2)
        background.paste(result, offset)  # 将结果图像粘贴到中心位置

        # 填充多余的区域为背景
        for obj in data['objects']:
            if obj['segmentation'] != segmentation:
                fill_polygon = [(int(x), int(y)) for x, y in obj['segmentation']]
                ImageDraw.Draw(background).polygon(fill_polygon, outline=(0, 0, 0, 0), fill=(0, 0, 0, 0))  # 透明背景

        # 保存处理后的图像
        background.save(mask_path, quality=100)
