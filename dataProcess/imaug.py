import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
import os
import random

# 设置图像和掩码路径
image_dir = "segimg/data/img"
mask_img_dir = "segimg/data/mask_img"

# 获取图像文件列表
file_list = os.listdir(image_dir)

for file in file_list:
    img_path = os.path.join(image_dir, file)
    mask_path = os.path.join(mask_img_dir, file.replace('jpg', 'jpg').replace('JPG', 'jpg'))
    #增强图片的后缀取名
    img_aug_path = os.path.join(image_dir, file.replace('.jpg', '_aug_2.jpg').replace('.JPG', '_aug_2.jpg'))
    mask_aug_path = os.path.join(mask_img_dir, file.replace('.jpg', '_aug_2.jpg'))

    # 使用PIL读取图像和掩码
    img = Image.open(img_path)
    mask = Image.open(mask_path)

    # 将PIL图像转换为numpy数组
    img = np.array(img)
    mask = np.array(mask)

    # 如果掩码是灰度图，扩展为三通道
    if len(mask.shape) == 2:  # 如果是灰度图
        mask = np.expand_dims(mask, axis=2)  # 添加一个通道维度
        mask = np.repeat(mask, 3, axis=2)  # 重复成三通道

    # 定义共同操作的增强策略
    seq_common_1 = iaa.Sequential([
        iaa.Fliplr(1)  # 水平翻转
    ])

    seq_common_2 = iaa.Sequential([
        iaa.Flipud(1)  # 垂直翻转
    ])

    # 定义仅图像的增强策略
    seq_image_only = iaa.Sequential([
        iaa.SomeOf((3, 5), [
            iaa.GaussianBlur(sigma=(0, 3.0)),  # 高斯模糊
            iaa.MotionBlur(k=3),  # 运动模糊
            iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # 添加高斯噪声
            iaa.SaltAndPepper(p=0.05),  # 椒盐噪声
            iaa.Multiply((0.8, 1.2)),  # 随机改变亮度
            iaa.LinearContrast((0.6, 1.4)),  # 对比度增强
            iaa.MultiplyHueAndSaturation((0.8, 1.2)),  # 改变色调和饱和度
            iaa.GammaContrast((0.8, 1.2))  # Gamma对比度调整
        ])
    ])

    # 增强图像和掩码的共同操作
    def augment_common(image, mask, seq):
        return seq(images=[image, mask])  # 通过命名参数传递

    # 应用图像单独增强
    def augment_image_only(image):
        return seq_image_only(images=[image])[0]

    # 随机选择增强方式
    rand = random.randint(0, 10)
    
    # 增强图像和掩码
    if rand > 6:
        aug_img, aug_mask = augment_common(img, mask, seq_common_1)
        # 对增强后的图像应用仅图像的增强
        aug_img = augment_image_only(aug_img)

    elif rand > 3:
        aug_img, aug_mask = augment_common(img, mask, seq_common_2)
        aug_img = augment_image_only(aug_img)

    else:
        aug_img = augment_image_only(img)
        aug_mask = mask  # 如果没有对掩码进行增强，保持原始掩码

    # 如果需要把掩码转换回单通道
    aug_mask = aug_mask[:, :, 0]

    # 将增强后的图像和掩码转换回PIL图像并保存
    aug_img_pil = Image.fromarray(aug_img)
    aug_mask_pil = Image.fromarray(aug_mask)

    # 保存增强后的图像和掩码
    aug_img_pil.save(img_aug_path,quality=100)
    aug_mask_pil.save(mask_aug_path,quality=100)

