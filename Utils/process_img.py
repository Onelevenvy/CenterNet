import torch
from torchvision.transforms import transforms
from Utils.utils import resize_image, cvtColor
import numpy as np
from defaultconfig import DefaultConfig


def preprocess_img(image, config=None):
    if config is None:
        config = DefaultConfig
    else:
        config = config
    image_shape = np.array(np.shape(image)[0:2])
    # 转成RGB图像
    image = cvtColor(image)
    # #   给图像增加灰条，实现不失真的resize,也可以直接resize进行识别
    img = resize_image(image, (config.img_input_shape[1], config.img_input_shape[0]),
                       config.letterbox_image)
    # 转成tensor
    img = transforms.ToTensor()(img)
    # 归一化
    img = transforms.Normalize([0.406, 0.456, 0.485], [0.225, 0.224, 0.229], inplace=True)(img)
    # # img = transforms.Normalize([0.40789655, 0.44719303, 0.47026116], [0.2886383, 0.27408165, 0.27809834],
    # #                             inplace=True)(img)

    # (3,h,w)-->（1,2,h,w）
    img = torch.unsqueeze(img, dim=0)

    return img, image_shape
