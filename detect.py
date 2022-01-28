from time import time
import torch
from PIL import Image
from Detector.centernet_detector import CenterNetDetector
from Utils.draw_bbox import draw_rectangle
from Utils.process_img import preprocess_img
from Utils.utils import get_classes
import numpy as np


class InferenceConfig:
    weight_path = 'Assets/centernet_resnet50_voc.pth'
    classes_path = 'Assets/voc_classes.txt'
    #   输入网络的图片resize后的大小
    img_input_shape = [512, 512]
    #   是否使用letterbox_image方法，对输入图像进行不失真的resize
    letterbox_image = False,
    # 得分置信度
    score_threshold = 0.3
    # 是否需要nms
    need_nms = True
    #  非极大抑制所用到的nms_iou大小
    nms_iou_threshold = 0.3
    # 是否使用 gpu
    cuda = True if torch.cuda.is_available() else False
    pretrained = False


if __name__ == "__main__":

    class_names, num_classes = get_classes(InferenceConfig.classes_path)
    '''
    输入图片路径，单张图片检测
    '''
    while True:
        img = input('Input image path:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            begin_time = time()
            # 图片预处理，包括resize,归一化，转成tensor等等
            img, img_shape = preprocess_img(image, config=InferenceConfig)
            detector = CenterNetDetector(InferenceConfig.img_input_shape, img_shape, num_classes,
                                         config=InferenceConfig, mode='inference')
            with torch.no_grad():
                detection_resuls = detector(img)
            end_time = time()
            run_time = end_time - begin_time
            print('inference time:', run_time)
            # 绘制bbox
            top_label = np.array(detection_resuls[0][:, 5], dtype='int32')
            top_conf = detection_resuls[0][:, 4]
            top_boxes = detection_resuls[0][:, :4]
            draw_rectangle(top_label, top_conf, top_boxes, image, InferenceConfig.img_input_shape, num_classes,
                           class_names)
            image.show()
