import torch


class DefaultConfig:
    '''
    config for inference
    '''
    weight_path = 'Assets/centernet_resnet50_voc.pth'
    classes_path = 'Assets/voc_classes.txt'
    #   输入网络的图片resize后的大小
    img_input_shape = [512, 512]
    #   是否使用letterbox_image方法，对输入图像进行不失真的resize
    letterbox_image = True
    # 得分置信度
    score_threshold = 0.3
    # 是否需要nms
    need_nms = True
    #   非极大抑制所用到的nms_iou大小
    nms_iou_threshold = 0.3
    # 是否使用 gpu
    cuda = True if torch.cuda.is_available() else False
    pretrained = False

    '''
    config for training
    '''