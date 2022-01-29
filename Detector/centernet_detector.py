import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.backends import cudnn
from torchvision.ops import nms
from NetModel.centernet import CenterNet
from defaultconfig import DefaultConfig


class DetectionHead(nn.Module):
    '''
    假设输入图片的大小为[512,512]
    input
    hm: [bs,nc,128,128]
    wh: [bs,2,128,128]
    offset: [bs,2,128,128]
    '''

    def __init__(self, num_classes, input_shape, image_shape, letterbox_image, score_threshold, need_nms,
                 nms_iou_threshold, cuda, ):
        super(DetectionHead, self).__init__()
        self.image_shape = image_shape
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.letterbox_image = letterbox_image
        self.score_threshold = score_threshold
        self.need_nms = need_nms
        self.nms_iou_threshold = nms_iou_threshold
        self.cuda = cuda

    def forward(self, inputs):
        hms, hw, offset = inputs
        outputs_afer_decode = self.decode_bbox(hms, hw, offset, self.score_threshold, self.cuda)
        predictions = self.postprocess(outputs_afer_decode, self.need_nms, self.image_shape,
                                       self.input_shape, self.letterbox_image, self.nms_iou_threshold)
        return predictions

    def pool_nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2
        heatmapmax = nn.functional.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (heatmapmax == heat).float()
        return heat * keep

    def decode_bbox(self, pred_hms, pred_whs, pred_offsetctxys, confidence=0.5, cuda=True):
        '''
        input:
        pred_hms: [bs,nc,h,w]
        prd_wh:[bs,2,h,w]
        prd_offsetctxy:[bs,2,h,w]
        :return:
        bbox:[x1,y1,x2,y2]
        '''
        pred_hms = self.pool_nms(pred_hms)  # [bs,nc,h,w]
        bs, nc, h, w = pred_hms.shape
        detects_results = []

        for i in range(bs):
            # [ nc, h, w]-->[h,w,nc]-->[h*w,nc]
            heat_map = pred_hms[i].permute(1, 2, 0).view([-1, nc])  # [h*w,nc]
            pred_wh = pred_whs[i].permute(1, 2, 0).view([-1, 2])  # [h*w,2]
            pred_offsetctxy = pred_offsetctxys[i].permute(1, 2, 0).view([-1, 2])  # [h*w,2]

            # 生成网格
            yv, xv = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')  # [h,w]
            xv, yv = xv.flatten().float(), yv.flatten().float()  # [h*w,]

            if cuda:
                xv = xv.cuda()
                yv = yv.cuda()
            # 取种类置信度的最大值，及相应的位置（及对应的种类） dim=-1 在 nc维度
            class_conf, class_pred = torch.max(heat_map, dim=-1)  # [h*w,nc]

            mask = class_conf > confidence  # [h*w,nc]

            # 用得分置信度筛选
            pred_wh_mask = pred_wh[mask]  # [h*w,2]
            pred_offsetctxy_mask = pred_offsetctxy[mask]  # [h*w,2]

            if len(pred_wh_mask) == 0:  # 如果batch内该图跑没有检测结果，继续下一张图片
                detects_results.append([])
                continue
            # 调整后的预测框中心

            afteroffsetx = torch.unsqueeze(xv[mask] + pred_offsetctxy_mask[..., 0], -1)  # [h*w,1]
            afteroffsety = torch.unsqueeze(yv[mask] + pred_offsetctxy_mask[..., 1], -1)  # [h*w,1]

            # 预测框的宽高
            pred_w, pred_h = pred_wh_mask[..., 0:1], pred_wh_mask[..., 1:2]  # [h*w,1]

            x1 = afteroffsetx - pred_w / 2  # [hw,1]
            y1 = afteroffsety - pred_h / 2
            x2 = afteroffsetx + pred_w / 2
            y2 = afteroffsety + pred_h / 2
            bboxes = torch.cat([x1, y1, x2, y2], dim=1)  # [h*w,4]

            # 归一化
            bboxes[:, [0, 2]] /= w
            bboxes[:, [1, 3]] /= h
            # bboxe [hw,4]  class_conf [hw,] class_pred [hw,]
            detects_result = torch.cat([bboxes, torch.unsqueeze(class_conf[mask], dim=-1),
                                        torch.unsqueeze(class_pred[mask], dim=-1).float()], dim=-1)  # [hw,4+1+1]

            detects_results.append(detects_result)
        return detects_results

    def centernet_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_xy = (box_xy - offset) * scale
            box_wh *= scale

        box_mins = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape[1:2], image_shape[0:1]] * 2, axis=-1)
        # boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        print(boxes)
        return boxes

    # 预测后处理
    def postprocess(self, prediction, need_nms, image_shape, input_shape, letterbox_image, nms_thres=0.3):
        '''
        input
        prediction:[hw,4+1+1]     4+种类置信度1+种类1
        :return

        '''
        output = [None for _ in range(len(prediction))]

        for i, image_pred in enumerate(prediction):
            detections = prediction[i]
            if len(detections) == 0:
                continue

            #   获得预测结果中包含的所有种类，labels取unique，提高效率
            unique_labels = detections[:, -1].cpu().unique()

            if detections.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                #   按classes遍历，获得某一类得分筛选后全部的预测结果
                detections_class = detections[detections[:, -1] == c]
                if need_nms:

                    keep = nms(
                        detections_class[:, :4],  # 坐标
                        detections_class[:, 4],  # 种类置信度
                        nms_thres
                    )
                    max_detections = detections_class[keep]
                else:
                    max_detections = detections_class

                output[i] = max_detections if output[i] is None else torch.cat([output[i], max_detections])

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_center_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:,
                                                                                                         0:2]
                output[i][:, :4] = self.centernet_correct_boxes(box_center_xy, box_wh, input_shape, image_shape,
                                                                letterbox_image)
        return output


class CenterNetDetector(nn.Module):

    def __init__(self, img_input_shape, img_shape, num_classes=20, config=None, mode='inference'):
        super(CenterNetDetector, self).__init__()
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config
        self.input_shape = img_input_shape
        self.mode = mode
        self.num_classes = num_classes
        self.img_shape = img_shape
        if mode == 'inference':
            self.detector = DetectionHead(self.num_classes, self.input_shape, self.img_shape,
                                          self.config.letterbox_image, self.config.score_threshold,
                                          self.config.need_nms, self.config.nms_iou_threshold, self.config.cuda)

            self.generate_inference()

    def generate_inference(self):
        self.model = CenterNet(num_classes=self.num_classes, pretrained=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(self.config.weight_path, map_location=device))
        self.model = self.model.eval()
        print('===>Success loading NetModel from {}'.format(self.config.weight_path))
        if self.config.cuda:
            self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True
            self.model = self.model.cuda()

    def forward(self, inputs):
        if self.mode == 'inference':
            img = inputs
            hm, wh, offset = self.model(img)
            output = self.detector([hm, wh, offset])
            return output

#


# if __name__ == '__main__':
#     #     hms = torch.ones(1, 20, 128, 128)
#     #     wh = torch.ones(1, 2, 128, 128)
#     #     xy = torch.ones(1, 2, 128, 128)
#     #     inputs = [hms,wh,xy]
#     #     model = DetectionHead(20,[800,600])
#     #     out = model(inputs)
#     #
#     #     print(out[0].shape)
#     #     print('xxx')
#
#     img = torch.ones(1, 3, 512, 512)
#     net = CenterNetDetector([512, 512], mode='inference')
#
#     model = CenterNet(num_classes=20, pretrained=False)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.load_state_dict(torch.load('Assets/centernet_resnet50_voc.pth', map_location=device))
#     model = model.eval()
#     print('weight loaded')
#
#     with torch.no_grad():
#         out = net(img)
#         print(out[0].shape)
