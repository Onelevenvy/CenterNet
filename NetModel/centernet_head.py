import torch
import torch.nn as nn


class CenterNetHead(nn.Module):
    def __init__(self, num_classes=20, channel=64, bn_momentum=0.1):
        super(CenterNetHead, self).__init__()

        self.cls_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, padding=0)
        )

        self.wh_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0)
        )

        self.reg_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        hm = self.cls_head(x).sigmoid_()
        wh = self.wh_head(x)
        offset = self.reg_head(x)
        return hm, wh, offset


# if __name__ == '__main__':
#     mode = CenterNetHead()
#     input = torch.ones(1,64,128,128)
#     output = mode(input)
#     for i in range(len(output)):
#         print(output[i].shape)

# torch.Size([1, 20, 128, 128])
# torch.Size([1, 2, 128, 128])
# torch.Size([1, 2, 128, 128])