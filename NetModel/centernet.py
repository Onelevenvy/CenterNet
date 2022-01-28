import torch
import torch.nn as nn

from NetModel.Backbone.resnet import resnet50
from NetModel.centernet_head import CenterNetHead
from NetModel.centernet_neck import resnet50_Decoder


class CenterNet(nn.Module):
    def __init__(self, num_classes=20, pretrained=False):
        super(CenterNet, self).__init__()
        self.backbone = resnet50(pretrained)
        self.decoder = resnet50_Decoder(2048)
        self.head = CenterNetHead(num_classes, channel=64)

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        x = self.head(x)
        return x

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


# if __name__ == '__main__':
#     img = torch.ones(1, 3, 512, 512)
#     model = CenterNet()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.load_state_dict(torch.load('../Assets/centernet_resnet50_voc.pth', map_location=device))
#     model.eval()
#     output = model(img)
#     for i in range(len(output)):
#         print(output[i].shape)
