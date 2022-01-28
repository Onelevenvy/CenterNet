import torch.nn as nn
import torch


class resnet50_Decoder(nn.Module):
    def __init__(self, inplanes, bn_momentum=0.1):
        super(resnet50_Decoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.in_channels = inplanes
        self.deconv_with_bias = False
        self.deconv_layers = self._make_deconv_layer(
            layers_num=3, planes=[256, 128, 64], kernel_size=[4, 4, 4]
        )

    def _make_deconv_layer(self, layers_num, planes, kernel_size):
        layers = []
        for i in range(layers_num):
            out_channels = planes[i]
            kernel = kernel_size[i]
            layers.append(
                nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=kernel,
                                   stride=2, padding=1, output_padding=0, bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(out_channels, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)

# if __name__ == '__main__':
#
#     model = resnet50_Decoder(2048)
#     iput = torch.ones(1, 2048, 16, 16)
#     out = model(iput)
#     print(out.shape)
#     # torch.Size([1, 64, 128, 128])
