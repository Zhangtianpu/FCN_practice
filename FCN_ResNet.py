import torch
from torch import nn
import torchvision

class FCNResNet(nn.Module):

    def __init__(self,ResNet_path,out_channel):
        super(FCNResNet,self).__init__()
        self._out_channel=out_channel
        #load pre-trained resnet model
        resnet=torchvision.models.resnet18(pretrained=False)
        resnet.load_state_dict(torch.load(ResNet_path))
        #获取resnet最后一层卷积输出通道数
        in_features=resnet.fc.in_features
        # replace last two fcn with conv and transpose conv
        self.net=nn.Sequential(*list(resnet.children())[:-2])
        self.net.add_module(name="final_conv",
                       module=nn.Conv2d(in_channels=in_features,
                                        out_channels=self._out_channel,
                                        kernel_size=1,
                                        padding=0,
                                        stride=1
                                        ))
        self.net.add_module(name="transpose_conv",
                       module=nn.ConvTranspose2d(in_channels=self._out_channel,
                                                 out_channels=self._out_channel,
                                                 kernel_size=64,
                                                 padding=16,
                                                 stride=32))

        # initialize kernel value of transpose conv with bilinear interpolation
        transpose_conv_weight=self.bilinear_kernel(in_channels=self._out_channel,
                             out_channels=self._out_channel,
                             kernel_size=64)
        self.net.transpose_conv.weight.data.copy_(transpose_conv_weight)


    def forward(self,input):
        """

        :param input: [batch,channel,h,w]
        :return:
        """
        return self.net(input)


    def bilinear_kernel(self,in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = (torch.arange(kernel_size).reshape(-1, 1),
              torch.arange(kernel_size).reshape(1, -1))
        filt = (1 - torch.abs(og[0] - center) / factor) * \
               (1 - torch.abs(og[1] - center) / factor)
        weight = torch.zeros(
            (in_channels, out_channels, kernel_size, kernel_size))
        weight[range(in_channels), range(out_channels), :, :] = filt
        return weight