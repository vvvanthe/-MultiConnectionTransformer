import torch
import torch.nn as nn
import math

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean)

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

class _Residual_Block(nn.Module):
    def __init__(self, dim=64):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output,identity_data)
        return output

class Net(nn.Module):
    def __init__(self,dim=64,num_block=16):
        super(Net, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.sub_mean = MeanShift(rgb_mean, -1)

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = self.make_layer(_Residual_Block, num_block)

        self.conv_mid = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),

        )

        self.conv_output = nn.Conv2d(in_channels=dim, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x_in):

        out = self.sub_mean(x_in)
        out = self.conv_input(out)
        residual = out
        out = self.conv_mid(self.residual(out))
        out = torch.add(out,residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        out = self.add_mean(out)

        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import time
if __name__ == '__main__':
    start_time = time.time()
    upscale = 2
    window_size = 7
    height = 120
    width = 200

    model = Net()


    x = torch.ones((3, 3, height, width))
    print(x.shape)
    y = model(x)
    print(count_parameters(model))

    # print(model.flops())
    print(y.shape)
    print("--- %s seconds ---" % (time.time() - start_time))