import torch
from torch import nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
from utils import utils
from torch.utils.checkpoint import checkpoint

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ACNet(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        super(ACNet, self).__init__()

        layers = [3, 4, 6, 3]
        block = Bottleneck

        # NC phase branch
        self.inplanes = 64
        self.conv1_N = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1_N = nn.BatchNorm2d(64)
        self.relu_N = nn.ReLU(inplace=True)
        self.maxpool_N = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # use PSPNet extractors
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # ART phase branch
        self.inplanes = 64
        self.conv1_A = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1_A = nn.BatchNorm2d(64)
        self.relu_A = nn.ReLU(inplace=True)
        self.maxpool_A = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_A = self._make_layer(block, 64, layers[0])
        self.layer2_A = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_A = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_A = self._make_layer(block, 512, layers[3], stride=2)

        # PV phase branch
        self.inplanes = 64
        self.conv1_P = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1_P = nn.BatchNorm2d(64)
        self.relu_P = nn.ReLU(inplace=True)
        self.maxpool_P = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_P = self._make_layer(block, 64, layers[0])
        self.layer2_P = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_P = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_P = self._make_layer(block, 512, layers[3], stride=2)
        self.down1 = nn.Conv2d(256, 512, kernel_size=1, stride=2)
        self.down2 = nn.Conv2d(512, 1024, kernel_size=1, stride=2)
        self.down3 = nn.Conv2d(1024, 2048, kernel_size=1, stride=2)

        # merge branch
        self.atten_NC_0 = self.channel_attention(64)
        self.atten_ART_0 = self.channel_attention(64)
        self.atten_PV_0 = self.channel_attention(64)
        self.maxpool_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.atten_NC_1 = self.channel_attention(64 * 4)
        self.atten_ART_1 = self.channel_attention(64 * 4)
        self.atten_PV_1 = self.channel_attention(64 * 4)
        # self.conv_2 = nn.Conv2d(64*4, 64*4, kernel_size=1) #todo 用cat和conv降回通道数
        self.atten_NC_2 = self.channel_attention(128 * 4)
        self.atten_ART_2 = self.channel_attention(128 * 4)
        self.atten_PV_2 = self.channel_attention(128 * 4)
        self.atten_NC_3 = self.channel_attention(256 * 4)
        self.atten_ART_3 = self.channel_attention(256 * 4)
        self.atten_PV_3 = self.channel_attention(256 * 4)
        self.atten_NC_4 = self.channel_attention(512 * 4)
        self.atten_ART_4 = self.channel_attention(512 * 4)
        self.atten_PV_4 = self.channel_attention(512 * 4)

        self.inplanes = 64
        self.layer1_m = self._make_layer(block, 64, layers[0])
        self.layer2_m = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_m = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_m = self._make_layer(block, 512, layers[3], stride=2)

        # final blcok
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weight initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            self._load_resnet_pretrained()

    def encoder(self, NC, ART, PV):
        NC = self.conv1_N(NC)
        NC = self.bn1_N(NC)
        NC = self.relu_N(NC)

        ART = self.conv1_A(ART)
        ART = self.bn1_A(ART)
        ART = self.relu_A(ART)

        PV = self.conv1_P(PV)
        PV = self.bn1_P(PV)
        PV = self.relu_P(PV)

        # atten_NC = self.atten_NC_0(NC)
        # atten_ART = self.atten_ART_0(ART)
        # atten_PV = self.atten_PV_0(PV)
        m0 = NC + ART + PV

        NC = self.maxpool_N(NC)
        ART = self.maxpool_A(ART)
        PV = self.maxpool_P(PV)
        m = self.maxpool_m(m0)

        # block 1
        NC = self.layer1(NC)
        ART = self.layer1_A(ART)
        PV = self.layer1_P(PV)
        m = self.layer1_m(m)

        # atten_NC = self.atten_NC_1(NC)
        # atten_ART = self.atten_ART_1(ART)
        # atten_PV = self.atten_PV_1(PV)

        m1 = m + NC + ART + PV


        # block 2
        NC = self.layer2(NC)
        ART = self.layer2_A(ART)
        PV = self.layer2_P(PV)
        m = self.layer2_m(m1)

        # atten_NC = self.atten_NC_2(NC)
        # atten_ART = self.atten_ART_2(ART)
        # atten_PV = self.atten_PV_2(PV)
        m2 = m + NC + ART + PV


        # block 3
        NC = self.layer3(NC)
        ART = self.layer3_A(ART)
        PV = self.layer3_P(PV)
        m = self.layer3_m(m2)

        # atten_NC = self.atten_NC_3(NC)
        # atten_ART = self.atten_ART_3(ART)
        # atten_PV = self.atten_PV_3(PV)
        m3 = m + NC + ART + PV


        # block 4
        NC = self.layer4(NC)
        ART = self.layer4_A(ART)
        PV = self.layer4_P(PV)
        m = self.layer4_m(m3)

        # atten_NC = self.atten_NC_4(NC)
        # atten_ART = self.atten_ART_4(ART)
        # atten_PV = self.atten_PV_4(PV)
        m4 = m + NC + ART + PV

        # TODO0
        m4 = self.avgpool(m4)
        m4 = m4.reshape(m4.size(0), -1)
        m4 = self.fc(m4)
        return m4
        # return m0, m1, m2, m3, m4  # channel of m is 2048
        # TODO

    def forward(self, NC, ART, PV, phase_checkpoint=False):
        fuses = self.encoder(NC, ART, PV)
        return fuses

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def channel_attention(self, num_channel, ablation=False):
        # todo add convolution here
        pool = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv2d(num_channel, num_channel, kernel_size=1)
        # bn = nn.BatchNorm2d(num_channel)
        activation = nn.Sigmoid()  # todo modify the activation function

        return nn.Sequential(*[pool, conv, activation])

    def _load_resnet_pretrained(self):
        pretrain_dict = model_zoo.load_url(utils.model_urls['resnet50'])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            # print('%%%%% ', k)
            if k in state_dict:
                if k.startswith('conv1'):
                    model_dict[k] = v
                    # print('##### ', k)
                    model_dict[k.replace('conv1', 'conv1_A', 'conv1_P')] = torch.mean(v, 1).data. \
                        view_as(state_dict[k.replace('conv1', 'conv1_A', 'conv1_P')])

                elif k.startswith('bn1'):
                    model_dict[k] = v
                    model_dict[k.replace('bn1', 'bn1_A', 'bn1_P')] = v
                elif k.startswith('layer'):
                    model_dict[k] = v
                    model_dict[k[:6] + '_A' + k[6:]] = v
                    model_dict[k[:6] + '_P' + k[6:]] = v
                    model_dict[k[:6] + '_m' + k[6:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


if __name__ == '__main__':
    s = ACNet()

    print(s)
