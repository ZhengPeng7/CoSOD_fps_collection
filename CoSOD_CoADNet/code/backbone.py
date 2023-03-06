from CoSOD_CoADNet.code.common_packages import *
from CoSOD_CoADNet.code.modules import CAM
from CoSOD_CoADNet.code.ops import CU
import torch.utils.model_zoo as model_zoo


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation[1], bias=False, dilation=dilation[1])
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

    
class DRN_A(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(DRN_A, self).__init__()
        self.out_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.avgpool = nn.AvgPool2d(28, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                     nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=(dilation, dilation)))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def Build_DRN_A_50(pretrained=False, **kwargs):
    model = DRN_A(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    return model


class Backbone_Wrapper_VGG16(nn.Module):
    def __init__(self):
        super(Backbone_Wrapper_VGG16, self).__init__()
        bb = models.vgg16_bn(pretrained=False).features
        self.C1 = nn.Sequential()
        self.C2 = nn.Sequential()
        self.C3 = nn.Sequential()
        self.C4 = nn.Sequential()
        self.C5 = nn.Sequential()
        for layer_index, (name, sub_module) in enumerate(bb.named_children()):
            if layer_index>=0 and layer_index<=5:
                self.C1.add_module(name, sub_module)           
            if layer_index>=7 and layer_index <= 12:
                self.C2.add_module(name, sub_module)
            if layer_index>=14 and layer_index <= 22:
                self.C3.add_module(name, sub_module)
            if layer_index>=24 and layer_index <= 32:
                self.C4.add_module(name, sub_module)
            if layer_index>=34 and layer_index <= 42:
                self.C5.add_module(name, sub_module)
        # channels = [64, 128, 256, 512, 512]
        self.C4_Att = CAM(512, 8)
        self.C5_Att = CAM(512, 8)
        self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, image):
        # image: [B, 3, 128, 128]
        ftr_s = self.C3(self.pool(self.C2(self.C1(image)))) # [B, 256, 64, 64]
        ftr_d = self.C5_Att(self.C5(self.pool(self.C4_Att(self.C4(self.pool(ftr_s)))))) # [B, 512, 16, 16]
        return ftr_d
    
    
class Backbone_Wrapper_ResNet50(nn.Module):
    def __init__(self):
        super(Backbone_Wrapper_ResNet50, self).__init__()
        bb = models.resnet50(pretrained=False)
        bb.conv1.stride = (1, 1)
        self.C1 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)
        self.C2 = bb.layer1
        self.C3 = bb.layer2
        self.C4 = bb.layer3
        self.C5 = bb.layer4
        # channels = [64, 256, 512, 1024, 2048]
        self.C4_Att = CAM(1024, 8)
        self.C5_Att = CAM(2048, 16)
        self.squeeze_channels = CU(2048, 1024, 1, True, 'relu')
    def forward(self, image):
        # image: [B, 3, 128, 128]
        ftr_s = self.C3(self.C2(self.C1(image))) # [B, 512, 64, 64]
        ftr_d = self.C5_Att(self.C5(self.C4_Att(self.C4(ftr_s)))) # [B, 2048, 16, 16]
        ftr_d = self.squeeze_channels(ftr_d) # [B, 1024, 16, 16]
        return ftr_d
    
    
class Backbone_Wrapper_Dilated_ResNet50(nn.Module):
    def __init__(self):
        super(Backbone_Wrapper_Dilated_ResNet50, self).__init__()
        bb = Build_DRN_A_50(pretrained=False)
        bb.conv1.stride = (1, 1)
        self.C1 = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.C2 = bb.layer1
        self.C3 = bb.layer2
        self.C4 = bb.layer3
        self.C5 = bb.layer4
        # channels = [64, 256, 512, 1024, 2048]
        self.C4_Att = CAM(1024, 8)
        self.C5_Att = CAM(2048, 16)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.squeeze_channels = CU(2048, 1024, 1, True, 'relu')
    def forward(self, image):
        # image: [B, 3, 128, 128]
        ftr_s = self.C2(self.C1(image)) # [B, 256, 64, 64]
        ftr_d = self.C5_Att(self.C5(self.C4_Att(self.C4(self.pool(self.C3(ftr_s)))))) # [B, 2048, 16, 16]
        ftr_d = self.squeeze_channels(ftr_d) # [B, 1024, 16, 16]
        return ftr_d
    
    
    
