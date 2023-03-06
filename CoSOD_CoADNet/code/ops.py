from CoSOD_CoADNet.code.common_packages import *


def US(x, up_scale_factor):
    # up-scale features
    # x: [B, C, H, W]
    # y: [B, C, H_us, W_us]
    isinstance(up_scale_factor, int)
    y = F.interpolate(x, scale_factor=2, mode='bilinear')
    return y


def DS(x, down_scale_factor, ds_mode='max'):
    # down-scale features
    # x: [B, C, H, W]
    # y: [B, C, H_ds, W_ds]
    isinstance(down_scale_factor, int)
    assert ds_mode in ['max', 'avg']
    if ds_mode == 'max':
        y = F.max_pool2d(x, down_scale_factor)
    if ds_mode == 'avg':
        y = F.avg_pool2d(x, down_scale_factor)
    return y


def GAP(x):
    # global average pooling
    # x: [B, C, H, W]
    # y: [B, C, 1, 1]
    y = F.adaptive_avg_pool2d(x, (1, 1))
    return y


class CU(nn.Module):
    # Convolution Unit
    def __init__(self, ic, oc, ks, is_bn, na):
        # ic: input channels
        # oc: output channels
        # ks: kernel size
        # is_bn: True/False
        # na: non-linear activation
        super(CU, self).__init__()
        assert isinstance(ic, int)
        assert isinstance(oc, int)
        assert isinstance(ks, int)
        assert isinstance(is_bn, bool)
        assert isinstance(na, str)
        assert np.mod(ks + 1, 2) == 0
        assert na in ['none', 'relu', 'sigmoid']
        self.is_bn = is_bn
        self.na = na
        self.convolution = nn.Conv2d(ic, oc, ks, padding=(ks-1)//2, bias=False)
        if self.is_bn:
            self.batch_normalization = nn.BatchNorm2d(oc)
        if self.na == 'relu':
            self.activation = nn.ReLU(inplace=True)
        if self.na == 'sigmoid':
            self.activation = nn.Sigmoid()
    def forward(self, x):
        # x: [B, ic, H, W]
        # y: [B, oc, H, W]
        y = self.convolution(x)
        if self.is_bn:
            y = self.batch_normalization(y)
        if self.na != 'none':
            y = self.activation(y)
        return y
    
    
class FC(nn.Module):
    # Fully-Connected Layer
    def __init__(self, ic, oc, is_bn, na):
        super(FC, self).__init__()
        # ic: input channels
        # oc: output channels
        # is_bn: True/False
        # na: non-linear activation
        assert isinstance(ic, int)
        assert isinstance(oc, int)
        assert isinstance(is_bn, bool)
        assert isinstance(na, str)
        assert na in ['none', 'relu', 'sigmoid']
        self.is_bn = is_bn
        self.na = na
        self.linear = nn.Linear(ic, oc, bias=False)
        if self.is_bn:
            self.batch_normalization = nn.BatchNorm1d(oc)
        if self.na == 'relu':
            self.activation = nn.ReLU(inplace=True)
        if self.na == 'sigmoid':
            self.activation = nn.Sigmoid()
    def forward(self, x):
        # x: [B, ic]
        # y: [B, oc]
        y = self.linear(x)
        if self.is_bn:
            y = self.batch_normalization(y)
        if self.na != 'none':
            y = self.activation(y)
        return y
    
    
class DilConv3(nn.Module):
    #  Dilated Convolution with 3*3 kernel size
    def __init__(self, ic, oc, is_bn, na, dr):
        super(DilConv3, self).__init__()
        # ic: input channels
        # oc: output channels
        # is_bn: True/False
        # na: non-linear activation
        # dr: dilation rate
        assert isinstance(ic, int)
        assert isinstance(oc, int)
        assert isinstance(is_bn, bool)
        assert isinstance(na, str)
        assert isinstance(dr, int)
        assert na in ['none', 'relu', 'sigmoid']
        self.is_bn = is_bn
        self.na = na
        self.dil_conv = nn.Conv2d(ic, oc, kernel_size=3, padding=dr, dilation=dr, bias=False)
        if self.is_bn:
            self.batch_normalization = nn.BatchNorm2d(oc)
        if self.na == 'relu':
            self.activation = nn.ReLU(inplace=True)
        if self.na == 'sigmoid':
            self.activation = nn.Sigmoid()
    def forward(self, x):
        # x: [B, ic, H, W]
        # y: [B, oc, H, W]
        y = self.dil_conv(x)
        if self.is_bn:
            y = self.batch_normalization(y)
        if self.na != 'none':
            y = self.activation(y) 
        return y
    
    
    