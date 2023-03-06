from CoSOD_CoADNet.code.common_packages import *
from CoSOD_CoADNet.code.ops import *
from CoSOD_CoADNet.code.misc import *


class CAM(nn.Module): 
    # Channel Attention Module
    def __init__(self, in_channels, squeeze_ratio):
        super(CAM, self).__init__()
        inter_channels = in_channels // squeeze_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        fc_1 = FC(in_channels, inter_channels, False, 'relu')
        fc_2 = FC(inter_channels, in_channels, False, 'none')
        self.fc = nn.Sequential(fc_1, fc_2)
    def forward(self, x):
        # x: [B, in_channels, fh, fw]
        # y: [B, in_channels, fh, fw]
        avg_pooled = self.avg_pool(x).squeeze(-1).squeeze(-1) # [B, in_channels]
        max_pooled = self.max_pool(x).squeeze(-1).squeeze(-1) # [B, in_channels]
        ap_weights = self.fc(avg_pooled) # [B, in_channels]
        mp_weights = self.fc(max_pooled) # [B, in_channels]
        weights = F.sigmoid(ap_weights + mp_weights) # [B, in_channels]
        y = x * weights.unsqueeze(-1).unsqueeze(-1) + x
        return y
    
    
class SAM(nn.Module): 
    # Spatial Attention Module
    def __init__(self, conv_ks):
        super(SAM, self).__init__()
        assert conv_ks>=3 and np.mod(conv_ks+1, 2)==0
        self.conv = CU(ic=2, oc=1, ks=conv_ks, is_bn=False, na='sigmoid')
    def forward(self, x):
        # x: [B, ic, fh, fw]
        # y: [B, ic, fh, fw]
        avg_pooled = torch.mean(x, dim=1, keepdim=True) # [B, 1, fh, fw]
        max_pooled = torch.max(x, dim=1, keepdim=True)[0] # [B, 1, fh, fw]
        cat_pooled = torch.cat((avg_pooled, max_pooled), dim=1) # [B, 2, fh, fw]
        weights = self.conv(cat_pooled) # [B, 1, fh, fw]
        y = x * weights + x
        return y
    
    
class LCA(nn.Module): 
    # Local Context Aggregation
    def __init__(self, in_channels, squeeze_ratio, dr_list):
        super(LCA, self).__init__()
        inter_channels = in_channels // squeeze_ratio
        self.conv_1 = CU(in_channels, inter_channels, 1, True, 'relu')
        self.conv_2 = DilConv3(in_channels, inter_channels, True, 'relu', dr_list[0])
        self.conv_3 = DilConv3(in_channels, inter_channels, True, 'relu', dr_list[1])
        self.conv_4 = DilConv3(in_channels, inter_channels, True, 'relu', dr_list[2])
        self.fusion = CU(inter_channels*4, in_channels, 1, True, 'relu')
    def forward(self, x):
        # x: [B, in_channels, fh, fw]
        # y: [B, in_channels, fh, fw]
        x_1 = self.conv_1(x) # [B, inter_channels, fh, fw]
        x_2 = self.conv_2(x) # [B, inter_channels, fh, fw]
        x_3 = self.conv_3(x) # [B, inter_channels, fh, fw]
        x_4 = self.conv_4(x) # [B, inter_channels, fh, fw]
        x_f = self.fusion(torch.cat((x_1, x_2, x_3, x_4), dim=1)) # [B, in_channels, fh, fw]
        y = x_f + x
        return y
    
    
class GCA(nn.Module):
    # Global Context Aggregation
    def __init__(self, in_channels, squeeze_ratio):
        super(GCA, self).__init__()
        self.map_q = CU(in_channels, in_channels//squeeze_ratio, 1, False, 'none')
        self.map_k = CU(in_channels, in_channels//squeeze_ratio, 1, False, 'none')
        self.map_v = CU(in_channels, in_channels, 1, False, 'none')
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, ftr):
        # ftr: [B, C, H, W]
        # ftr_fusion: # [B, C, H, W]
        B, C, H, W = ftr.size()
        N = H * W
        ftr_q = self.map_q(ftr).view(B, -1, N).transpose(1, 2).contiguous() # [B, N, C']
        ftr_k = self.map_k(ftr).view(B, -1, N) # [B, C', N]
        aff_mat = F.softmax(torch.bmm(ftr_q, ftr_k), dim=1) # [B, N, N]
        ftr_v = self.map_v(ftr).view(B, -1, N) # [B, C, N]
        ftr_gca = torch.bmm(ftr_v, aff_mat).view(B, C, H, W) # [B, C, H, W]
        ftr_fusion = ftr + ftr_gca * self.gamma # [B, C, H, W]
        return ftr_fusion
    
    
class IaSH(nn.Module):
    # Intra-Saliency Head
    def __init__(self, bb_type):
        super(IaSH, self).__init__()
        assert bb_type in ['VGG16', 'ResNet50', 'Dilated_ResNet50']
        self.bb_type = bb_type
        upsample_2x = nn.Upsample(scale_factor=2, mode='bilinear')
        if bb_type == 'VGG16':
            self.conv_1 = nn.Sequential(upsample_2x, CU(512, 256, 3, True, 'relu'))
            self.conv_2 = nn.Sequential(upsample_2x, CU(256, 128, 3, True, 'relu'))
            self.output = nn.Sequential(CU(128, 128, 3, True, 'relu'), CU(128, 64, 3, True, 'relu'), CU(64, 1, 3, False, 'sigmoid'))
        if bb_type == 'ResNet50':
            self.conv_1 = nn.Sequential(CU(1024, 512, 1, True, 'relu'), upsample_2x, CU(512, 256, 3, True, 'relu'))
            self.conv_2 = nn.Sequential(upsample_2x, CU(256, 128, 3, True, 'relu'))
            self.output = nn.Sequential(CU(128, 128, 3, True, 'relu'), CU(128, 64, 3, True, 'relu'), CU(64, 1, 3, False, 'sigmoid'))
        if bb_type == 'Dilated_ResNet50':          
            self.conv_1 = nn.Sequential(CU(1024, 512, 1, True, 'relu'), upsample_2x, CU(512, 256, 3, True, 'relu'))
            self.conv_2 = nn.Sequential(upsample_2x, CU(256, 128, 3, True, 'relu'))
            self.output = nn.Sequential(CU(128, 128, 3, True, 'relu'), CU(128, 64, 3, True, 'relu'), CU(64, 1, 3, False, 'sigmoid'))
    def forward(self, si_ftr):
        # si_ftr: [Bs, D, 16, 16]
        sm = self.output(self.conv_2(self.conv_1(si_ftr))) # [Bs, 1, 64, 64]
        return sm
    
    
class OIaSG(nn.Module):
    # Online-Intra Saliency Guidance
    def __init__(self):
        super(OIaSG, self).__init__()
        self.fusion_1 = CU(ic=2, oc=1, ks=3, is_bn=False, na='sigmoid')
        self.fusion_2 = CU(ic=2, oc=1, ks=3, is_bn=False, na='sigmoid')
    def forward(self, gi_sm, gi_ftr_d):
        # gi_sm: [Bg*M, 1, 16, 16]
        # gi_ftr_d: [Bg*M, Cd, 16, 16]
        ftr_avg = torch.mean(gi_ftr_d, dim=1, keepdim=True) # [Bg*M, 1, 16, 16]
        ftr_max = torch.max(gi_ftr_d, dim=1, keepdim=True)[0] # [Bg*M, 1, 16, 16]
        ftr_concat = torch.cat((ftr_avg, ftr_max), dim=1) # [Bg*M, 2, 16, 16]
        ftr_fusion = self.fusion_1(ftr_concat) # [Bg*M, 1, 16, 16]
        gm = self.fusion_2(torch.cat((ftr_fusion, gi_sm), dim=1)) # [Bg*M, 1, 16, 16]
        intra_sal_ftr = gi_ftr_d + gi_ftr_d * gm # [Bg*M, Cd, 16, 16]
        return intra_sal_ftr
    
    
class BlockFusion(nn.Module):
    def __init__(self, in_channels, squeeze_ratio):
        super(BlockFusion, self).__init__()
        inter_channels = in_channels // squeeze_ratio
        conv_1 = CU(in_channels, in_channels, 1, True, 'relu')
        conv_2 = CU(in_channels, inter_channels, 3, True, 'relu')
        conv_3 = CU(inter_channels, in_channels, 3, True, 'relu')
        self.conv = nn.Sequential(conv_1, conv_2, conv_3)
    def forward(self, ftr):
        # ftr: [B, C, H, W]
        # ftr_fusion: [B, C, H, W]
        ftr_fusion = self.conv(ftr) # [B, C, H, W]
        return ftr_fusion + ftr
    
    
class GGD(nn.Module):
    # Gated Group Distribution
    def __init__(self, in_channels, squeeze_ratio):
        super(GGD, self).__init__()
        inter_channels = in_channels // squeeze_ratio
        self.squeeze_channel = nn.Sequential(CU(in_channels*2, in_channels//2, 1, True, 'relu'), CAM(in_channels//2, 8))
        conv_1 = CU(in_channels//2, inter_channels, 3, True, 'relu')
        conv_2 = CU(inter_channels, in_channels, 3, False, 'sigmoid')
        self.gie = nn.Sequential(conv_1, conv_2)
    def forward(self, G, U):
        # G: [Bg, D, H, W]
        # U: [Bg, D, H, W] 
        P = self.gie(self.squeeze_channel(torch.cat((G, U), dim=1))) # [B, D, H, W]
        ftr_cosal = (G - U) * P + U # [B, D, H, W], "G*P+U*(1-P)"
        return ftr_cosal
    
    
class GCPD(nn.Module):
    # Group Consistency Preserving Decoder
    def __init__(self, in_channels, out_channels, squeeze_ratio):
        super(GCPD, self).__init__()
        tf_1 = CU(in_channels, in_channels//2, 1, True, 'relu')
        tf_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        tf_3 = CU(in_channels//2, in_channels//2, 3, True, 'relu')
        self.transform = nn.Sequential(tf_1, tf_2, tf_3)
        inter_channels = in_channels // squeeze_ratio
        smlp_1 = FC(in_channels, inter_channels, False, 'relu')
        smlp_2 = FC(inter_channels, in_channels//2, False, 'sigmoid')
        self.smlp = nn.Sequential(smlp_1, smlp_2)
        self.conv = CU(in_channels//2, out_channels, 3, True, 'relu')
    def forward(self, X):
        # X: [B, M, ic, fh, fw]
        # X_d: [B, M, oc, fh*2, fw*2]
        B, M, ic, fh, fw = X.size()
        X_t = self.transform(X.view(-1, ic, fh, fw)) # [B*M, ic//2, fh*2, fw*2]
        V = GAP(X_t).squeeze(-1).squeeze(-1).view(B, M, ic//2) # [B, M, ic//2]
        y = torch.sum(V * F.softmax(V, dim=1), dim=1) # [B, ic//2]
        V_cat_y = torch.cat((V, y.unsqueeze(1).repeat(1, M, 1)), dim=-1) # [B, M, ic]
        A = self.smlp(V_cat_y.view(-1, ic)) # [B, M, ic//2]
        X_t = X_t * A.unsqueeze(-1).unsqueeze(-1) + X_t # [B*M, ic//2, fh*2, fw*2]
        X_d = self.conv(X_t).view(B, M, -1, fh*2, fw*2) # [B, M, oc, fh*2, fw*2]
        return X_d
    
    
class CoSH(nn.Module):
    # Co-Saliency Head
    def __init__(self, in_channels):
        super(CoSH, self).__init__()
        dims = [in_channels, in_channels, 64, 1]
        head_1 = CU(dims[0], dims[1], 3, True, 'relu')
        head_2 = CU(dims[1], dims[2], 3, True, 'relu')
        head_3 = CU(dims[2], dims[3], 3, False, 'sigmoid')
        self.head = nn.Sequential(head_1, head_2, head_3)
    def forward(self, x):
        # x: [B, M, in_channels, H, W]
        # y: [B, M, 1, H, W]
        B, M, C, H, W = x.size()
        y = self.head(x.view(-1, C, H, W)).view(B, M, 1, H, W)
        return y
    
    
