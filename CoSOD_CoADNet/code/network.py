from CoSOD_CoADNet.code.common_packages import *
from CoSOD_CoADNet.code.ops import *
from CoSOD_CoADNet.code.misc import *
from CoSOD_CoADNet.code.modules import *
from CoSOD_CoADNet.code.backbone import Backbone_Wrapper_VGG16, Backbone_Wrapper_ResNet50, Backbone_Wrapper_Dilated_ResNet50

from test_fps import BATCH_SIZE
the_batch_size = [BATCH_SIZE, 5][0]


class CoADNet_Dilated_ResNet50(nn.Module):
    def __init__(self, mode='test', compute_loss=False):
        super(CoADNet_Dilated_ResNet50, self).__init__()
        assert mode in ['train', 'test']
        self.mode = mode
        self.compute_loss = compute_loss
        self.M = the_batch_size
        self.D = 1024
        self.S = 8
        assert np.mod(self.D, self.S) == 0
        self.backbone = Backbone_Wrapper_Dilated_ResNet50()
        self.IaSH = IaSH('Dilated_ResNet50')
        self.online_intra_saliency_guidance = OIaSG()
        self.lca = nn.ModuleList([LCA(self.D//self.S, 4, [1, 3, 5]) for s in range(self.S)])
        self.gca = nn.ModuleList([GCA(self.D//self.S, 2) for s in range(self.S)])
        self.block_fusion = BlockFusion(self.D, 8)
        self.ggd = GGD(self.D, 8)
        decode_dims = [self.D, 512, 256, 128]
        self.gcpd_1 = GCPD(decode_dims[0], decode_dims[1], 8)
        self.gcpd_2 = GCPD(decode_dims[1], decode_dims[2], 2)
        self.gcpd_3 = GCPD(decode_dims[2], decode_dims[3], 2)
        self.cosal_head = CoSH(decode_dims[3])
    def forward(self, gi, si=None, gl=None, sl=None):
        # gi: [Bg, M, 3, 128, 128]
        # si: [Bs, 3, 128, 128]
        # gl: [Bg, M, 1, 128, 128]
        # sl: [Bs, 1, 128, 128]
        M = self.M # number of groups
        D = self.D # backbone feature dimension
        S = self.S # number of blocks
        gi = gi.unsqueeze(0)
        H, W = gi.size(3), gi.size(4)
        if self.mode == 'train':
            assert si is not None
            assert gl is not None
            assert sl is not None
            Bg = gi.size(0)
            Bs = si.size(0)
            si_ftr = self.backbone(si) # [Bs, 1024, 16, 16]
            si_sm = self.IaSH(si_ftr) # [Bs, 1, 64, 64]
        if self.mode == 'test':
            assert si is None
            assert gl is None
            assert sl is None
            Bg = gi.size(0)
        # feature extrcation on gi
        gi_ftr = self.backbone(gi.view(-1, 3, H, W)) # [Bg*M, D, 16, 16]
        gi_sm = self.IaSH(gi_ftr) # [Bg*M, 1, 64, 64]
        gi_ftr_g = self.online_intra_saliency_guidance(DS(gi_sm, 4), gi_ftr).view(Bg, M, D, H//8, W//8) # [Bg, M, D, 16, 16]
        # block-wise group shuffling
        # re-organize a tensor "gi_ftr_g" into a tuple "shuffled_blocks"
        # "shuffled_blocks[sid] (sid=1,...,S)" is a tensor with dimension of [Bg, M*D/S, 16, 16]
        shuffled_blocks = torch.chunk(self.block_wise_shuffle(gi_ftr_g, S), S, dim=1) 
        # local & global aggregation
        bag = []
        for sid in range(S):
            blk = self.attentional_aggregation(shuffled_blocks[sid], M) # [Bg, D/S, 16, 16]
            blk = self.lca[sid](blk) # [Bg, D/S, 16, 16]
            blk = self.gca[sid](blk) # [Bg, D/S, 16, 16]
            bag.append(blk)
        concat_blk = torch.cat(bag, dim=1) # [Bg, D, 16, 16]
        gs = self.block_fusion(concat_blk) # [Bg, D, 16, 16]
        # gated group distribution
        X_0 = []
        for mid in range(M):
            gi_ftr_cosal = self.ggd(gs, gi_ftr_g[:, mid, :, :, :]) # [Bg, D, 16, 16]
            X_0.append(gi_ftr_cosal.unsqueeze(1)) # [Bg, 1, D, 16, 16]
        X_0 = torch.cat(X_0, dim=1) # [Bg, M, D, 16, 16]
        # feature decoding
        X_1 = self.gcpd_1(X_0) # [Bg, M, 512, 32, 32]
        X_2 = self.gcpd_2(X_1) # [Bg, M, 256, 64, 64]
        X_3 = self.gcpd_3(X_2) # [Bg, M, 256, 128, 128]
        csm = self.cosal_head(X_3) # [Bg, M, 1, 128, 128]
        if self.mode=='train' and self.compute_loss==True:
            cosod_loss = self.compute_cosod_loss(csm, gl)
            sod_loss = self.compute_sod_loss(si_sm, sl)
            return csm, cosod_loss, sod_loss
        else:
            return csm
    def block_wise_shuffle(self, gi_ftr_g, S):
        # gi_ftr_g: [Bg, M, D, fh, fw]
        Bg, M, D, fh, fw = gi_ftr_g.size()
        assert np.mod(D, S) == 0
        return gi_ftr_g.view(Bg, M, S, D//S, fh, fw).transpose(1, 2).contiguous().view(Bg, M, D, fh, fw).view(Bg, M*D, fh, fw)
    def attentional_aggregation(self, blk, M):
        # blk: [B, M*D/S, fh, fw]
        # blk_agg: # [B, D/S, fh, fw]
        blk = F.softmax(blk, dim=1) * blk # [B, M*D/S, fh, fw]
        blk_bag = torch.chunk(blk, M, dim=1) # blk_bag[mid]: [B, D/S, H, W], mid=1,...,M
        blk_agg = blk_bag[0] # [B, D/S, fh, fw]
        for mid in range(1, M):
            blk_agg = blk_agg + blk_bag[mid] # [B, D/S, fh, fw]
        return blk_agg
    def compute_cosod_loss(self, csm, gl):
        # csm: [Bg, M, 1, 128, 128]
        # gl: [Bg, M, 1, 128, 128]
        Bg, M, _, H, W = gl.size()
        cm = csm.view(Bg*M, 1, H, W) # [Bg*M, 1, H, W]
        gt = gl.view(Bg*M, 1, H, W) # [Bg*M, 1, H, W]
        return F.binary_cross_entropy(cm, gt)
    def compute_sod_loss(self, si_sm, sl):
        # si_sm: [Bs, 1, 64, 64]
        # sl: [Bs, 1, 128, 128]
        return F.binary_cross_entropy(si_sm, DS(sl, 2, 'max'))


class CoADNet_VGG16(nn.Module):
    def __init__(self, mode, compute_loss):
        super(CoADNet_VGG16, self).__init__()
        assert mode in ['train', 'test']
        self.mode = mode
        self.compute_loss = compute_loss
        self.M = the_batch_size
        self.D = 512
        self.S = 4
        assert np.mod(self.D, self.S) == 0
        self.backbone = Backbone_Wrapper_VGG16()
        self.IaSH = IaSH('VGG16')
        self.online_intra_saliency_guidance = OIaSG()
        self.lca = nn.ModuleList([LCA(self.D//self.S, 1, [1, 3, 5]) for s in range(self.S)])
        self.gca = nn.ModuleList([GCA(self.D//self.S, 1) for s in range(self.S)])
        self.block_fusion = BlockFusion(self.D, 2)
        self.ggd = GGD(self.D, 1)
        decode_dims = [self.D, 512, 256, 128]
        self.gcpd_1 = GCPD(decode_dims[0], decode_dims[1], 4)
        self.gcpd_2 = GCPD(decode_dims[1], decode_dims[2], 1)
        self.gcpd_3 = GCPD(decode_dims[2], decode_dims[3], 1)
        self.cosal_head = CoSH(decode_dims[3])
    def forward(self, gi, si=None, gl=None, sl=None):
        # gi: [Bg, M, 3, 128, 128]
        # si: [Bs, 3, 128, 128]
        # gl: [Bg, M, 1, 128, 128]
        # sl: [Bs, 1, 128, 128]
        M = self.M # number of groups
        D = self.D # backbone feature dimension
        S = self.S # number of blocks
        H, W = gi.size(3), gi.size(4)
        if self.mode == 'train':
            assert si is not None
            assert gl is not None
            assert sl is not None
            Bg = gi.size(0)
            Bs = si.size(0)
            si_ftr = self.backbone(si) # [Bs, 512, 16, 16]
            si_sm = self.IaSH(si_ftr) # [Bs, 1, 64, 64]
        if self.mode == 'test':
            assert si is None
            assert gl is None
            assert sl is None
            Bg = gi.size(0)
        # feature extrcation on gi
        gi_ftr = self.backbone(gi.view(-1, 3, H, W)) # [Bg*M, D, 16, 16]
        gi_sm = self.IaSH(gi_ftr) # [Bg*M, 1, 64, 64]
        gi_ftr_g = self.online_intra_saliency_guidance(DS(gi_sm, 4), gi_ftr).view(Bg, M, D, H//8, W//8) # [Bg, M, D, 16, 16]
        # block-wise group shuffling
        # re-organize a tensor "gi_ftr_g" into a tuple "shuffled_blocks"
        # "shuffled_blocks[sid] (sid=1,...,S)" is a tensor with dimension of [Bg, M*D/S, 16, 16]
        shuffled_blocks = torch.chunk(self.block_wise_shuffle(gi_ftr_g, S), S, dim=1) 
        # local & global aggregation
        bag = []
        for sid in range(S):
            blk = self.attentional_aggregation(shuffled_blocks[sid], M) # [Bg, D/S, 16, 16]
            blk = self.lca[sid](blk) # [Bg, D/S, 16, 16]
            blk = self.gca[sid](blk) # [Bg, D/S, 16, 16]
            bag.append(blk)
        concat_blk = torch.cat(bag, dim=1) # [Bg, D, 16, 16]
        gs = self.block_fusion(concat_blk) # [Bg, D, 16, 16]
        # gated group distribution
        X_0 = []
        for mid in range(M):
            gi_ftr_cosal = self.ggd(gs, gi_ftr_g[:, mid, :, :, :]) # [Bg, D, 16, 16]
            X_0.append(gi_ftr_cosal.unsqueeze(1)) # [Bg, 1, D, 16, 16]
        X_0 = torch.cat(X_0, dim=1) # [Bg, M, D, 16, 16]
        # feature decoding
        X_1 = self.gcpd_1(X_0) # [Bg, M, 512, 32, 32]
        X_2 = self.gcpd_2(X_1) # [Bg, M, 256, 64, 64]
        X_3 = self.gcpd_3(X_2) # [Bg, M, 256, 128, 128]
        csm = self.cosal_head(X_3) # [Bg, M, 1, 128, 128]
        if self.mode=='train' and self.compute_loss==True:
            cosod_loss = self.compute_cosod_loss(csm, gl)
            sod_loss = self.compute_sod_loss(si_sm, sl)
            return csm, cosod_loss, sod_loss
        else:
            return csm
    def block_wise_shuffle(self, gi_ftr_g, S):
        # gi_ftr_g: [Bg, M, D, fh, fw]
        Bg, M, D, fh, fw = gi_ftr_g.size()
        assert np.mod(D, S) == 0
        return gi_ftr_g.view(Bg, M, S, D//S, fh, fw).transpose(1, 2).contiguous().view(Bg, M, D, fh, fw).view(Bg, M*D, fh, fw)
    def attentional_aggregation(self, blk, M):
        # blk: [B, M*D/S, fh, fw]
        # blk_agg: # [B, D/S, fh, fw]
        blk = F.softmax(blk, dim=1) * blk # [B, M*D/S, fh, fw]
        blk_bag = torch.chunk(blk, M, dim=1) # blk_bag[mid]: [B, D/S, H, W], mid=1,...,M
        blk_agg = blk_bag[0] # [B, D/S, fh, fw]
        for mid in range(1, M):
            blk_agg = blk_agg + blk_bag[mid] # [B, D/S, fh, fw]
        return blk_agg
    def compute_cosod_loss(self, csm, gl):
        # csm: [Bg, M, 1, 128, 128]
        # gl: [Bg, M, 1, 128, 128]
        Bg, M, _, H, W = gl.size()
        cm = csm.view(Bg*M, 1, H, W) # [Bg*M, 1, H, W]
        gt = gl.view(Bg*M, 1, H, W) # [Bg*M, 1, H, W]
        return F.binary_cross_entropy(cm, gt)
    def compute_sod_loss(self, si_sm, sl):
        # si_sm: [Bs, 1, 64, 64]
        # sl: [Bs, 1, 128, 128]
        return F.binary_cross_entropy(si_sm, DS(sl, 2, 'max'))

    
class CoADNet_ResNet50(nn.Module):
    def __init__(self, mode, compute_loss):
        super(CoADNet_ResNet50, self).__init__()
        assert mode in ['train', 'test']
        self.mode = mode
        self.compute_loss = compute_loss
        self.M = the_batch_size
        self.D = 1024
        self.S = 8
        assert np.mod(self.D, self.S) == 0
        self.backbone = Backbone_Wrapper_Dilated_ResNet50()
        self.IaSH = IaSH('ResNet50')
        self.online_intra_saliency_guidance = OIaSG()
        self.lca = nn.ModuleList([LCA(self.D//self.S, 4, [1, 3, 5]) for s in range(self.S)])
        self.gca = nn.ModuleList([GCA(self.D//self.S, 2) for s in range(self.S)])
        self.block_fusion = BlockFusion(self.D, 8)
        self.ggd = GGD(self.D, 8)
        decode_dims = [self.D, 512, 256, 128]
        self.gcpd_1 = GCPD(decode_dims[0], decode_dims[1], 8)
        self.gcpd_2 = GCPD(decode_dims[1], decode_dims[2], 2)
        self.gcpd_3 = GCPD(decode_dims[2], decode_dims[3], 2)
        self.cosal_head = CoSH(decode_dims[3])
    def forward(self, gi, si=None, gl=None, sl=None):
        # gi: [Bg, M, 3, 128, 128]
        # si: [Bs, 3, 128, 128]
        # gl: [Bg, M, 1, 128, 128]
        # sl: [Bs, 1, 128, 128]
        M = self.M # number of groups
        D = self.D # backbone feature dimension
        S = self.S # number of blocks
        H, W = gi.size(3), gi.size(4)
        if self.mode == 'train':
            assert si is not None
            assert gl is not None
            assert sl is not None
            Bg = gi.size(0)
            Bs = si.size(0)
            si_ftr = self.backbone(si) # [Bs, 1024, 16, 16]
            si_sm = self.IaSH(si_ftr) # [Bs, 1, 64, 64]
        if self.mode == 'test':
            assert si is None
            assert gl is None
            assert sl is None
            Bg = gi.size(0)
        # feature extrcation on gi
        gi_ftr = self.backbone(gi.view(-1, 3, H, W)) # [Bg*M, D, 16, 16]
        gi_sm = self.IaSH(gi_ftr) # [Bg*M, 1, 64, 64]
        gi_ftr_g = self.online_intra_saliency_guidance(DS(gi_sm, 4), gi_ftr).view(Bg, M, D, H//8, W//8) # [Bg, M, D, 16, 16]
        # block-wise group shuffling
        # re-organize a tensor "gi_ftr_g" into a tuple "shuffled_blocks"
        # "shuffled_blocks[sid] (sid=1,...,S)" is a tensor with dimension of [Bg, M*D/S, 16, 16]
        shuffled_blocks = torch.chunk(self.block_wise_shuffle(gi_ftr_g, S), S, dim=1) 
        # local & global aggregation
        bag = []
        for sid in range(S):
            blk = self.attentional_aggregation(shuffled_blocks[sid], M) # [Bg, D/S, 16, 16]
            blk = self.lca[sid](blk) # [Bg, D/S, 16, 16]
            blk = self.gca[sid](blk) # [Bg, D/S, 16, 16]
            bag.append(blk)
        concat_blk = torch.cat(bag, dim=1) # [Bg, D, 16, 16]
        gs = self.block_fusion(concat_blk) # [Bg, D, 16, 16]
        # gated group distribution
        X_0 = []
        for mid in range(M):
            gi_ftr_cosal = self.ggd(gs, gi_ftr_g[:, mid, :, :, :]) # [Bg, D, 16, 16]
            X_0.append(gi_ftr_cosal.unsqueeze(1)) # [Bg, 1, D, 16, 16]
        X_0 = torch.cat(X_0, dim=1) # [Bg, M, D, 16, 16]
        # feature decoding
        X_1 = self.gcpd_1(X_0) # [Bg, M, 512, 32, 32]
        X_2 = self.gcpd_2(X_1) # [Bg, M, 256, 64, 64]
        X_3 = self.gcpd_3(X_2) # [Bg, M, 256, 128, 128]
        csm = self.cosal_head(X_3) # [Bg, M, 1, 128, 128]
        if self.mode=='train' and self.compute_loss==True:
            cosod_loss = self.compute_cosod_loss(csm, gl)
            sod_loss = self.compute_sod_loss(si_sm, sl)
            return csm, cosod_loss, sod_loss
        else:
            return csm
    def block_wise_shuffle(self, gi_ftr_g, S):
        # gi_ftr_g: [Bg, M, D, fh, fw]
        Bg, M, D, fh, fw = gi_ftr_g.size()
        assert np.mod(D, S) == 0
        return gi_ftr_g.view(Bg, M, S, D//S, fh, fw).transpose(1, 2).contiguous().view(Bg, M, D, fh, fw).view(Bg, M*D, fh, fw)
    def attentional_aggregation(self, blk, M):
        # blk: [B, M*D/S, fh, fw]
        # blk_agg: # [B, D/S, fh, fw]
        blk = F.softmax(blk, dim=1) * blk # [B, M*D/S, fh, fw]
        blk_bag = torch.chunk(blk, M, dim=1) # blk_bag[mid]: [B, D/S, H, W], mid=1,...,M
        blk_agg = blk_bag[0] # [B, D/S, fh, fw]
        for mid in range(1, M):
            blk_agg = blk_agg + blk_bag[mid] # [B, D/S, fh, fw]
        return blk_agg
    def compute_cosod_loss(self, csm, gl):
        # csm: [Bg, M, 1, 128, 128]
        # gl: [Bg, M, 1, 128, 128]
        Bg, M, _, H, W = gl.size()
        cm = csm.view(Bg*M, 1, H, W) # [Bg*M, 1, H, W]
        gt = gl.view(Bg*M, 1, H, W) # [Bg*M, 1, H, W]
        return F.binary_cross_entropy(cm, gt)
    def compute_sod_loss(self, si_sm, sl):
        # si_sm: [Bs, 1, 64, 64]
        # sl: [Bs, 1, 128, 128]
        return F.binary_cross_entropy(si_sm, DS(sl, 2, 'max'))
