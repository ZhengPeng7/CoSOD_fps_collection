import os

from test_fps import test_fps

from MCCL.models.GCoNet import MCCL
from GCoNet_plus.models.GCoNet import GCoNet_plus
from GCoNet.models.GCoNet import GCoNet
from DCFM.models.main import DCFM
from CoSOD_CoADNet.code.network import CoADNet_Dilated_ResNet50 as CoADNet
from CADC.CoSODNet.CoSODNet import CoSODNet as CADC
from gicd.models.GICD import GICD   # Unsolved
from ICNet.ICNet.network import ICNet
from GCAGC_CVPR2020.model3.model2_graph4_hrnet_sal import Model2 as GCAGC


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model_lst = ['MCCL', 'GCoNet_plus', 'GCoNet', 'DCFM', 'CoADNet', 'CADC', 'ICNet', 'GCAGC']
for model in model_lst[:2]:
    m = eval(model+'()')
    time_per_frame = test_fps(m)
    print('Model {}, running time {:.4f} s, FPS = {}.'.format(model, time_per_frame, 1 / time_per_frame))
