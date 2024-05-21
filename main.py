import argparse

from MCCL.models.GCoNet import MCCL
from GCoNet_plus.models.GCoNet import GCoNet_plus
from GCoNet.models.GCoNet import GCoNet
from DCFM.models.main import DCFM
from CoSOD_CoADNet.code.network import CoADNet_Dilated_ResNet50 as CoADNet
from CADC.CoSODNet.CoSODNet import CoSODNet as CADC
from gicd.models.GICD import GICD   # Unsolved
from ICNet.ICNet.network import ICNet
from GCAGC_CVPR2020.model3.model2_graph4_hrnet_sal import Model2 as GCAGC
from test_fps import test_fps, BATCH_SIZE



def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--models', default='GCoNet_plus', type=str, help=".")
    parser.add_argument('--data', default='CoCA', type=str, help=".")
    args = parser.parse_args()

    repeat_time = 1    # > 1 may lead to reading cache for faster running, which is not fair.
    for data in args.data.split(','):
        for model_name in args.models.split(','):
            real_batch_size = BATCH_SIZE if data == 'random' else 'all'     # Only work when data == 'random'.
            if data == 'CoCA' and model_name not in ['CoADNet', 'GCAGC']:
                print('Model {} on data {} cannot deal with data with batch size {}. Please Change the BATCH_SIZE in `test_fps.py` to 5 to have a test on them for CoCA.'.format(model_name, data, real_batch_size))
                continue
            model = eval(model_name+'()')
            time_per_frame_lst = []
            for _ in range(repeat_time):
                time_per_frame = test_fps(model, model_name, size=256, data=data, batch_size=BATCH_SIZE)
                time_per_frame_lst.append(time_per_frame)
            time_per_frame = sum(time_per_frame_lst) / repeat_time
            print('Model {} on data {} with batch size {}, running time {:.6f} s, FPS = {:.4f}.'.format(model_name, data, real_batch_size, time_per_frame, 1 / time_per_frame))
    return sum(time_per_frame_lst) / repeat_time


if __name__ == '__main__':
    main()
