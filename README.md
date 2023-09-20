> Benchmark for the **inference speed** test of existing CoSOD models.

### Settings:

+ Batch size: 2
+ Image size: 256x256
+ GPU: A100

### Models:

This benchmark currently has models below (sorted by alphabet):

1. [CADC](https://github.com/nnizhang/CADC)     (ICCV 2021)
2. [CoADNet](https://github.com/KeeganZQJ/CoSOD-CoADNet)  (NeurIPS 2020)
3. [DCFM](https://github.com/siyueyu/DCFM)     (CVPR 2022)
4. [GCAGC](https://github.com/ltp1995/GCAGC-CVPR2020)    (CVPR 2020)
5. [GCoNet](https://github.com/fanq15/GCoNet)   (CVPR 2021)
6. [GCoNet+](https://github.com/ZhengPeng7/GCoNet_plus)  (T-PAMI 2023)
7. [ICNet](https://github.com/blanclist/ICNet)    (NeurIPS 2020)
8. [MCCL](https://github.com/ZhengPeng7/MCCL)     (AAAI 2023)

This repo tries to contain the codes only related to the inference to make it as simple as possible.
For the consistent settings, we modify some parts of these codes like the fixed image size and channels in network, with no unnecessary influence on their performance.

All of these codes are executed in Python=3.8 on an A100 GPU, with libs below. We choose PyTorch 1.13.1 to avoid the possible API differences in PyTorch 2.x in the future.

`conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y && pip install -r req.txt`.

If you find this repo useful, you can cite our GCoNet+ paper, where the inference speed benchmark is included:
```
@article{zheng2022gconet+,
  author={Zheng, Peng and Fu, Huazhu and Fan, Deng-Ping and Fan, Qi and Qin, Jie and Tai, Yu-Wing and Tang, Chi-Keung and Van Gool, Luc},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={GCoNet+: A Stronger Group Collaborative Co-Salient Object Detector}, 
  year={2023},
  volume={45},
  number={9},
  pages={10929-10946},
  doi={10.1109/TPAMI.2023.3264571}
}
```
