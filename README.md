> Benchmark for the **inference speed** test of existing CoSOD models.

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

All of these codes are executed in Python=3.8 on one A100 GPU, with libs below. We choose PyTorch 1.13.1 since it is the last version of PyTorch 1.X.

`conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y && pip install -r req.txt`.
