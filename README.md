Benckmark for the inference speed test of CoSOD models.
This benchmark currently has models below (sorted by aplhabet):
1. CADC     (ICCV 2021)
2. CoADNet  (NeurIPS 2020)
3. DCFM     (CVPR 2022)
4. GCAGC    (CVPR 2020)
5. GCoNet   (CVPR 2021)
6. GCoNet+  (arXiv 2021)
7. ICNet    (NeurIPS 2020)
8. MCCL     (AAAI 2023)

This repo tries to contain the codes only related to the inference to make it as simple as possible.
For the consistent settings, we modify some parts of these codes like the fixed image size and channels in network, with no unnecessary influence on their performance.
All of these codes are executed in Python=3.8, PyTorch==1.13.1 on one A100 GPU.
