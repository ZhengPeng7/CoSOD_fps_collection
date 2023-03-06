from time import time
import torch

def test_fps(model, size=256, batch_size=2):
    # Init model

    # print('111!!!!')
    model.cuda()
    model.eval()

    N = 2
    time_total = 0.
    buf_iter = -500
    # print('!!!!')
    with torch.no_grad():
        for i in range(1+buf_iter, 1000+1):
            # print(i, end=', ')
            inputs = torch.randn(batch_size, 3, size, size).float().cuda()
            time_st = time()
            _ = model(inputs)
            if i > 0:
                time_latest = time() - time_st
                time_total += time_latest
                if i % 300 == 0:
                    print(i, 'time_avg: {:.4f}s, time_curr: {:.4f}s.'.format(time_total / i / N, time_latest / N))
    return time_total / i / N
