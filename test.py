import torch
import torch.nn as nn
from utils.timers import toc, tic
from torch.cuda import synchronize

conv1 = nn.Conv2d(3, 64, 3, 1, 1)
bn1 = nn.BatchNorm2d(64)
relu1 = nn.ReLU()
conv2 = nn.Conv2d(64, 256, 3, 1, 1)
bn2 = nn.BatchNorm2d(256)
relu2 = nn.ReLU()

def test(img1, words, repeattimes):
    out = conv1(img1)
    out = bn1(out)
    out = relu1(out)
    out = conv2(out)
    out = bn2(out)
    out = relu2(out)
    t=tic()
    synchronize()
    for i in range(repeattimes):
        out = conv1(img1)
        out = bn1(out)
        out = relu1(out)
        out = conv2(out)
        out = bn2(out)
        out = relu2(out)
    synchronize()
    toc(t, word=words, act_number=repeattimes, mute=False)


img1 = torch.rand([8,3,64,64])
img2 = torch.zeros([8,3,64,64])
img3 = img1.clone()
img3[:,:,:,0:32] = 0

test(img1, "cbr-cbr-rand", 100)
test(img2, "cbr-cbr-zeros", 100)
test(img3, "cbr-cbr-rz", 100)