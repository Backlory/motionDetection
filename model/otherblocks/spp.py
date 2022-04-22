import torch
import torch.nn as nn

class SPPBlock(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13), e=1.0):
        super(SPPBlock, self).__init__()
        c_ = c1 // 2
        self.cbl_before = nn.Sequential(
            nn.Conv2d(c1, c_, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU())
        self.max_pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        self.cbl_after = nn.Sequential(
            nn.Conv2d(c_ * 4, c2, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU())

    def forward(self, x):
        x = self.cbl_before(x)
        out1 = self.max_pool1(x)
        out2 = self.max_pool2(x)
        out3 = self.max_pool3(x)
        out = torch.cat([x, out1, out2, out3], 1)
        out = self.cbl_after(out)
        return out

if __name__ == "__main__":
    model = SPPBlock(128, 128)
    input = torch.rand([8,128,40,40])
    out = model(input)
    print(out.shape)