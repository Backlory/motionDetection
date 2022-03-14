import torch
import torch.nn as nn

class HomographyNet(nn.Module):
    def __init__(self) -> None:
        self.conv = nn.Conv2d(3, )
        super().__init__()
    def forward(self, x):
        return x

if __name__ == "__main__":
    model = HomographyNet()
