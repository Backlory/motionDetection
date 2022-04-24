import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
try:
    from utils.timers import tic, toc
except:
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    from utils.timers import tic, toc
class IdentityBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return x

class ASPPBN(nn.Module):
    def __init__(self, in_planes=512,out_planes=256):
        super(ASPPBN, self).__init__()
 
        self.conv_1x1_1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_planes)
 
        self.conv_3x3_1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_planes)
 
        self.conv_3x3_2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_planes)
 
        self.conv_3x3_3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_planes)
 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        self.conv_1x1_2 = nn.Conv2d(in_planes, out_planes, kernel_size=1,stride=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_planes)
 
        self.conv_1x1_3 = nn.Conv2d(out_planes*5, out_planes, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_planes)
 
 
    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)
 
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
 
        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))
 
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3,out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
 
        return out

class MovingDetectNet(nn.Module):
    def __init__(self, batch_norm=True):
        super(MovingDetectNet, self).__init__()
        self.cnn0 = nn.Conv2d(3,3,3,stride=1, padding=1)
        self.cnn1 = nn.Conv2d(6,1,3,stride=1, padding=1)
    
    def clear(self):
        return

    def forward(self, img_t0, img_t1):
        img_t0 = self.cnn0(img_t0)
        out = torch.concat([img_t0, img_t1], axis=1)
        out = self.cnn1(out)
        return out
        

if __name__ == "__main__":
    device = 'cuda'  # cpu, cuda
    def get_sample():
        img_t0 = torch.rand(8, 1, 128, 128).to(device)
        img_t1 = torch.rand(8, 1, 128, 128).to(device)
        return img_t0, img_t1

    model = MovingDetectNet().to(device)
    try:
        summary(model, [(1, 128, 128),(1, 128, 128)])
    except:
        pass
    model.train()
    import torch.optim as opt
    optimizer = opt.Adam(model.parameters(), lr=0.1)
    img_t0, img_t1 = get_sample()
    for i in range(10):
        optimizer.zero_grad()
        #
        torch.cuda.synchronize()
        t = tic()
        output = model(img_t0, img_t1)
        torch.cuda.synchronize()
        toc(t,'inference', mute=False)
        #
        a = [torch.sum(x.flatten()) for x in output]
        a = sum(a)
        loss = torch.sum(a*2)
        print(loss)
        loss.backward()
        optimizer.step()
        
    model.eval()
    state_dict = model.state_dict()
    torch.save(state_dict, "temp_weight.pt")
    for idx, item in enumerate(list(state_dict.keys())):
        pass
        print(idx, item,'======', list(state_dict[item].shape))