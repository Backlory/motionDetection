from turtle import forward
import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.optim as optim
import os
import sys

from utils.img_display import img_square
from model.backbone.shufflenetv2 import ShuffleNetV2
from model.Corr.Corr import CorrBlock
# 模型

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        #assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv( inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.LeakyReLU(negative_slope=0.1, inplace=True))
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(branch_features),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.depthwise_conv(branch_features,branch_features,kernel_size=3,stride=self.stride,padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features,branch_features,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(branch_features),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        
    def depthwise_conv(self, i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out1 = self.branch1(x1)
            out2 = self.branch2(x2)
        else:
            out1 = self.branch1(x)
            out2 = self.branch2(x)
        out = torch.cat((out1, out2), dim=1)
        
        out = self.channel_shuffle(out, 2)

        return out
    def channel_shuffle(self, x:torch.Tensor, groups:int):
        # type: (torch.Tensor, int) -> torch.Tensor
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

class testmodel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = ShuffleNetV2()

        self.corrBlock = CorrBlock()
        
        self.outlayer = nn.Conv2d(49, 2, 1, 1, 0, bias=False)
        self.softmax = nn.Softmax(1)

    def _initialize_weights(self):
        print("init weights...")
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "first" in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)

    
    def forward(self, img_t1, img_t2):
        
        img_t1_out1, img_t1_out2, img_t1_out3 = self.backbone(img_t1)
        img_t2_out1, img_t2_out2, img_t2_out3 = self.backbone(img_t2)

        in1 = img_t1_out3
        in2 = img_t2_out3
        radius = 5
        out_cos, out_diss = self.corrBlock(in1, in2, radius)
        n,c,h,w = out_cos.shape
        k = int(img_t1.shape[2] / h)
        temp1 = out_cos.view(n, radius, radius, h, w)
        temp2 = out_diss.view(n, radius, radius, h, w)
        cv2.namedWindow("1", cv2.WINDOW_FREERATIO)
        cv2.resizeWindow("1", 400, 200)
        for i in range(int(h*0.3), int(h*0.9),int(max(h*0.01, 1))):
            for j in range(int(w*0.3), int(w*0.9),int(max(w*0.01, 1))):
                watcher = []
                print(i, j)
                watcher.append(img_t1[0, :, k*(i-radius//2):k*(i+radius//2), k*(j-radius//2):k*(j+radius//2)].clone())
                watcher.append(temp1[0:1,:,:,i,j].clone())
                watcher.append(temp2[0:1,:,:,i,j].clone())
                temp = img_square(watcher, 1, 3)
                #temp = cv2.resize(temp, (400,200), cv2.INTER_NEAREST)
                cv2.imshow("1", temp)
                cv2.waitKey(1)

        out = self.outlayer(corr)
        out = self.softmax(out)
        return out

# 数据
def loadimg(filedir):
    img = cv2.imread(filedir)
    img = cv2.resize(img, (512, 512))
    #img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.medianBlur(img, 3)
    input = torch.Tensor(img.transpose(2,0,1) / 255).float()[None].contiguous()
    return input

def ten2bboxs(ten):
    assert{False}
    assert(ten.shape[1] == 2)   #[n, 2, h, w]
    bboxes = ten.detach().cpu().numpy()
    return bboxes

def vis_bbox(img, bboxes):
    h_img,w_img,_ = img.shape
    for bbox in bboxes:
        x,y,w,h = bbox
        w = w * w_img
        h = h * h_img
        x = x * w_img - w/2
        y = y * h_img - h/2
        
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    cv2.imshow("1", img)
    cv2.waitKey(0)
    return img

# loss
class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        assert(input.shape == targets.shape)
        input = input[:,0,:,:]
        targets = targets[:,0,:,:]
        N = targets.size()[0]
        smooth = 1

        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 交集
        intersection = input_flat * targets_flat 
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss
#======================================================================
def main():
    device = torch.device("cuda:0")
    print("model...")
    model = testmodel().to(device)
    print("criterion...")
    criterion = BinaryDiceLoss()
    print("optimizer...")
    optimizer = optim.Adam(model.parameters(), 1e-4)
    try:
        assert(device.type == "cuda")
        from torch.cuda.amp import GradScaler, autocast
    except:
        print("no cuda device available, so amp will be forbidden.")
        from utils.cuda import GradScaler, autocast
    print("scaler...")
    scaler = GradScaler(enabled=True)
    print("train...")
    if True:
        len_all = len(os.listdir("../data/Janus_UAV_Dataset/Train/video_1/video/"))
        for i in range(len_all * 100):
            i = i % len_all
            img_t1 = loadimg(f"../data/Janus_UAV_Dataset/Train/video_1/video/{str(i).zfill(3)}.png").to(device)
            img_t2 = loadimg(f"../data/Janus_UAV_Dataset/Train/video_1/video/{str(i+1).zfill(3)}.png").to(device)
            
            gt = cv2.imread(f"../data/Janus_UAV_Dataset/Train/video_1/gt_mov/{str(i).zfill(3)}.png", cv2.IMREAD_GRAYSCALE) 
            gt = 255 - cv2.resize(gt, (128, 128))
            _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
            target = cv2.merge([gt, 255-gt])
            target = torch.Tensor(target.transpose(2,0,1) / 255).float()[None].contiguous().to(device)
            
            output = model(img_t1, img_t2)
            loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            print(i ,"== loss:", "{:.3f}".format(loss.item()))
        
            watcher = [img_t1[0], img_t2[0], gt, output[0,0], None]
            cv2.imwrite("1.png", img_square(watcher, 2, 2))
        print(output)

    return

if __name__ == "__main__":
    main()
