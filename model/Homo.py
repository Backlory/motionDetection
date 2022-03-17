import torch
import torch.nn as nn
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
class Block(nn.Module):
    def __init__(self, inchannels, midchannels, outchannels, batch_norm=False, pool=True):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, midchannels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(midchannels)
        else:
            self.bn1 = IdentityBlock()
        self.conv2 = nn.Conv2d(midchannels, outchannels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.bn2 =nn.BatchNorm2d(outchannels)
        else:
            self.bn2 = IdentityBlock()
        if pool:
            self.pool = nn.MaxPool2d(2, 2)
        else:
            self.pool = IdentityBlock()
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.pool(out)
        return out

class HomographyNet(nn.Module):
    def __init__(self, batch_norm=True):
        super(HomographyNet, self).__init__()
        self.cnn = nn.Sequential(
            Block(2, 64, 64, batch_norm),
            Block(64, 64, 64, batch_norm),
            Block(64, 128, 128, batch_norm),
            Block(128, 128, 128, batch_norm, pool=False),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128 * 16 * 16, 160),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(160, 4 * 2),
            #nn.Linear(128*16*16, 4 * 2),
        )

    def forward(self, a, b):
        x = torch.cat((a, b), dim=1)  # combine two images in channel dimension
        x = self.cnn(x)
        x = self.fc(x)
        delta = x.view(-1, 4, 2)
        return delta
        

if __name__ == "__main__":
    device = 'cuda'  # cpu, cuda
    def get_sample():
        img_t0 = torch.rand(8, 1, 128, 128).to(device)
        img_t1 = torch.rand(8, 1, 128, 128).to(device)
        return img_t0, img_t1

    model = HomographyNet().to(device)
    summary(model, [(1, 128, 128),(1, 128, 128)])
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
        print(idx, item,'======', list(state_dict[item].shape))
    
    pytorch_model = HomographyNet().to(device)
    pytorch_model.load_state_dict(torch.load("temp_weight.pt"))
    
    # 保存
    img_t0, img_t1 = get_sample()
    traced_model = torch.jit.trace(model, (img_t0, img_t1))
    traced_model.save('temp_model_trace.pt')
    script_model = torch.jit.script(model)
    script_model.save('temp_model_script.pt')

    # 重新加载模型
    img_t0, img_t1 = get_sample()
    pytorch_model = HomographyNet()
    pytorch_model.load_state_dict(torch.load("temp_weight.pt"))
    traced_model = torch.jit.load('temp_model_trace.pt')
    script_model = torch.jit.load('temp_model_script.pt')

    # 输出验证
    for i in range(5):
        img_t0, img_t1 = get_sample()
        output1 = model(img_t0, img_t1)
        output2 = traced_model(img_t0.clone(), img_t1.clone())
        output3 = script_model(img_t0.clone(), img_t1.clone())
        if isinstance(output1, tuple):
            output1 = output1[0]
            output2 = output2[0]
            output3 = output3[0]
        print(output1.flatten()[0:5])
        print(output2.flatten()[0:5])
        print(output3.flatten()[0:5])
        for idx in range(len(output1)):
            assert((output1[idx] / output2[idx]<1.05).all())
            assert((0.95<output1[idx] / output2[idx]).all())
            assert((output1[idx] / output3[idx]<1.05).all())
            assert((0.95<output1[idx] / output3[idx]).all())
            print("consistent check pass!")