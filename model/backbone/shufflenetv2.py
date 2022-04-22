import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


def channel_shuffle(x:torch.Tensor, groups:int):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

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

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
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

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        model_size="1.0x",
        with_last_conv=False,
        kernal_size=3,
        pretrain=True,
    ):
        super(ShuffleNetV2, self).__init__()
        
        print("model size is ", model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        self.with_last_conv = with_last_conv
        self.kernal_size = kernal_size
        if model_size == "0.5x":
            self._stage_out_channels = [24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            self._stage_out_channels = [24, 116, 232, 464, 1024]
        elif model_size == "1.5x":
            self._stage_out_channels = [24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            self._stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #各个stage
        
        input_channels = output_channels
        output_channels = self._stage_out_channels[1]
        self.stage2 = nn.Sequential(
            ShuffleV2Block(input_channels, output_channels, 2),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
        )
        input_channels = output_channels
        output_channels = self._stage_out_channels[2]
        self.stage3 = nn.Sequential(
            ShuffleV2Block(input_channels, output_channels, 2),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
        )
        input_channels = output_channels
        output_channels = self._stage_out_channels[3]
        self.stage4 = nn.Sequential(
            ShuffleV2Block(input_channels, output_channels, 2),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
        )
        

        '''stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, self.stage_repeats, self._stage_out_channels[1:]):
            seq = [ShuffleV2Block(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(ShuffleV2Block(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels'''
        
        output_channels = self._stage_out_channels[-1]
        self._initialize_weights(pretrain)

    def forward(self, x):
        out1 = self.conv1(x)
        #out1 = self.maxpool(out1)
        '''output = []
        
        for i in range(2, 5):
            stage = getattr(self, "stage{}".format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)'''
        #
        out2 = self.stage2(out1)
        out3 = self.stage3(out2)
        out4 = self.stage4(out3)
        return (out1, out2, out3, out4)

    def _initialize_weights(self, pretrain=True):
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
        if pretrain:
            url = r"weights\shufflenetv2_x1-5666bf0f80.pth"
            pretrained_state_dict = torch.load(url)
            if False:
                for k,v in self.state_dict().items():
                    if (k in pretrained_state_dict.keys()):
                        pass
                        print(pretrained_state_dict[k].shape ==v.shape)
                    else:
                        print((k in pretrained_state_dict.keys()), "==", k,"=>",v.shape)
            if True:
                for k,v in pretrained_state_dict.items():
                    if (k in self.state_dict().keys()):
                        pass
                        print(self.state_dict()[k].shape ==v.shape)
                    else:
                        print((k in self.state_dict().keys()), "==", k,"=>",v.shape)

            self.load_state_dict(pretrained_state_dict, strict=False)
            

if __name__ == "__main__":
    device = torch.device('cuda:0')
    backbone = ShuffleNetV2(pretrain=True).to(device)
    
    img = cv2.imread("../data/Janus_UAV_Dataset/Train/video_1/video/067.png")
    img = img.astype(np.float32) / 255
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3) / 255
    std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3) / 255
    img = (img - mean) / std
    #运行时间测试
    
    with torch.no_grad(): 
        img_ten = torch.from_numpy(img.transpose(2, 0, 1))[None].to(device)    
        out = backbone(img_ten)
        torch.cuda.synchronize()
        time1 = time.time()
        for i in range(500):
            out = backbone(img_ten)
            torch.cuda.synchronize()
        temp = (time.time() - time1) * 1000 / 500
        print("viz time: {:.3f} ms".format(temp))
        print(1)
    #
    # 保存    
    with torch.no_grad():  
        torch.cuda.empty_cache()
        def get_sample():
            return torch.rand(8, 3, 512, 512).to(device)
        model = ShuffleNetV2(pretrain=True).to(device)  
        model.eval()
        state_dict = model.state_dict()
        torch.save(state_dict, "temp/temp_weight.pt")
        img_t0 = get_sample()
        print("trace")
        traced_model = torch.jit.trace(model, img_t0)
        traced_model.save('temp/temp_model_trace.pt')
        print("script")
        script_model = torch.jit.script(model)
        script_model.save('temp/temp_model_script.pt')

        # 重新加载模型
        img_t0 = get_sample()
        pytorch_model = ShuffleNetV2(pretrain=False).to(device)
        pytorch_model.load_state_dict(torch.load("temp/temp_weight.pt"))
        traced_model = torch.jit.load('temp/temp_model_trace.pt')
        script_model = torch.jit.load('temp/temp_model_script.pt')

        # 输出验证
        for i in range(5):
            img_t0 = get_sample()
            output1 = model(img_t0)
            output2 = traced_model(img_t0.clone())
            output3 = script_model(img_t0.clone())
            if isinstance(output1, tuple):
                output1 = output1[0]
                output2 = output2[0]
                output3 = output3[0]
            for idx in range(len(output1)):
                assert((output1[idx] / output2[idx]<1.05).all())
                assert((0.95<output1[idx] / output2[idx]).all())
                assert((output1[idx] / output3[idx]<1.05).all())
                assert((0.95<output1[idx] / output3[idx]).all())
                #print("consistent check pass!")