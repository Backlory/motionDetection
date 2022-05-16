import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.thirdparty_RAFT.core.update import BasicUpdateBlock
from model.thirdparty_RAFT.core.corr import MaskCorrBlock
from model.thirdparty_RAFT.core.utils.utils import upflow8
from model.thirdparty_RAFT.core.raft import RAFT
from utils.timers import tic, toc

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class Mask_RAFT(RAFT):
    def __init__(self, gridLength = 32, args = {
        "RAFT_model":"model/thirdparty_RAFT/model/raft-sintel.pth",
        "RAFT_path":r"E:\dataset\dataset-fg-det\Janus_UAV_Dataset\Train\video_1\video",
        "RAFT_small":"store_true",
        "RAFT_mixed_precision":"store_false",
        "RAFT_alternate_corr":"store_true"
        }):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model',default=args["RAFT_model"], help="restore checkpoint")  #things
        parser.add_argument('--path', default=args["RAFT_path"], help="dataset for evaluation")
        parser.add_argument('--small', action=args["RAFT_small"], help='use small model')
        parser.add_argument('--mixed_precision', action=args["RAFT_mixed_precision"], help='use mixed precision')
        parser.add_argument('--alternate_corr', action=args["RAFT_alternate_corr"], help='use efficent correlation implementation')
        args = parser.parse_args()
        
        super().__init__(args)

        
        hdim = self.hidden_dim
        cdim = self.context_dim
        
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        
        status = torch.load(args.model)
        status = {k[7:]:v for k,v in status.items()}
        self.load_state_dict(status)
        print("weights have been loaded for MASK-RAFT...")

        #self.Maskfnet = MaskBasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
        #self.Maskfnet.load_state_dict(self.fnet.state_dict())
        #self.Maskcnet = MaskBasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        #self.Maskcnet.load_state_dict(self.cnet.state_dict())
        #self.Maskupdate_block = MaskBasicUpdateBlock(self.args, hidden_dim=hdim)
        

        self.gridLength = gridLength // 8    #因为三次下采样

    def forward(self, image1, image2, Masks=[None, None], iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        assert(len(image1) == 1)
        assert(len(image2) == 1)
        #tp = tic()
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim
        gL = self.gridLength
        
        with autocast(enabled=self.args.mixed_precision):
            fmap1 = self.fnet(image1)          #h/w各下降1/8，通道256  
            fmap2 = self.fnet(image2)          #h/w各下降1/8，通道256  
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        
        # Mask
        Mask_small, Mask_small_2 = Masks
        Mask_small = Mask_small / 255.0
        Mask_small_2 = Mask_small_2 / 255.0
        Mask_big = F.interpolate(Mask_small, fmap1.shape[2:4], mode="nearest")
        Mask_big_2 = F.interpolate(Mask_small_2, fmap1.shape[2:4], mode="nearest")
        Masks = [Mask_big, Mask_big_2]
        _, _, h1, w1 = fmap1.shape
        h_num = h1 // gL
        w_num = w1 // gL
        fmap1 = fmap1 * Mask_big
        fmap2 = fmap2 * Mask_big_2
        
        # 计算corr
        corr_fn = MaskCorrBlock(fmap1, fmap2, 
                                radius=self.args.corr_radius, 
                                Masks=Masks,
                                gridLength=gL )
        
        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)    #上下文网络。只对patch用
            cnet = cnet * Mask_big
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)   #正面进，初始状态   [1, 128, 64, 64])
            inp = torch.relu(inp)   #侧面进

        # 光流初始化
        coords0, coords1 = self.initialize_flow(image1) #某点的原始坐标， 某点运动后的坐标，尺寸为[n,2,h//8,w//8]
        if flow_init is not None:
            coords1 = coords1 + flow_init
        
        #toc(tp, "iters init", mute=False); 
        
        tp = tic()
        up_mask = torch.zeros([h_num, w_num, 576,gL,gL], device=coords0.device)
        for itr in range(iters):
            
            corr = corr_fn.__call_mask__(coords1) # [20, 20, 1, 324, 4, 4]
            
            #coords1 = coords1.reshape([2, h_num, gL, w_num, gL]).permute([1,3,0,2,4])
            #corr = corr_fn(coords1) # [20, 20, 1, 324, 4, 4]
            #coords1 = coords1.permute([2,0,3,1,4]).contiguous().view(1,2,h1,w1)
            #corr = corr.permute([2,3,0,4,1,5]).contiguous().view(1,324,h1,w1)
            
            

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 += delta_flow
            coords1 = Mask_big * coords1 + (1-Mask_big) * coords0
        
        #toc(tp, "iters", mute=False); tp = tic()
        
        flow_up = self.upsample_flow(coords1 - coords0, up_mask)
        #toc(tp, "upsample_flow", mute=False); tp = tic()
        

        return coords1 - coords0, flow_up, coords0, coords1, fmap1

        # fmap1_patches = []
        # fmap2_patches = []
        # fea_Position = []
        # fmap1_pad = F.pad(fmap1, [3,3,3,3])
        # fmap2_pad = F.pad(fmap1, [3,3,3,3])
        # for i in range(self.gridEdgeNum):
        #     for j in range(self.gridEdgeNum):
        #         if Mask[0, 0, i, j] > 0:
        #             y = i * self.gridLength
        #             x = j * self.gridLength
        #             fmap1_patches.append(fmap1_pad[:, :, y:y+self.gridLength+6, x:x+self.gridLength+6])
        #             fmap2_patches.append(fmap2_pad[:, :, y:y+self.gridLength+6, x:x+self.gridLength+6])
        #             fea_Position.append([i,j])
        # if len(fmap1_patches) == 0:
        #     return
        # else:
        #     fmap1_patches = torch.cat(fmap1_patches)
        #     fmap2_patches = torch.cat(fmap2_patches)