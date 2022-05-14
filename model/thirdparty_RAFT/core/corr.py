import torch
import torch.nn.functional as F
from model.thirdparty_RAFT.core.utils.utils import bilinear_sampler, coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):    #每层都要取
            corr = self.corr_pyramid[i]     #n个点，单通道，与其余64*64个点的corr值
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device) #一个网格，记录了一个方形区域内每个点相对中心的偏移量
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i  #索引值随下采样倍数而降低
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl   # 广播运算，[4096, 1, 1, 2]的索引区，加[1, 9, 9, 2]代表每个点都要取9*9=81个子向量

            corr = bilinear_sampler(corr, coords_lvl)   #根据coords_lvl，对corr矩阵做双线性采样，得到的采样结果替换corr
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd)
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)  #第一个htwd是t0时刻的，第二个是t1时刻的
        return corr  / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())

class MaskCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, Masks=[None, None], gridLength=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.gridLength = gridLength
        
        # Mask
        self.Mask_big, self.Mask_big_2 = Masks
        #fmap1 = fmap1 * self.Mask_big
        #fmap2 = fmap2 * self.Mask_big_2
        
        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape #h1 w1代表原图中位置，h2w2代表t1时刻
        
        assert(batch == 1)
        h_num = h1 // gridLength
        w_num = w1 // gridLength
        self.h_num = h_num
        self.w_num = w_num
        
        corr = corr.reshape(h1*w1, dim, h2, w2)
        
        # pyramid
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)
        
        
    def __call_mask__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):    #每层都要取
            corr = self.corr_pyramid[i]     #n个点，单通道，与其余64*64个点的corr值
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device) #一个网格，记录了一个方形区域内每个点相对中心的偏移量
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
            
            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i  #索引值随下采样倍数而降低
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl   # 广播运算，[4096, 1, 1, 2]的索引区，加[1, 9, 9, 2]代表每个点都要取9*9=81个子向量
            corr = bilinear_sampler(corr, coords_lvl)   #根据coords_lvl，对corr矩阵做双线性采样，得到的采样结果替换corr
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    def __call__(self, coords):
        r = self.radius
        h_num = self.h_num
        w_num = self.w_num
        gridLength = self.gridLength
        h1 = h_num*gridLength
        w1 = w_num*gridLength
        h_num, w_num, _, gridLength, _ = coords.shape

        dx = torch.linspace(-r, r, 2*r+1, device=coords.device) #相对中心偏移量
        dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
        delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
        out_pyramid = []
        for i in range(self.num_levels):    #每层都要取
            gridLength_layer = gridLength // (2**i)
            h1_layer = h1 // (2**i)
            w1_layer = w1 // (2**i)
            corr = self.corr_pyramid[i]     #n个点，单通道，与其余64*64个点的corr值
            corr = corr.view(h_num, gridLength, w_num, gridLength, h1_layer, w1_layer)
            corr = corr.permute([0,2,1,3,4,5]).contiguous()
            
            corr_81_list = torch.zeros([h_num, w_num, 1, gridLength, gridLength, (2*r+1)**2], 
                                        dtype=torch.float32, device=coords.device)
            for i in range(h_num):
                for j in range(w_num):
                    if (self.Mask_big[0,0,i * gridLength,j * gridLength] > 0):
                        corr_p = corr[i,j,...].view(gridLength**2, 1, h1_layer, w1_layer)
                        centroid_lvl = coords[i,j,...].permute([1,2,0])/ 2**i
                        centroid_lvl = centroid_lvl.reshape(gridLength*gridLength, 1, 1, 2) 
                        #centroid_lvl[:,:,:,0] -= j * gridLength 不能剪，蕴含了
                        #centroid_lvl[:,:,:,1] -= i * gridLength
                        coords_lvl = centroid_lvl + delta_lvl   # 广播
                        corr_81 = bilinear_sampler(corr_p, coords_lvl)
                        corr_81 = corr_81.view(1, gridLength, gridLength, -1)
                        corr_81_list[i,j,...] = corr_81
            out_pyramid.append(corr_81_list)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute([0,1,2,5,3,4]).contiguous().float()#20,20,1,324,4,4

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd)
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())

