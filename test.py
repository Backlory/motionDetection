import sys

from utils.timers import tic, toc   
sys.path.append('core')
sys.path.append('thirdparty_RAFT/core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from model.thirdparty_RAFT.core.raft import RAFT
from model.thirdparty_RAFT.core.utils import flow_viz
from model.thirdparty_RAFT.core.utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    #img = cv2.resize(img, (128, 128), cv2.INTER_LINEAR)
    #img = cv2.GaussianBlur(img, (5, 5), 1)
    #d = 128; img = img[128:128+d, 96:96+d,:]
    #simg = cv2.resize(img, (256,256), interpolation=cv2.INTER_NEAREST)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, name):
    if isinstance(flo, np.ndarray):
        assert(len(flo.shape) == 3)
        assert(flo.shape[2] == 2)
    elif isinstance(flo, torch.Tensor):
        flo = flo[0].permute(1,2,0).cpu().numpy()
    img = img[0].permute(1,2,0).cpu().numpy().astype(np.uint8)

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)

    flo_gray = cv2.cvtColor(flo, cv2.COLOR_BGR2GRAY)
    flo_activate = cv2.absdiff(flo_gray, cv2.blur(flo_gray, (64,64)))
    flo_activate = flo_activate / flo_activate.max() * 255
    flo_activate = flo_activate.astype(np.uint8)
    flo_activate = cv2.merge([flo_activate, flo_activate, flo_activate])
    img_flo = np.concatenate([img, flo, flo_activate], axis=1)

    cv2.namedWindow("image", cv2.WINDOW_FREERATIO)
    cv2.imshow('image', img_flo)
    #cv2.imwrite(f'{name}.png', img_flo)
    cv2.waitKey(1)
    return img_flo

def demo(args):
    model = RAFT(args)
    status = torch.load(args.model)
    status = {k[7:]:v for k,v in status.items()}
    model.load_state_dict(status)

    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        last_flow = None
        idx = 0
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            with torch.no_grad():
                flow_low, flow_up, coords0, coords1 = model(image1, image2, iters=20, test_mode=True, flow_init = last_flow)
            last_flow = flow_low.detach()
            t = tic()
            flow_up = flow_up[0].permute(1,2,0).cpu().numpy()
            toc(t, "cpu", mute=False)
            viz(image1, flow_up, idx)
            idx += 1 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="model/thirdparty_RAFT/model/raft-sintel.pth", help="restore checkpoint")  #things
    #parser.add_argument('--path', default="demo-frames", help="dataset for evaluation")
    parser.add_argument('--path', default=r"E:\dataset\dataset-fg-det\Janus_UAV_Dataset\Train\video_1\video", help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_false', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
