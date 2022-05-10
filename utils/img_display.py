from math import ceil
import numpy as np
import torch
import cv2
import os, sys,sys

__all__ = ['img_square', 'save_pic', 'show_pic']

def prepare_path(name_dir):
    '''
    生成路径。
    '''
    if os.path.exists(name_dir):
        pass
    elif sys.platform=='win32':
        name_dir = name_dir.replace('/','\\')
        os.system('mkdir ' + name_dir)
    else:
        name_dir = name_dir.replace('\\','/')
        os.system('mkdir -p ' + name_dir)
    return 1

#@fun_run_time
def save_pic(data,filepath = '', visible=True):
    '''
    输入二维图片，三维、四维图片都可。BGR的。
    如果路径不存在则创建。
    
    save_pic(temp,'test/test/1.png')，
    '''
    if np.max(data)<=1 and np.max(data)>0: 
        data=data*255.
    data=np.uint8(data)
    #
    try:
        filedir, filename = os.path.split(filepath)
        prepare_path(filedir)
    except:
        pass
    #
    if len(data.shape)==3:
        if (data.shape[0] == 1) or (data.shape[0] == 3):
            data = np.transpose(data,(1,2,0))
    elif len(data.shape)==4:
        if (data.shape[1] == 1) or (data.shape[1] == 3):    #NCHW,BGR
            data = np.transpose(data,(0,2,3,1))              #nhwc,BGR
        data = img_square(data)
    #
    if cv2.imwrite(filepath, data):
        if visible:
            print(f'\t----image saved in {filepath}')
    else:
        print(f'\t----Warning: image not saved in {filepath}!')



def show_pic(data,windowname = 'default',showtype='freeze'):
    '''
    展示CV图片。二维，三维，四维都可。BGR的。
    show_pic(pic,"asd","freeze") 冻结型显示
    show_pic(pic,"asd","freedom")自由型显示
    '''
    if np.max(data)<=1 and np.max(data)>0: 
        data=data*255.
    data=np.uint8(data)
    #
    if len(data.shape)==3:
        if (data.shape[0] == 1) or (data.shape[0] == 3):    #CHW
            data = np.transpose(data,(1,2,0))                #HWC
    elif len(data.shape)==4:
        if (data.shape[1] == 1) or (data.shape[1] == 3):
            data = np.transpose(data,(0,2,3,1))
        data = img_square(data)
    #
    cv2.namedWindow(windowname,0)
    cv2.imshow(windowname, data)

    if showtype=='freeze':
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

def _check_ifnhwc(imgs):
    '''
    检查图片是否符合cv图像输出规范，即是否是nhwc，且c为1或3通道.
    不检查格式是uint8还是float32，也不管是不是BGR。
    '''
    temp = imgs.shape
    #
    try:
        assert(len(temp) == 4)
    except:
        raise Exception('图片格式不对，应为[num, height, width, channal]') 
    #
    try:
        assert(temp[3] == 1 or temp[3] == 3)
    except:
        raise Exception('通道数应为1或3！') 
    return 1


def img_square(imgs:list, h_number = -1, w_number = -1, big_or_small='biggest'):
    '''
    1、将所给图片列表中所有图片都转成nhw3,支持(n1hw, nhw1, n3hw, nhw, 2hw)
    2、大小统一到最大图片大小
    3、整合为最合适的矩形，hwc。4d->3d
    允许使用None作为占位符
    不论图像是BGR还是RGB。
    会对非uint8类型做归一化，映射到0~255。会对范围仅为-30~30的图像做归一化
    big_or_small='biggest'代表会就着大的来，'smallest'反之
    
    h_number和w_number均为0，代表默认正方形。
    h_number和w_number均不为0，代表制定高和宽上的图片数。多余的图片会被丢弃。
    h_number和w_number有一个不为0，代表指定行或列数，此时列或行数自动计算。
    
    patches_pred = []
    patches_pred.append(img1) #hw1
    patches_pred.append(img2) #hw3
    patches_pred.append(None) #placeholder
    patches_pred.append(img3) #3hw
    patches_pred.append(img4) #1hw
    patches_pred.append(img5) #hw
    
    img = img_square(patches_pred, 0, 0)
    img = img_square(patches_pred, 2, 5)
    '''
    # 确定行列
    num = len(imgs)
    if w_number * h_number < 0:
        if h_number != -1:#选定行数
            w_number = ceil(num/h_number)
        elif w_number != -1:#选定列数
            h_number = ceil(num/w_number)
    elif w_number * h_number > 0:
        if h_number < 0 and w_number < 0:#默认正方形
            h_number = ceil(num**0.5)
            w_number = h_number
    if h_number*w_number-num > 0:
        imgs = imgs + [None] * (h_number*w_number-num)
    
    #
    if isinstance(imgs, list):
        #如果有torch，转成numpy
        imgs = [x[0] if type(x) is torch.Tensor and len(x.shape) == 4 else x for x in imgs ]
        imgs = [x.detach().cpu().numpy() if type(x) is torch.Tensor else x for x in imgs ]
        # 规整图片，获取合适尺寸
        size_max = (1,1,1)  #h, w, h*w
        size_min = (9e5,9e5,9e5)
        for idx in range(len(imgs)):
            img = imgs[idx]
            #
            if img is None:
                img = np.zeros((10,10,3)).astype(np.uint8)
            if img.dtype != np.uint8 or (img.max()<=30 and img.min()>=-30):   #normalization
                m = img.min()
                M = img.max()
                if m != M:
                    img = (img-m)/(M-m) * 255
                img = img.astype(np.uint8)
            #
            if len(img.shape)==2:   #hw->hw3
                img = np.dstack([img,img,img])
            elif len(img.shape)==3:
                if img.shape[0] == 3: #3hw->hw3
                    img = img.transpose(1,2,0)
                elif img.shape[0] == 2: #2hw->hw3
                    img = np.dstack([img[0],img[1],np.zeros_like(img[0])])
                elif img.shape[0] == 1: #1hw->hw3
                    img = img[0,:,:]
                    img = np.dstack([img,img,img])
                elif img.shape[2] == 1: #hw1->hw3
                    img = img[:,:,0]
                    img = np.dstack([img,img,img])
            else:
                raise ValueError("Check the imgs!")
            assert( len(img.shape)==3 and img.shape[2]==3)
            h, w = img.shape[0], img.shape[1]
            temp = h*w
            if temp > size_max[2]:
                size_max = (h, w, temp)
            if temp < size_min[2]:
                size_min = (h, w, temp)
            imgs[idx] = img
        # 处理大小
        if big_or_small == 'biggest':
            imgs = [cv2.resize(img, (size_max[1], size_max[0]), interpolation=cv2.INTER_NEAREST) for img in imgs]
        elif big_or_small == 'smallest':
            imgs = [cv2.resize(img, (size_min[1], size_min[0]), interpolation=cv2.INTER_NEAREST) for img in imgs]
        imgs = np.array(imgs)

    _check_ifnhwc(imgs)
    #
    num, height, width, chan = imgs.shape
    img_out_mat = np.zeros((h_number * height, w_number * width, chan),
                            dtype = imgs.dtype)
    #
    for m in range(h_number): #m行n列，m*height+n+1
        for n in range(w_number):#拼接	
            img_out_mat[m*height:(m+1)*height, n*width:(n+1)*width, :] = imgs[m*w_number+n,:,:,:]
    #
    return img_out_mat

