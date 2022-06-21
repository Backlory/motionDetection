import sys
import cv2
from cv2 import RANSAC
import numpy as np
import torch
import random

from utils.img_display import save_pic, img_square
from utils.mics import colorstr
from utils.timers import tic, toc

from model.Homo import Homo_cnn, Homo_fc

from algorithm.infer_VideoProcess import Inference_VideoProcess

# This draws matches and optionally a set of inliers in a different color
# Note: I lifted this drawing portion from stackoverflow and adjusted it to my needs because OpenCV 2.4.11 does not
# include the drawMatches function
def drawMatches(img1, kp1, img2, kp2, matches, inliers = None):
    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns, y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        inlier = False

        if inliers is not None:
            for i in inliers:
                if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                    inlier = True

        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points, draw inliers if we have them
        if inliers is not None and inlier:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
        elif inliers is not None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

        if inliers is None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    return out


# Runs sift algorithm to find features
def findFeatures(img, orb):
    #orb = cv2.cornerHarris()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    #temp = np.zeros_like(img)
    #img = cv2.drawKeypoints(img, keypoints, temp)#"orb_keypoints.png"
    #cv2.imwrite('sift_keypoints.png', img)
    return keypoints, descriptors

# Matches features given a list of keypoints, descriptors, and images
def matchFeatures(kp1, kp2, desc1, desc2, img1, img2):
    matcher = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = matcher.match(desc1, desc2)
    #matchImg = drawMatches(img1,kp1,img2,kp2,matches)
    #cv2.imwrite('Matches.png', matchImg)
    return matches


# Computers a homography from 4-correspondences
def calculateHomography(correspondences):
    #loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    #svd composition
    u, s, v = np.linalg.svd(matrixA)

    #reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    #normalize and now we have h
    h = (1/h.item(8)) * h
    return h


#Calculate the geometric distance between estimated points and original points
def geometricDistance(correspondence, h):
    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


#Runs through ransac algorithm, creating homographies from random correspondences
def ransac(corr, thresh):
    '''
    corr = np.matrix(n, 4) -> [x1, y1, x2, y2]
    thresh = 0.6，有多少点进入内点即终止
    '''
    maxInliers = []
    finalH = None
    for i in range(1000):
        #find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        #call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 20:   #【】【】【】【】
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        #print ("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr)*thresh):
            break
    return finalH, maxInliers

class Inference_Homo_RANSAC():
    def __init__(self, args={}) -> None:
        self.args = args
        self.orb = cv2.ORB_create(nfeatures=200, nlevels=1, scaleFactor=2)  #取消金字塔抽取
        # 设备

    def run_test(self, fps_target=30):
        print("==============")
        print(f"run testing wiht fps_target = {fps_target}")
        #path = r"E:\dataset\dataset-fg-det\UAC_IN_CITY\video_all_1.mp4"
        #path = r"E:\dataset\dataset-fg-det\Janus_UAV_Dataset\train_video\video_all.mp4"
        path = r'E:\dataset\dataset-fg-det\Kaggle-Drone-Videos\video_all.mp4'
        cap = cv2.VideoCapture(path)
        #
        tempVideoProcesser = Inference_VideoProcess(cap=cap,fps_target=fps_target)
        fps = tempVideoProcesser.fps_now
        self.ss1,self.ss2 = [],[]
        self.effect_all = []
        t_use_all = []
        idx = 0
        frameUseless = 0
        while(True):
            idx += 1
            #if idx>10:break

            img_t0, img_t1 = tempVideoProcesser()
            if img_t0 is None:
                print("all frame have been read.")
                break
            # ==============================================↓↓↓↓
            try:
                img_t0, img_t1_warped, diffOrigin, diffWarp, if_usefull, t_use = self.__call__(img_t1, img_t0)
                #temp = [img_t0, img_t1, img_t1_warped, cv2.absdiff(img_t0, img_t1_warped), cv2.absdiff(img_t0, img_t1)]
                #cv2.imwrite(f"{round(fps)}_{stride}.png", img_square(temp, 2,3))
                
                # ==============================================↑↑↑↑
                if not if_usefull:
                    frameUseless += 1
                    diffOrigin = 1
                    diffWarp = diffOrigin
                    effect = 1 - diffWarp/diffOrigin
                self.ss1.append(diffOrigin)
                self.ss2.append(diffWarp)
                if if_usefull:
                    effect = 1 - diffWarp/diffOrigin
                    self.effect_all.append(effect)
                t_use_all.append(t_use)
                print(f'\r== frame {idx} ==> diff_origin = {diffOrigin}, diff_warp = {diffWarp}', "rate=", effect,"time=",t_use,'ms',  end="")
            except:
                pass
            #cv2.imshow("test_origin", img_t0)
            #cv2.imshow("test_diff_origin",  cv2.absdiff(img_t0, img_t1))
            #cv2.imshow("test_diff_warp",  cv2.absdiff(img_t0, img_t1_warped))
            #cv2.waitKey(1)
            #if cv2.waitKey(int(1000/fps)) == 27: break
        print("\nframeUseless = ", frameUseless)
        
        #保存到文件
        savedStdout = sys.stdout
        with open("log.txt", "a+") as f:
            sys.stdout = f
            effect_all = np.average(self.effect_all)
            ss1_all = np.average(self.ss1)
            ss2_all = np.average(self.ss2)
            avg_t_use_all = np.average(t_use_all)
            print(f"{0}|{fps_target}|{0}|{idx}|{frameUseless}|{1-frameUseless/idx}|{ss1_all}|{ss2_all}|{1-ss2_all/ss1_all}|{effect_all}|{avg_t_use_all}")
        sys.stdout = savedStdout
            
        cv2.destroyAllWindows()
        cap.release()


    def __call__(self, img_t0, img_base, stride=4, alpha=0, checkusefull = True):
        '''
        将t0向t_base扭曲
        '''
        assert(img_t0.shape == img_base.shape)
        assert(img_t0.shape[2] == 3)
        h, w, _ = img_t0.shape
        assert(w >= h)
        assert(h == 512)
        #
        img_t0_gray = cv2.cvtColor(img_t0, cv2.COLOR_BGR2GRAY)
        img_base_gray = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
        
        # RANSAC方法
        correspondenceList = []
        t = tic()
        H_warp = self.core(img_t0_gray, img_base_gray)
        
        if False:
            for match in matches:
                (x1, y1) = keypoints[0][match.queryIdx].pt
                (x2, y2) = keypoints[1][match.trainIdx].pt
                correspondenceList.append([x1, y1, x2, y2])
            corrs = np.matrix(correspondenceList)
            H_warp_0, inliers = ransac(corrs, 0.6)
            print ("Final inliers count: ", len(inliers))
        #matchImg = drawMatches(img_t0_gray,kp1,img_base_gray,kp2,matches,inliers)
        #cv2.imwrite('InlierMatches.png', matchImg)
        h, w, _ = img_t0.shape
        img_t0_warp = cv2.warpPerspective(img_t0, H_warp, (w, h))
        
        # 有效性检查
        if checkusefull:
            ret, mask = cv2.threshold(img_t0_warp, 1, 1, cv2.THRESH_BINARY)
            
            diffOrigin = cv2.absdiff(img_t0, img_base)       #扭前
            diffWarp = cv2.absdiff(img_t0_warp, img_base)   #扭后
            
            diffOrigin = cv2.multiply(diffOrigin, mask)
            diffWarp = cv2.multiply(diffWarp, mask)
            
            diffOrigin = np.round(np.sum(diffOrigin), 4)
            diffWarp = np.round(np.sum(diffWarp), 4)
            
            if_usefull = (diffOrigin > diffWarp)
            if not if_usefull:
                img_t0_warp = img_t0
                
            t_use=toc(t,"ransac",mute=True)
            return img_base, img_t0_warp, diffOrigin, diffWarp, if_usefull, t_use
        else:
            return img_base, img_t0_warp

    def core(self, img_t0_gray, img_base_gray):
        try:
            kp1, desc1 = findFeatures(img_t0_gray, self.orb)
            kp2, desc2 = findFeatures(img_base_gray, self.orb)
            keypoints = [kp1,kp2]
            tp, qp = [], []
            matches = matchFeatures(kp1, kp2, desc1, desc2, img_t0_gray, img_base_gray)
            for match in matches:
                tp.append(keypoints[0][match.queryIdx].pt)
                qp.append(keypoints[1][match.trainIdx].pt)
            tp, qp = np.float32((tp, qp))
            H_warp, status = cv2.findHomography(tp, qp, cv2.RANSAC, 5)
        except:
            print("error!")
            H_warp = np.array([1,0,0,0,1,0,0,0,1], dtype=np.float32()).reshape(3,3)
        if H_warp is None:
            H_warp = np.array([1,0,0,0,1,0,0,0,1], dtype=np.float32()).reshape(3,3)
        return H_warp

    def time_test(self):
        print("==============")
        #
        path = r"E:\dataset\dataset-fg-det\Janus_UAV_Dataset\train_video\video_all.mp4"
        cap = cv2.VideoCapture(path)
        #
        tempVideoProcesser = Inference_VideoProcess(cap=cap,fps_target=30)
        fps = tempVideoProcesser.fps_now
        img_base, img_t0 = tempVideoProcesser()
        
        assert(img_t0.shape == img_base.shape)
        assert(img_t0.shape[2] == 3)
        h, w, _ = img_t0.shape
        assert(w >= h)
        assert(h == 512)
        #
        img_t0_gray = cv2.cvtColor(img_t0, cv2.COLOR_BGR2GRAY)
        img_base_gray = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
        

        t = tic()
        estimation_thresh = 0.60
        for _ in range(300):
            # RANSAC方法
            correspondenceList = []
            kp1, desc1 = findFeatures(img_t0_gray, self.orb)
            kp2, desc2 = findFeatures(img_base_gray, self.orb)
            keypoints = [kp1,kp2]
            matches = matchFeatures(kp1, kp2, desc1, desc2, img_t0_gray, img_base_gray)
            for match in matches:
                (x1, y1) = keypoints[0][match.queryIdx].pt
                (x2, y2) = keypoints[1][match.trainIdx].pt
                correspondenceList.append([x1, y1, x2, y2])
            corrs = np.matrix(correspondenceList)
            H_warp, inliers = ransac(corrs, estimation_thresh)

        toc(t, "inference", 300, False)



