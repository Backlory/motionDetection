import numpy as np
import torch

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + 
                    np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) +
                                                np.sum(self.confusion_matrix, axis=0) -
                                                np.diag(self.confusion_matrix))
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        FWIoU = (freq[freq > 0] * MIoU[freq > 0]).sum()
        return FWIoU
    
    def pre_recall_f1(self, i):
        CM = self.confusion_matrix
        TP = CM[i, i]
        FP = CM[:, i].sum() - TP
        FN = CM[i, :].sum() - TP
        Pre = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2*TP / (2*TP + FP + FN)
        return Pre, Recall, F1

    def m_pre_recall_f1(self):
        class_Pre = []
        class_Recall = []
        class_F1 = []
        for i in range(self.num_class):
            #
            Pre, Recall, F1 = self.pre_recall_f1(i)
            #
            class_Pre.append(Pre)
            class_Recall.append(Recall)
            class_F1.append(F1)
        mPre = np.mean(class_Pre)
        mRecall = np.mean(class_Recall)
        mF1 = np.mean(class_F1)
        return mPre, mRecall, mF1

    def AuC(self):
        return 0

    def _generate_matrix(self, pre_image:torch.tensor, gt_image:torch.tensor):
        n = len(gt_image)
        confusion_matrix = np.zeros([self.num_class, self.num_class])
        
        gt_image = torch.argmax(gt_image, 1).view(-1)   # n, h, w -> nhw
        preds = torch.argmax(pre_image, 1).view(-1)   # 
        
        for p, t in zip(preds, gt_image):
            confusion_matrix[p, t] += 1
        return confusion_matrix

    def add_batch(self, pre_image, gt_image):
        #gt_image和pre_image都是0、1、2这类的
        assert (gt_image.shape == pre_image.shape)
        self.confusion_matrix += self._generate_matrix(pre_image, gt_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2, int)

    def evaluateAll(self):
        mIoU = self.Mean_Intersection_over_Union()
        FWIoU = self.Frequency_Weighted_Intersection_over_Union()
        Acc = self.Pixel_Accuracy()
        mAcc = self.Pixel_Accuracy_Class()
        if self.num_class > 2:
            mPre, mRecall, mF1 = self.m_pre_recall_f1()
        else:
            mPre, mRecall, mF1 = self.pre_recall_f1(1)
        AuC = self.AuC()
        return mIoU, FWIoU, Acc, mAcc, mPre, mRecall, mF1, AuC



if __name__ == "__main__":
    print('='*20)
    myEvaluator = Evaluator(3)
    myEvaluator.reset()
    myEvaluator.confusion_matrix = np.array(
        [
            [15, 0, 1],
            [0, 17, 4],
            [0, 3, 5]
        ]
    ) 
    '''  
    myEvaluator.confusion_matrix = np.array(
        [
            [7, 8, 9],
            [1, 2, 3],
            [3, 2, 1]
        ]
    )'''
    print(myEvaluator.confusion_matrix)
    mIoU, FWIoU, Acc, mAcc, mPre, mRecall, mF1, AuC = myEvaluator.evaluateAll()
    print(f"mIoU={mIoU:.4f}, FWIoU={FWIoU:.4f}, Acc={Acc:.4f}, mAcc={mAcc:.4f}")
    print(f"mPre={mPre:.4f}, mRecall={mRecall:.4f}, mF1={mF1:.4f}, AuC={AuC:.4f}")
    for i in range(3):
        print(myEvaluator.pre_recall_f1(i))