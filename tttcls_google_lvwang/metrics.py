import torch
import torch.nn as nn


class ClassificationMetric(nn.Module):
    def __init__(self, numClass, device='cpu'):
        super().__init__()
        self.numClass = numClass
        self.device = device
        self.reset(device)
    # OA
    def OverallAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc
    # UA
    def Precision(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
    # PA
    def Recall(self):
        # acc = (TP) / TP + FN
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def F1score(self):
        # 2*Recall*Precision/(Recall+Precision)
        p = self.Precision()
        r = self.Recall()
        return 2*p*r/(p+r)

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        # mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        # label = self.numClass * imgLabel[mask] + imgPredict[mask]
        label = self.numClass * imgLabel.flatten() + imgPredict.flatten()
        count = torch.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def getConfusionMatrix(self):  # 同FCN中score.py的fast_hist()函数
        # cfM = self.confusionMatrix / np.sum(self.confusionMatrix, axis=0)
        cfM = self.confusionMatrix
        return cfM

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self, device):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))
        if device=='cuda':
            self.confusionMatrix = self.confusionMatrix.cuda()


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# multi-label classification metric
class MultilabelMetric(nn.Module):
    def __init__(self, numClass, device='cpu'):
        super().__init__()
        self.numClass = numClass
        self.device = device
        self.reset(device)
    # OA
    def OverallAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc
    # UA
    def Precision(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
    # PA
    def Recall(self):
        # acc = (TP) / TP + FN
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def F1score(self):
        # 2*Recall*Precision/(Recall+Precision)
        p = self.Precision()
        r = self.Recall()
        return 2*p*r/(p+r)

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        # mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        # label = self.numClass * imgLabel[mask] + imgPredict[mask]
        label = self.numClass * imgLabel.flatten() + imgPredict.flatten()
        count = torch.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def getConfusionMatrix(self):  # 同FCN中score.py的fast_hist()函数
        # cfM = self.confusionMatrix / np.sum(self.confusionMatrix, axis=0)
        cfM = self.confusionMatrix
        return cfM

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self, device):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))
        if device=='cuda':
            self.confusionMatrix = self.confusionMatrix.cuda()


def plot_confusionmatrix(cm):
    r = cm.shape[0]
    c = cm.shape[1]
    for i in range(r):
        for j in range(c):
            print('%.3f'%cm[i,j], end=' ')
        print('\n', end='')


def acc2file(oa, f1, ua, pa, cm, txtpath):
    with open(txtpath, "a") as f:
        f.write('oa, f1, ua, pa, confusion_matrix\n')
        f.write(str(oa)+'\n')
        for i in f1:
            f.write(str(i)+' ')
        f.write('\n')
        for i in ua:
            f.write(str(i)+' ')
        f.write('\n')
        for i in pa:
            f.write(str(i)+' ')
        f.write('\n')

        r = cm.shape[0]
        for i in range(r):
            for j in range(r):
                f.write(str(cm[i,j])+' ')
            f.write('\n')

if __name__=="__main__":
    m = ClassificationMetric(3,device='cpu')
    ref = torch.tensor([0,0,1,1,2,2])
    pred = torch.tensor([0,1,0,1,0,2])
    m.addBatch(pred, ref)
    print(m.Precision())
    print(m.Recall())
    print(m.F1score())
