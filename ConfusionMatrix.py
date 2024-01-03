import torch
import numpy as np
class ConfusionMatrix():
    def __init__(self,num_classes):
        self._num_classes=num_classes
        self.confusion_matrix=None
        self._num_labels=None
        self._num_predicted_true=None


    def generate(self,y_pred,y_true):
        """

        :param y_pred: [batch,high,width],tensor
        :param y_true: [batch,high,width],tensor
        :return:
        """
        # k=(y_true>=0)&(y_true<self._num_classes)
        index=self._num_classes*y_true+y_pred
        self.confusion_matrix=torch.bincount(index.view(-1),minlength=self._num_classes**2).reshape(self._num_classes,self._num_classes)
        self._num_labels=y_pred.numel()
        self._num_predicted_true=torch.sum(y_pred==y_true)


    def MIOU(self):

        # get the value of diagonal [self._num_classes]
        diag_value=torch.diag(self.confusion_matrix)
        # calculate the union set of all classes, [self._num_classes]
        union=torch.sum(self.confusion_matrix,dim=0)+torch.sum(self.confusion_matrix,dim=1)-diag_value
        # check for the presence of 0 value in union tensor, which is caused by lack of specific class
        k=(union!=0)
        ratio=diag_value[k]/union[k]

        #Mean Intersection over Union
        miou=torch.mean(ratio,dtype=torch.float32)

        # calculate MIOU while ignoring the class of background which usually is represented by 0
        miou_without_background=torch.mean(ratio[1:],dtype=torch.float32)
        # calculate mean accuracy at pixel level

        avg_accruacy=self._num_predicted_true/self._num_labels

        return miou,miou_without_background,avg_accruacy
