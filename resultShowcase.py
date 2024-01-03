import Utils
import os
import matplotlib.pyplot as plt
import numpy as np
from FCN_ResNet import FCNResNet
import torch
from VOCDatasetLoader import VOCSegmentationDataset
from torch.utils.data import DataLoader
from PIL import Image,ImagePalette
from torchvision import transforms
from torchvision.transforms import functional as f

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_curve(x,train,test,title,xlabel,ylabel,is_save=False,path=None):
    plt.figure(figsize=(20, 5))
    plt.plot(x, train)
    plt.plot(x, test)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(['train', 'test'])
    plt.title(title)
    plt.grid(visible=True,linestyle='--')

    if is_save:
        if path is None:
            raise Exception("Path is None")
        plt.savefig(path, dpi=600, format='png', bbox_inches='tight')
    plt.show()

def case_study_display(sample_number,epoch,is_save=False,path=None):
    num_classes = 21

    fcnNet = FCNResNet(ResNet_path='./pretrained_parameters/resnet18.pth',
                       out_channel=num_classes).to(device)
    Utils.load_model(fcnNet, experiment_dir='./models', epoch=epoch, device=device)

    root = '/home/ztp/workspace/Dataset-Tool-Segmentation/data/VOC/VOC2012'
    testVocDataset = VOCSegmentationDataset(is_train=False, crop_size=(320, 480), voc_root=root)

    palette = np.array(testVocDataset._VOC_COLORMAP).astype(dtype=np.uint8)

    _,figs=plt.subplots(nrows=sample_number,ncols=3,figsize=(12,10))
    for index in range(sample_number):
        input_image_path = testVocDataset._images_path[index]
        input_img = Image.open(input_image_path)
        processed_image = testVocDataset._trans(input_img)
        processed_image = processed_image.unsqueeze(dim=0)

        label_image = testVocDataset._labels_path[index]
        label_img = Image.open(label_image)

        output = fcnNet(processed_image.to(device))
        output = torch.argmax(output, dim=1).cpu().squeeze()
        pred_img = palette[output]
        if index==0:
            figs[index,0].set_title(label='Origin Img')
            figs[index,1].set_title(label='Labeled Img')
            figs[index,2].set_title(label='Predicted Img')
        figs[index,0].imshow(input_img)
        figs[index,1].imshow(label_img)
        figs[index,2].imshow(pred_img)
    if is_save:
        if path is None:
            raise Exception("Path is None")
        plt.savefig(path, dpi=600, format='png', bbox_inches='tight')

    plt.show()





if __name__ == '__main__':
    folder_path="./logs"
    file_name='logs.pkl'
    logs=Utils.load_logs(folder_path=folder_path,filename=file_name)

    train_loss=np.array(logs['train_loss'])
    test_loss=[ loss for loss in logs['test_loss']]
    test_loss=np.array(test_loss)
    X=range(len(train_loss))

    train_acc=np.array(logs['train_acc'])
    test_acc=[acc for acc in logs['test_acc']]
    test_acc=np.array(test_acc)

    train_miou=np.array(logs['train_miou'])
    test_miou=[miou for miou in logs['test_miou']]
    test_miou=np.array(test_miou)

    plot_curve(X,train_loss,test_loss,"Learning Curve","Epoch","Loss",is_save=True,path='./ExperimentImages/LearningCurve.png')
    plot_curve(X,train_acc,test_acc,"Accuracy Curve","Epoch","Accuracy",is_save=True,path='./ExperimentImages/AccuracyCurve.png')
    plot_curve(X, train_miou, test_miou, "Mean Intersection Over Union", "Epoch", "MIOU",is_save=True,path='./ExperimentImages/MIOU.png')

    case_study_display(sample_number=6,epoch=81,is_save=True,path='./ExperimentImages/CaseStudy.png')



