import torch
from torch import nn
from VOCDatasetLoader import VOCSegmentationDataset
from torch.utils.data import DataLoader
from FCN_ResNet import FCNResNet
from torch import optim
from basic_trainer import Trainer

import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import Utils
import numpy as np

# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("using %s device"%device)
parser = argparse.ArgumentParser(description='PyTorch FCN Training')

#If it's 1, start off training progress with "torch.distributed.launch", If it's 0,begin with "mp.spawn"
parser.add_argument('--multiprocessing_distributed',type=int,default=1)
parser.add_argument("--local_rank", type=int, default=-1)
#It represents the total count of your GPU which used to train model
parser.add_argument("--ngpus_per_node",type=int,default=1)
parser.add_argument("--seed",type=int,default=7)



def worker(rank,word_size):
    dist.init_process_group('nccl',rank=rank,world_size=word_size)
    torch.cuda.set_device(rank)


    train_config = {"epochs": 100,
                    "num_classes": 21,
                    'milestones':[70,80,90],
                    "WORLD_SIZE":dist.get_world_size(),
                    "experiment_dir":"./models"}

    #get the label of GPU in current process
    train_config['local_rank'] = dist.get_rank()
    train_config['device'] = "cuda:{}".format(train_config['local_rank'])

    # load dataset
    root = '/home/ztp/workspace/Dataset-Tool-Segmentation/data/VOC/VOC2012'
    trainVocDataset = VOCSegmentationDataset(is_train=True, crop_size=(320, 480), voc_root=root)
    testVocDataset = VOCSegmentationDataset(is_train=False, crop_size=(320, 480), voc_root=root)
    # trainVocDataloader = DataLoader(dataset=trainVocDataset, batch_size=64, shuffle=True)
    #testVocDataLoader = DataLoader(dataset=testVocDataset, batch_size=64, shuffle=False)

    # make sure each process call independent subset from raw datasets
    # if we have two running processes and the batch_size=32, our model will train 32*2 samples at once.
    dist_trainVocSampler = torch.utils.data.distributed.DistributedSampler(trainVocDataset,
                                                                           shuffle=True)
    dist_testVocSampler=torch.utils.data.distributed.DistributedSampler(testVocDataset,shuffle=False)

    trainVocDataloader = DataLoader(dataset=trainVocDataset,
                                    batch_size=32,
                                    sampler=dist_trainVocSampler,
                                    num_workers=train_config['WORLD_SIZE'])


    testVocDataloader=DataLoader(dataset=testVocDataset,
                                 batch_size=32,
                                 sampler=dist_testVocSampler,
                                 num_workers=train_config['WORLD_SIZE'])

    """
    create FCN model
    """
    num_classes = 21

    fcnNet = FCNResNet(ResNet_path='./pretrained_parameters/resnet18.pth',
                       out_channel=num_classes).cuda()
    #to alter BatchNorm in ResNet with SyncBatchNorm
    fcnNet=nn.SyncBatchNorm.convert_sync_batchnorm(fcnNet)
    torch.nn.parallel.DistributedDataParallel(module=fcnNet,
                                              device_ids=[train_config['local_rank']],
                                              output_device=[train_config['local_rank']])
    """
    set up optimizer and loss function
    """
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=fcnNet.parameters(), lr=0.001, eps=1.0e-8)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_config['milestones'], gamma=0.01)

    """
    train model
    """

    trainer = Trainer(model=fcnNet,
                      loss=loss,
                      optimizer=optimizer,
                      scheduler=lr_scheduler,
                      train_loader=trainVocDataloader,
                      test_loader=testVocDataloader,
                      train_sampler=dist_trainVocSampler,
                      test_sampler=dist_testVocSampler,
                      train_config=train_config,
                      device=train_config['device'])
    trainer.train()
    Utils.save_logs(data=trainer.logs,folder_path='./logs',filename='logs.pkl')

    dist.destroy_process_group()
if __name__ == '__main__':

    args=parser.parse_args()

    # set up random seed for current cpu
    torch.manual_seed(args.seed)
    #set up random seed for current gpu
    torch.cuda.manual_seed(args.seed)
    #set up random seed for all gpus
    torch.cuda.manual_seed_all(args.seed)
    #set up random seed for numpy
    np.random.seed(args.seed)

    if args.multiprocessing_distributed:
        local_rank=args.local_rank
        worker(local_rank,args.ngpus_per_node)
    else:
        mp.spawn(worker, nprocs=2, args=(args.ngpus_per_node,))

