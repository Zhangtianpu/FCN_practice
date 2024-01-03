import torch
import os
from ConfusionMatrix import ConfusionMatrix
import time
import torch.distributed as dist


class Trainer(object):

    def __init__(self,model,loss,optimizer,scheduler,train_loader,test_loader,train_sampler,test_sampler,train_config,device):
        """
        模型训练类，包含了模型训练，模型存储，模型加载以及模型评估
        :param model:
        :param loss:
        :param optimizer:
        :param scheduler:
        :param train_loader:
        :param val_loader:
        :param test_loader:
        :param train_config:
        :param device:
        """

        super(Trainer, self).__init__()
        self.model=model
        self.train_config=train_config
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.train_sampler=train_sampler
        self.test_sampler=test_sampler
        self.device=device
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.loss=loss
        self.logs={'train_loss':[],
                   'train_acc':[],
                   'train_miou':[],
                   'train_miou_no_bg':[],
                   'test_loss':[],
                   'test_acc':[],
                   'test_miou':[],
                   'test_miou_no_bg':[],
                   'epoch':[]}


    def train_epoch(self,matrix):
        """
        训练模型
        :return: 当前批次的损失 float
        """


        self.model.train()
        #当前批次模型总损失值
        total_loss=0

        total_miou=0
        total_miou_without_background=0
        total_accuracy=0

        iteration_index=0


        for X, labels in self.train_loader:

            iteration_index+=1
            #将样本数据加载到gpu上
            X,labels=self._load_gpu(X,labels)

            #得到模型预测结果[64,21,320,480]
            output=self.model(X)

            # calculate confusion matrix, MIOU, Accuracy
            matrix.generate(y_pred=torch.argmax(output,dim=1),y_true=labels)
            miou,miou_withou_background,accuracy=matrix.MIOU()

            #计算模型损失 labels [64,320,480]
            loss=self.loss(output,labels)

            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

            total_loss+=loss.item()

            total_miou+=miou.item()
            total_miou_without_background+=miou_withou_background.item()
            total_accuracy+=accuracy.item()

        return total_loss/iteration_index,total_miou/iteration_index,total_miou_without_background/iteration_index,total_accuracy/iteration_index


    def train(self):

        #Mean Intersection over Union
        best_mean_iou=0
        matrix = ConfusionMatrix(self.train_config['num_classes'])

        for epoch in range(1,self.train_config['epochs']+1):
            start_time = time.time()
            self.train_sampler.set_epoch(epoch)
            #训练模型
            train_epoch_loss,train_epoch_miou,train_miou_without_background,train_epoch_accuracy=self.train_epoch(matrix)

            end_time=time.time()
            train_time=end_time-start_time

            self.logs['train_loss'].append(train_epoch_loss)
            self.logs['train_acc'].append(train_epoch_accuracy)
            self.logs['train_miou'].append(train_epoch_miou)
            self.logs['train_miou_no_bg'].append(train_miou_without_background)

            print("[process %s] [%s]: training_time:%.2f秒\t train_loss:%.3f\t train_moiu:%.3f\t train_miou_no_bg:%.3f\t train_accuracy:%.3f\tlr:%.6f" % (
            self.train_config['local_rank'],epoch,train_time, train_epoch_loss,train_epoch_miou,train_miou_without_background,train_epoch_accuracy,self.scheduler.get_lr()[0]))

            # 使用验证集，验证模型效果，并保存模型
            avg_loss,avg_miou,avg_miou_without_background,avg_accuracy=self.evaluate()
            self.logs['test_loss'].append(avg_loss)
            self.logs['test_acc'].append(avg_accuracy)
            self.logs['test_miou'].append(avg_miou)
            self.logs['test_miou_no_bg'].append(avg_miou_without_background)

            if best_mean_iou<avg_miou and dist.get_rank()==0:
                best_mean_iou=avg_miou
                self.save_model(epoch)
                print("[Validation]\tavg_loss:%.3f\tavg_miou:%.3f\tavg_miou_no_bg:%.3f\taccuracy:%.3f\n"%(avg_loss,avg_miou,avg_miou_without_background,avg_accuracy))


            self.scheduler.step()
            time.sleep(3)


    def evaluate(self):
        self.model.eval()
        matrix = ConfusionMatrix(self.train_config['num_classes'])
        total_miou=torch.zeros(1).to(self.device)
        total_miou_without_background=torch.zeros(1).to(self.device)
        total_accuracy=torch.zeros(1).to(self.device)
        total_loss=torch.zeros(1).to(self.device)

        step=0
        with torch.no_grad():
            for X,label in self.test_loader:
                step+=1
                X,labels=self._load_gpu(X,label)
                output=self.model(X)
                loss = self.loss(output, labels)
                total_loss+=loss
                matrix.generate(y_pred=torch.argmax(output, dim=1), y_true=labels)
                miou, miou_without_background, accuracy = matrix.MIOU()
                total_miou+=miou
                total_miou_without_background+=miou_without_background
                total_accuracy+=accuracy

        avg_loss=self.reduce_value(total_loss/step,avg=True)
        avg_miou=self.reduce_value(total_miou/step,avg=True)
        avg_miou_without_background=self.reduce_value(total_miou_without_background/step,avg=True)
        avg_accuracy=self.reduce_value(total_accuracy/step,avg=True)
        return avg_loss.item(),avg_miou.item(),avg_miou_without_background.item(),avg_accuracy.item()

    def reduce_value(self,value,avg=True):

        world_size=self.train_config['WORLD_SIZE']
        dist.all_reduce(value,op=dist.ReduceOp.SUM)
        if avg:
            return value/world_size
        else:
            return value


    def _load_gpu(self,X,labels):
        """
        加载样本数据到gpu
        :param X:
        :param labels:
        :return:
        """
        return X.float().to(self.device),labels.to(self.device)

    def save_model(self,epoch):
        """
        保存模型
        :param epoch: 训练模型的批次
        :return:
        """
        save_model_path=os.path.join(self.train_config['experiment_dir'],'FCN_ResNet18_%s.pkl'%epoch)
        torch.save(self.model.state_dict(),save_model_path)
        print("save model:%s"%save_model_path)


    def predict(self):
        self.model.eval()

        data_loader = self.test_loader
        results=[]
        correct = 0.0
        size = len(self.test_loader.dataset)
        with torch.no_grad():
            for X, labels in data_loader:
                X, labels = self._load_gpu(X, labels)
                ouptuts = self.model(X)

                label = ouptuts.argmax(1).cpu().detach().numpy()
                correct += (ouptuts.argmax(1) == labels).type(torch.float).sum().item()
                results.extend(label)
        correct /= size
        return correct,results

    def print_model_parameters(self):
        for name, parms in self.model.named_parameters():
            print('-->name:', name)
            print('-->para:', parms)
            print('-->grad_requirs:', parms.requires_grad)
            print('-->grad_value:', parms.grad)
            print("===")





