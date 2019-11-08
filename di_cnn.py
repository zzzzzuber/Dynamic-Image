import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import dataloader
#import driving_modification as dataloader
from utils import *
from network import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=10, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

def main():
    global arg
    arg = parser.parse_args()
    print (arg)

    #Prepare DataLoader
   
    #原始版本dataloader
    data_loader = dataloader.di_dataloader(
                        Batch_size=arg.batch_size,
                        num_workers=8
                        )

    train_loader, test_loader = data_loader.run()

    model = DI_CNN(
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                    
                        train_loader=train_loader,
                        test_loader=test_loader)
                        
    model.run()

class DI_CNN():
    def __init__(self, nb_epochs, lr, resume, batch_size, start_epoch,train_loader, test_loader):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.resume = resume
        self.batch_size=batch_size
        self.start_epoch=start_epoch
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=AverageMeter()
        self.dic_video_level_preds={}
        
    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = resnet101(pretrained= True, channel=3).cuda()
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        #self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)
    
    def resume_and_evaluate(self):
        if os.path.isfile(self.resume):
            print("==> loading checkpoint '{}'".format(self.resume))
            checkpoint = torch.load(self.resume)
            self.start_epoch = checkpoint['epoch']
            self.best_prec1 = checkpoint['best_prec1']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                .format(self.resume, checkpoint['epoch'], self.best_prec1))
        else:
            print("==> no checkpoint found at '{}'".format(self.resume))

    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True

        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            prec1 = self.validate_1epoch()
            is_best = prec1.avg > self.best_prec1.avg
            #lr_scheduler
            #self.scheduler.step(val_loss)
            #arg.lr = adjust_learning_rate(self.optimizer,self.epoch)
            # save model
            if is_best:
                self.best_prec1 = prec1
                with open('record/di_video_preds.pickle','wb') as f:
                    pickle.dump(self.dic_video_level_preds,f)
                f.close()
            
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : self.optimizer.state_dict()
            },is_best,'record/checkpoint.pth.tar','record/model_best.pth.tar')

    
    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        #switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (img_name,img,label) in enumerate(progress):

            #print(img_name)
            # measure data loading time
            #print(img.shape)
            
            data_time.update(time.time() - end)
            
            label = label.cuda(async=True)
            target_var = Variable(label).cuda()

            # compute output
            output = Variable(torch.zeros(len(img),101).float(),requires_grad=True).cuda()
            input_var = Variable(img).cuda()
            output = self.model(input_var)
            #print(output.shape,target_var.shape)
            #print(output.data)
            #print(target_var.data)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            #losses.update(loss.data[0], data.size(0))
            losses.update(loss.item(), img.size(0))
            #top1.update(prec1[0], data.size(0))
            top1.update(prec1.item(), img.size(0))
            #top5.update(prec5[0], data.size(0))
            top5.update(prec5.item(), img.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[np.round(batch_time.avg,3)],
                'Data Time':[np.round(data_time.avg,3)],
                'Loss':[float(np.round(losses.avg,5))],
                'Prec@1':[float(np.round(top1.avg,4))],
                'Prec@5':[float(np.round(top5.avg,4))],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/rgb_di_train.csv','train')
        del losses
        del progress
        # del self.train_loader

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        time.sleep(10)
        batch_time = AverageMeter()
        #losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        #self.dic_video_level_preds={}
        end = time.time()
        progress = tqdm(self.test_loader)
        preds = {}
        #outputs = {}
        for i, (img_name,img,label) in enumerate(progress):
            
            label = label.cuda(async=True)
            
            with torch.no_grad():
                data_var = Variable(img).cuda(async=True)
               
                label_var = Variable(label).cuda(async=True)
           
            # compute output
            output = self.model(data_var)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # #Calculate video level prediction
            
            preds = output.data.cpu().numpy()
            
            # # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            
            top1.update(prec1.item(), img.size(0))
            
            top5.update(prec5.item(), img.size(0))
            
            nb_data = preds.shape[0]
            for j in range(nb_data):
                videoName = img_name[j].split('/',1)[0]
                if videoName not in self.dic_video_level_preds.keys():
                    self.dic_video_level_preds[videoName] = preds[j,:]
                else:
                    self.dic_video_level_preds[videoName] += preds[j,:]
        
        
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[np.round(batch_time.avg,3)],
                'Prec@1':[np.round(top1.val)],
                'Prec@5':[np.round(top5.get(),3)]}
        record_info(info, 'record/rgb_di_test.csv','test')
        return top1
'''
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = arg.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
'''
if __name__=='__main__':
    main()