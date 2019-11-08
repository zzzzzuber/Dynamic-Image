import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from skimage import io, color, exposure
import os
import numpy as np 

class di_dataset(Dataset):

    def __init__(self,root_dir,mode, transform=None):
        self.root_dir = root_dir
        self.image_names = os.listdir(self.root_dir)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        self.train2dict,self.val2dict = self.file2dict()
        if self.mode == 'train':
            return len(self.train2dict.keys())
        elif self.mode == 'val':
            return len(self.val2dict.keys())
        else:
            print('===============> There are only train and val mode in this task...')
    
    def load_di_image(self,image_name):
        classname = image_name.split('_')[1]  #v_HandStandPushups_g21_c05.jpg
        if classname == 'HandStandPushups':
            image_name = image_name.replace('HandStandPushups','HandstandPushups')
        image_name = image_name+'.jpg'
        img_path = self.root_dir+'/'+image_name
        img = Image.open(img_path)
        transformed_img = self.transform(img)
        img.close()
        return transformed_img

    def file2dict(self):
        train2dict={}
        val2dict={}
        classInd={}

        with open('/home/zhuc/pytorch/dynamic_image/pytorch-DI/data/ucf101_splits/trainlist01.txt','r')as fr:
                trainlist = fr.readlines()
        for line in trainlist:
            video_path, label = line.strip('\n').split(' ')
            videoname = video_path.split('/')[-1].split('.')[0]
            label = int(label)-1
            train2dict[videoname] = label
        
        with open('/home/zhuc/pytorch/dynamic_image/pytorch-DI/data/ucf101_splits/testlist01.txt','r')as frr:
            testlist = frr.readlines()

        with open('/home/zhuc/pytorch/dynamic_image/pytorch-DI/data/ucf101_splits/classInd.txt','r') as fc:
            labellist = fc.readlines()

        for l in labellist:
            l = l.strip('\n')
            label_number, classname = l.split(' ')
            classInd[classname]=int(label_number)
        

        for tline in testlist:
            labelname, test_videoname = tline.strip('\n').split('/')#ApplyEyeMakeup/v_ApplyEyeMakeup_g04_c07.avi
            test_videoname = test_videoname.split('.')[0] #v_ApplyEyeMakeup_g04_c07
            classname_list = list(classInd.keys())
            for i in range(len(classname_list)):
                if classname_list[i] == labelname:
                    label = int(classInd[labelname])-1
                else:
                    i = i+1
            val2dict[test_videoname]=label
        
        return train2dict,val2dict
            
            
    def __getitem__(self,index):
        #index means index number
        self.train2dict,self.val2dict = self.file2dict()
        
        if self.mode=='train':
            image_name = list(self.train2dict.keys())[index].split('.')[0]
            label = self.train2dict[image_name]
            img = self.load_di_image(image_name)
            sample = (image_name,img,label)

        elif self.mode=='val':
            image_name = list(self.val2dict.keys())[index].split('.')[0]
            label = self.val2dict[image_name]
            img=self.load_di_image(image_name)
            sample = (image_name,img,label)
        return sample

class di_dataloader():
    def __init__(self,Batch_size,num_workers):
        self.Batch_size = Batch_size
        self.num_workers = num_workers
    
    def train(self):
        training_set = di_dataset(root_dir='/home/zhuc/pytorch/dynamic_image/pytorch-DI/data/DI',mode='train',transform=transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]))
        print('==> Training data : %s' % str(len(training_set)))
        
        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.Batch_size,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader
    
    def validate(self):
        validation_set = di_dataset(root_dir='/home/zhuc/pytorch/dynamic_image/pytorch-DI/data/DI',mode='val', transform = transforms.Compose([
                #transforms.Scale([224,224])
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print('==> Validate data : %s' % str(len(validation_set)))
        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.Batch_size, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader

    def run(self):
        train_loader = self.train()
        val_loader = self.validate()
        return train_loader,val_loader

if __name__ == '__main__':
    root_dir = '/home/zhuc/pytorch/dynamic_image/pytorch-DI/data/DI'
    dataloader = di_dataloader(Batch_size=10, num_workers=8)
    train_loader,val_loader = dataloader.run()
    
   
    
        
            

    
