# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import skimage.io
from PIL import Image
import cv2

video_root_path = '/home/zhuc/pytorch/two-stream/data/UCF101/jpegs_256'
saves_root_path = '/home/zhuc/pytorch/dynamic image/pytorch-DI/data/DI'

def computeDynamicImages_fromImages(images_directory):
    images_directory_list = os.listdir(images_directory)
    images_directory_list.pop()
    Nframes = len(os.listdir(images_directory))
    indv = np.arange(0, Nframes)
    fw = np.zeros(Nframes)
    videoname = images_directory.split('/')[-1]
    #根据文件夹内的图片确定需要初始化的DI尺寸，输出(N,224,224,3)
    def get_image_shape(images_dir):
        img = skimage.io.imread(os.path.join(images_dir,os.listdir(images_dir)[0]))
        N = len(os.listdir(images_dir))
        imageShape = (N,) + img.shape
        return imageShape
    
    videoShape = get_image_shape(images_directory)
    InputVideo = np.zeros(videoShape)
    for i,file in enumerate(images_directory_list):
        InputVideo[i] = skimage.io.imread(os.path.join(images_directory,file))
    for i in range(1,Nframes+1):
        fw[i-1] = round(
                sum((2*np.arange(i,Nframes+1)-Nframes-2)/(np.arange(i,Nframes+1))),
                2)
        
    DI = np.sum(
            np.multiply(InputVideo[indv,:,:,],np.reshape(fw,newshape=(Nframes,1,1,1))),
            0)    
    DI = DI - np.min(DI)
    DI = 255*DI/np.max(DI)
    DI = np.uint8(DI)
    img_dir = saves_root_path+'/'+videoname+'.jpg'
    if not os.path.exists(img_dir):
        cv2.imwrite(saves_root_path+'/'+videoname+'.jpg',DI)
    else:
        print('has already...')
    
def main():
    videos = os.listdir(video_root_path)
    for video in videos:
        video_directory = video_root_path+'/'+video
        computeDynamicImages_fromImages(video_directory)
        print(video_directory+'  has finished!')
    
main()
def generateDI_from_partial_Of_Frames(images_root_directory, save_root_path=None):
    if save_root_path == None:
        save_root_path = os.path.join(os.path.dirname(images_root_directory),os.path.basename(images_root_directory)+"-DI")
    if not os.path.isdir(images_root_directory):
        raise ValueError("dataset root path not exists")
    else:
        if not os.path.exists(save_root_path):
            os.makedirs(save_root_path)
            
    label_folders = os.listdir(images_root_directory)
    for label_folder in label_folders:
        if not os.path.exists(os.path.join(save_root_path, label_folder)):
            os.makedirs(os.path.join(save_root_path, label_folder))         
        videoname_folders = os.listdir(os.path.join(images_root_directory, label_folder))
        for videoname_folder in videoname_folders[:10]: 
#            for number in range(5):
            DI = computeDynamicImages_fromImages(os.path.join(images_root_directory, label_folder, videoname_folder))
            DI_file_name = str(videoname_folder.split('.')[0]) + '-DI.png'
            skimage.io.imsave(os.path.join(save_root_path, label_folder, DI_file_name),DI)
    
               

def find_wrong_shape_image(images_root_directory):
    def get_image_shape(images_dir):
        imageShape = []
        imageShape.append(N)
        img = cv2.imread(os.path.join(images_dir,os.listdir(images_dir)[0]))
        tmp = tuple(img.shape)
        imageShape = tuple(imageShape)
        imageShape = imageShape + tmp
        return imageShape
    label_folders = os.listdir(images_root_directory)#101 classes
    for label_folder in label_folders:
        image_folders = os.listdir(os.path.join(images_root_directory, label_folder))
        image_folders.pop()
        images_num = len(image_folders)
        for i in range(1,images_num+1):
            img_name=label_folder+'_'+str(i)+'.jpg'
            print(os.path.join(images_root_directory, label_folder, img_name))
            img = cv2.imread(os.path.join(images_root_directory, label_folder, img_name))
            if img.shape != (240,320,3):
                print(label_folder,img.shape)
                    
#find_wrong_shape_image(images_root_directory='/home/zhuc/pytorch/two-stream/data/UCF101/jpegs_256')
