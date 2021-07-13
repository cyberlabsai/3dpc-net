import sys
sys.path.append(".") # Adds higher directory to python modules path.
import os
import torch
from torchvision import transforms
from utils.utils import read_cfg
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import random 


class FASDataset_RAW(Dataset):
    ''' Dataloader for face PAD
    Args:
        root_dir (str): Root directory path
        txt_file (str): txt file to dataset annotation
        depth_map_size (int): Size of pixel-wise binary supervision map
        transform: takes in a sample and returns a transformed version
        smoothing (bool): use label smoothing
        num_frames (int): num frames per video per epoch
    '''
    def __init__(self, root_dir, images_file, transform, smoothing, num_frames=3):
        super().__init__()
        self.root_dir = root_dir
        self.img_list, self.point_cloud_list, self.labels = self.load_image_data(os.path.join(self.root_dir, images_file))
        self.transform = transform

        self.filenames = list(self.img_list.keys())

        self.num_frames = num_frames

        if smoothing:
            self.label_weight = 1.0
        else:
            self.label_weight = 0.99
            
    #=================================================================
    def load_image_data(self, path_to_train_images):
        img_paths = {}
        point_cloud_paths = {}
        labels = {}
        
        #check if folder is empty
        if (len(os.listdir(path_to_train_images))==0):
            print("Error: " + path_to_train_images + " is empty!!")
            sys.exit(0)

        # loop for every file 
        for file in os.listdir(path_to_train_images):
            filePath = os.path.join(path_to_train_images, file)
            
            filename, file_extension = os.path.splitext(file)
            if(file_extension == '.jpg'):

                if filename not in img_paths:         img_paths[filename] = []
                if filename not in labels:            labels[filename] = []

                img_paths[filename].append(filePath)
                label=self.get_label(filename)
                labels[filename].append(label)

            elif (file_extension == '.npy'):
                if filename not in point_cloud_paths: point_cloud_paths[filename] = []
                point_cloud_paths[filename].append(filePath)

            """
            for img in os.listdir(filePath):
                img_name = img.split('.jpg')[0]
                label = img_name.split('_')[0]

                #PC_name=

                if label == 'live': label = 1
                else:   label = 0

                depthPath = os.path.join(path_to_depth_images, file)
                depthPath = os.path.join(depthPath, img)

                imgPath = os.path.join(filePath, img)
                img_paths[file].append(imgPath)
                labels[file].append(label)
                depth_img_paths[file].append(depthPath)
            """

        return img_paths, point_cloud_paths, labels
    #=============================================================
    def get_label(self, string_input): # OULU protocol 1
        label=int(string_input[7])
        if label == 1 : return True #live
        else: return False # Spoof
    
    #=============================================================
    def __getitem__(self, index):
        ''' Get image, output map and label for a given index
        Args:
            index (int): index of image
        Returns:
             img (PIL image)
             mask: output map (32x32)
             label: 1 (live), 0 (spoof)
        '''
        #! Criar toda iteraçao uma nova lista com o tamanho de len(self.ids) * num_frames e pegar um item dessa lista
        #! Ou buscar dentro do id o numero de num_frames necessario
        #index = index % len(self.filenames)
        filename = self.filenames[index]
        
        imgPath = self.img_list[filename][0]                #[file_index]
        PointCloudPath = self.point_cloud_list[filename][0] #[file_index]
        label = self.labels[filename][0]                   #[file_index]


        img = Image.open(imgPath) # read image

        point_cloud_map = np.load(PointCloudPath) #! read pointcloud 10k points

        n_points = point_cloud_map
        rot_img_crop = img


        if self.transform:
            img = self.transform(rot_img_crop)
            n_points = transforms.ToTensor()(n_points.astype(np.float32)).squeeze()

        return img, n_points, label
    #=============================================================
    def __len__(self):
        return len(self.filenames) #* self.num_frames
        # return len(self.ids) #* self.num_frames
        
    #=============================================================
    def downsample(self,vertices, n_samples=2500): #Downsample point cloud vertices randomly
        vertices_df = pd.DataFrame(vertices.transpose((1, 0)))

        samp= vertices_df.shape[0]

        if samp >= n_samples:
            vertices_downsampled = vertices_df.sample(n=n_samples, replace=False)
        else :
            vertices_downsampled = vertices_df.sample(n=n_samples, replace=True)

        vertices_downsampled = np.array(vertices_downsampled).transpose((1, 0))
        #vertices_downsampled = np.array(vertices_downsampled)
        return vertices_downsampled
    
#=================================================================
"""
if __name__=="__main__":
    
    cfg=read_cfg(cfg_file="../config/DPC3_NET_config.yaml")
    root_path= '../'
    images_path=cfg['dataset']['train_images']

    train_transform = transforms.Compose ([transforms.Resize(cfg['model']['input_size']),
                                         transforms.ToTensor(),
                                         transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])])
                                         #TODO: Check Normalization

    dt=FASDataset(root_dir=root_path,
                  images_file=images_path,
                  transform=train_transform,
                  smoothing=True)


    trainloader = torch.utils.data.DataLoader(dataset=dt,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=1)

    for i, (img, point_map, label) in enumerate(trainloader):
        print(img.size())
        print("i: " + str(i))
        #print(point_map)
        #print(label)
"""



    