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


class FASDataset(Dataset):
    ''' Dataloader for face PAD
    Args:
        root_dir (str): Root directory path
        txt_file (str): txt file to dataset annotation
        depth_map_size (int): Size of pixel-wise binary supervision map
        transform: takes in a sample and returns a transformed version
        smoothing (bool): use label smoothing
        num_frames (int): num frames per video per epoch
    '''
    def __init__(self, root_dir, images_file, transform, smoothing, 
                 max_d, pm, rot_crop=True, split_validation=0, num_frames=3):
        super().__init__()
        self.root_dir = root_dir
        self.img_list, self.point_cloud_list, self.labels = self.load_image_data(os.path.join(self.root_dir, images_file))
        self.transform = transform

        # print(self.img_list)
        
        #print("keys: "  + str(self.img_list.keys()))
        self.filenames = list(self.img_list.keys())
        # print(self.filenames)
        # print(len(self.filenames))
        
        # Remove frame number from filename and gets only the person id.
        self.ids = [file[:file.rfind('_')] for file in self.filenames]
        # print(len(self.ids))
        # Filter duplicated ids
        self.ids = list(dict.fromkeys(self.ids))
        
        if split_validation != 0:
            # val_frac = 0.4
            frames_per_id = {}
            for id in self.ids:
                frames_per_id[id] = []
                for file in self.filenames:
                    if id in file:
                        frames_per_id[id].append(file)
                        
            for id in list(frames_per_id.keys()):
                frames = frames_per_id[id]
                # print(len(frames))
                new_len = int(len(frames) * split_validation)
                random.shuffle(frames)
                for frame in frames:
                    if len(frames) <= new_len:
                        break
                    
                    frames.remove(frame)
                    self.img_list.pop(frame)
                    self.point_cloud_list.pop(frame)
                    self.labels.pop(frame)
                    
            print(f"Original size data: {len(self.filenames)}")
            self.filenames = list(self.img_list.keys())
            

        self.num_frames = num_frames

        if smoothing:
            self.label_weight = 1.0
        else:
            self.label_weight = 0.99
            
        self.max_d = max_d
        self.pm = pm
        
        self.rot_crop = rot_crop
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
    
    def normalize_cloud2(self, points):
        # Points shape = (3, 10k)
        l = points.max(axis=1)
        s = points.min(axis=1)
        
        lx = l[0]
        ly = l[1]
        lz = l[2]
        
        sx = s[0]
        sy = s[1]
        sz = s[2]
        
        max_d = max([lx - sx, ly - sy, lz - sz])
        pm = np.array([[(lx+sx)/2], [(ly+sy)/2], [(lz+sz)/2]])
        
        return (points - pm)/(2*max_d) + 0.5
    
    def normalize_cloud(self, points):
        # Formula got from 3DPC-Net paper
        points_normalized = (points - self.pm)/(2*self.max_d) + 0.5
        return points_normalized

    def rot_pointcloud(self,pointcloud,image, filename, crop_info):  # Random rotation of a PointCloud in Z axis
        try:
            rotated_data = np.zeros(pointcloud.shape, dtype=np.float32)
            #pointcloud= np.transpose(pointcloud)
            rotation_angle = np.random.uniform(low=-1.0,high=1.0) * np.pi/6 
            cosval = np.cos(rotation_angle) 
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, -sinval, 0],
                                        [sinval, cosval, 0],
                                        [0, 0, 1]])
            #print(rotation_angle*(180/np.pi))

            rotated_img=image.rotate(rotation_angle*(180/np.pi))
            rotated_data = np.dot(pointcloud, rotation_matrix)
            rotated_data=np.transpose(rotated_data)

            return rotated_data, rotated_img
            
        except ValueError as E:
            print("ERROR!!!!!!!!!")
            print(filename)
            print(E)
            print(f"{pointcloud.shape} : {rotation_matrix.shape}")
            print(image.size)
            print(rotation_angle)



            print(f"[CROP INFO] dy_img: {crop_info[0]}")
            print(f"[CROP INFO] dx_img: {crop_info[1]}")
            print(f"[CROP INFO] new_iw: {crop_info[2]}")
            print(f"[CROP INFO] new_ih: {crop_info[3]}")
            print(f"[CROP INFO] sx_crop: {crop_info[4]}")
            print(f"[CROP INFO] lx_crop: {crop_info[5]}")
            print(f"[CROP INFO] sy_crop: {crop_info[6]}")
            print(f"[CROP INFO] ly_crop: {crop_info[7]}")
            print(f"[CROP INFO] image w: {crop_info[8]}")
            print(f"[CROP INFO] image h: {crop_info[9]}")

            exit()
    #=============================================================
    def uniform_crop(self, img, points):
        iw = img.size[0]
        ih = img.size[1]
    
        dy_img = random.randint(0, 90) # 90 parece ser um bom limite empírico muito justo
        dx_img = random.randint(0, 50) # 50 parece ser um bom limite empírico muito justo
        new_iw = iw - dx_img
        new_ih = ih - dy_img
        
        #img_npy = np.array(img)

        #print(str(dy_img) + " : " + str(new_ih) + "  " + str(dx_img) + " : " + str(new_iw))
        
        #cropped_img = img_npy[dy_img:new_ih, dx_img:new_iw,:].copy()
        cropped_img = img.crop((dx_img,dy_img,new_iw,new_ih))

        #Point Cloud crop
        lx = points[0, :].max() # 0 for x axis
        sx = points[0, :].min()

        ly = points[1, :].max() # 1 for y axis
        sy = points[1, :].min()

        dx = (lx - sx)/iw * dx_img
        dy = (ly - sy)/ih * dy_img

        sx_crop = sx + dx
        lx_crop = lx - dx
        sy_crop = sy + dy
        ly_crop = ly - dy
        
        new_points = []
        for i in range(points.shape[1]):
            point = points[:, i]
            if point[0] < lx_crop and point[0] > sx_crop:
                if point[1] < ly_crop and point[1] > sy_crop:
                    new_points.append(point)
                    

        crop_info=[dy_img, dx_img, new_iw, new_ih, sx_crop, lx_crop, sy_crop, ly_crop, iw,ih]
        return np.array(new_points), cropped_img, crop_info  #Image.fromarray(cropped_img)
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
        
        # print(filename)
        
        #file_index = random.randint(0, len(self.img_list[filename]) - 1)
        
        imgPath = self.img_list[filename][0]                #[file_index]
        PointCloudPath = self.point_cloud_list[filename][0] #[file_index]
        label = self.labels[filename][0]                   #[file_index]

        img = Image.open(imgPath) # read image

        cloud_map = np.load(PointCloudPath) #! read point cloud 10k points

        #! Data aug
        if self.rot_crop:
            #Crop image and PointCloud
            cloud_map, img, crop_info = self.uniform_crop(img, cloud_map)

            #Rotate image and PointCloud
            cloud_map, img = self.rot_pointcloud(cloud_map, img, imgPath, crop_info)
 
        # cloud_map_normalized = self.normalize_cloud(cloud_map)
        cloud_map_normalized = cloud_map
        cloud_map_normalized_sampled = self.downsample(cloud_map_normalized) # point cloud downsample to 2.5K points
        cloud_map_normalized_sampled = self.normalize_cloud2(cloud_map_normalized_sampled)

        if self.transform:
            img = self.transform(img)
            cloud_map_normalized_sampled = transforms.ToTensor()(cloud_map_normalized_sampled.astype(np.float32)).squeeze()

        return img, cloud_map_normalized_sampled, label, filename
    #=============================================================
    def __len__(self):
        return len(self.filenames) #* self.num_frames
        # return len(self.ids) #* self.num_frames
    #=============================================================
    def downsample(self, vertices, n_samples=2500):
        #Downsample point cloud vertices randomly
        vertices_df = pd.DataFrame(vertices.transpose(1, 0))

        samp= vertices_df.shape[0]

        if samp >= n_samples:
            vertices_downsampled = vertices_df.sample(n=n_samples, replace=False)
        else :
            vertices_downsampled = vertices_df.sample(n=n_samples, replace=True)

        vertices_downsampled = np.array(vertices_downsampled).transpose((1, 0))
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



    