#libraries
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.models as models
#import matplotlib.pyplot as plt
import random

#from tools.resnet import ResNet18 # resnet18 networt

#=========================================
class DPC3_NET(nn.Module):
    def __init__(self):
        super(DPC3_NET,self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(resnet18.conv1,
                                      resnet18.bn1,
                                      resnet18.relu,
                                      resnet18.maxpool,
                                      resnet18.layer1,
                                      resnet18.layer2,
                                      resnet18.layer3,
                                      resnet18.layer4,
                                      resnet18.avgpool) # output: 1x512
        
        self.flatten=nn.Flatten() 
        self.FC1=nn.Linear(512,256) # TODO:test other output sizes!!!!!!!!!!!!!!!!
        self.decoder=nn.Sequential(nn.Conv1d(258,129,kernel_size=1), nn.ReLU(), nn.Conv1d(129,3,kernel_size=1), nn.Tanh()) #TODO: check padding!!
    #------------------------------
    def forward(self,x):
        device = x.get_device()
        x=self.backbone(x)
        x=self.flatten(x)
        x=self.FC1(x)

        #concatenate embedings to pointcloud
        #print(x.shape)
        x_2d_tensor=[]
        for vec in x.tolist():
            x_2d=self.sampling_unit_square(vec)
            x_2d_tensor.append(x_2d)
           
        x_2d_tensor=torch.tensor(x_2d_tensor)
        # print(f"[DEBUG] X 2D Tensor: {x_2d_tensor.size()}")
        x_2d_tensor=x_2d_tensor.permute(0, 2, 1) #change positions on tensor
        x_2d_tensor=x_2d_tensor.to(device)
        # print(f"[DEBUG] Before decoder: {x_2d_tensor.size()}")

        y=self.decoder(x_2d_tensor)
        # print(f"Model out shape: {y.size()} - {y.device}")
        return y 
    #------------------------------
    def sampling_unit_square(self,vec1D,num_samples=2500):
        vec2d=[]
        for i in range(num_samples):
            coord=[random.uniform(0,1),random.uniform(0,1)] 
            aux_vec=vec1D+coord
            vec2d.append(aux_vec)           
        return vec2d
#============================================         
"""
if __name__=="__main__":

    model=DPC3_NET()
    vec1D=[0,0] # enbedings
    vec2d=model.sampling_unit_square(vec1D,10)
    vec2d=np.array(vec2d)
    print(vec2d.shape)
    print(vec2d)
"""



