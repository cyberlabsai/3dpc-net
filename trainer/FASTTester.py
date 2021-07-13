import csv
import os
import pdb
from random import randint
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
from trainer.base import BaseTrainer
from utils.meters import AvgMeter
from utils.eval import predict, calc_accuracy
from pytorch3d.loss import chamfer_distance
import csv
import numpy as np



class FASTTester(BaseTrainer):
    def __init__(self, cfg, network, device, testloader, csv_path, exp_folder,
                 model_path=''):
        
        # Initialize variables
        self.cfg = cfg
        self.network = network
        self.device = device
        self.testloader = testloader
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.network = torch.nn.DataParallel(self.network)
        
        self.network = self.network.to(device)

        # ! AvgMeter not using tensorboard writer
        self.test_acc_metric = AvgMeter(writer='', 
                                         name='Accuracy/train', num_iter_per_epoch=len(self.testloader),
                                         per_iter_vis=True)
        
        

        self.load_model(model_path)
        print(f"Model {model_path} loaded!")
        
        self.exp_folder = exp_folder
        
        self.csv_path = csv_path
            

    def load_model(self, model_path):
        
        print(model_path)

        state = torch.load(model_path)

        self.network.load_state_dict(state['state_dict'])
        self.network.eval()
        
    def save_csv(self, filename, label, score):
        dataset_info = self.cfg['dataset']['name'] + '_' + self.csv_path
        save_path = os.path.join(self.exp_folder, dataset_info)
        label_dict = {True: 'Live', False: 'Spoof'}
        # open the file in the write mode
        with open(save_path, 'w') as f:
            # create the csv writer
            writer = csv.writer(f, delimiter=',')
            
            
            for i in range(len(filename)):
                # write a row to the csv file
                writer.writerow([filename[i], label_dict[label[i]], score[i]])
        
    
    def test(self):
        self.test_acc_metric.reset(0)

        pbar = tqdm(self.testloader, total=len(self.testloader), dynamic_ncols=True)
        
        # Open CSV file
        # dataset_info = self.cfg['dataset']['name'] + '_' + self.csv_path
        # save_path = os.path.join(self.exp_folder, dataset_info)
        # csv_file = open(save_path, 'w')
        # writer = csv.writer(csv_file, delimiter=',')
        filenames = []
        labels = []
        scores = []
        
        for i, (img, point_map, label, filename) in enumerate(pbar):
            # if i >= 100:
            #     break
            
            #! Point cloud maps don't come normalized in test
            img, point_map, label = img.to(self.device), point_map.to(self.device), label.to(self.device)
            net_point_map = self.network(img)
            
            # point_map[label > 0, 2] = point_map[label > 0, 2] + 0.1
            # point_map[label == 0, 2] = point_map[label == 0, 2] - 0.1

            # print(filename)
            # print("==========  [DEBUG]  ==========")
            cond = label > 0
            # print(torch.mean(point_map[cond], dim=(0, 2)))
            # print(torch.amax(point_map[cond], dim=(0, 2)))
            # print(torch.amin(point_map[cond], dim=(0, 2)))
            # print("==========  [DEBUG]  ==========")
            preds, score = predict(net_point_map)
            
            # print('-'*50)
            # print(score[cond])
            # preds, score = predict(net_point_map)

            # print(score)
            # print(label)
            accuracy = calc_accuracy(preds, label)

            # Update metrics
            self.test_acc_metric.update(accuracy)
            
            pbar.set_description(f"Test - Acc: {self.test_acc_metric.avg:.4f}")
            
            filenames.append(filename)
            labels.append(label.to('cpu'))
            scores.append(score.to('cpu'))
        
        Acc = self.test_acc_metric.avg
        
        print(f'\nTest Acc: {Acc}\n')
        
        print("Saving CSV...")
        filenames = np.concatenate(filenames, axis=0)
        labels = np.concatenate(labels, axis=0)
        scores = np.concatenate(scores, axis=0)
        
        self.save_csv(filenames, labels, scores)
                



