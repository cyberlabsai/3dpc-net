#Title: Train.py
#Authors: Fischer ; Thimabru @ cyberlabs
#Date: 5/2021
#Usage: python3 train.py --dp <dataset folder>
import torch
from tools.FASDataset import FASDataset
from tools.FASDataset_RAW import FASDataset_RAW
from trainer.FASTTrainer import FASTrainer
from utils.utils import read_cfg, save_cfg, get_optimizer, get_device, build_network
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import os
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def training_data_RAW(cfg):
    train_transform = transforms.Compose ([transforms.Resize(cfg['model']['input_size']),
                                         transforms.ToTensor(),
                                         transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])])
    

    trainset = FASDataset_RAW(root_dir=cfg['dataset']['root'],
                          # images_file=cfg['dataset']['train_images'],
                          images_file=cfg['dataset']['train_images'],
                          transform=train_transform,
                          smoothing=cfg['train']['smoothing'])
    
    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=cfg['train']['batch_size'],
                                              shuffle=True,
                                              num_workers=2)

    return trainloader

def training_data(cfg, max_d, pm):
    train_transform = transforms.Compose ([transforms.Resize(cfg['model']['input_size']),
                                         transforms.ToTensor(),
                                         transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])])
    

    trainset = FASDataset(root_dir=cfg['dataset']['root'],
                          images_file=cfg['dataset']['train_images'],
                          transform=train_transform,
                          smoothing=cfg['train']['smoothing'], 
                          max_d=max_d, pm=pm,
                          rot_crop=cfg['dataset']['augmentation']['rot_crop'])
    
    print(f"Training on {len(trainset)} images")
    
    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=cfg['train']['batch_size'],
                                              shuffle=True,
                                              num_workers=cfg['dataset']['num_workers'],
                                              pin_memory=True)

    return trainloader

def val_data(cfg, max_d, pm):
    val_transform = transforms.Compose([transforms.Resize(cfg['model']['input_size']),
                                        transforms.ToTensor(),
                                        transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])]) 
    
    valset = FASDataset(root_dir=cfg['dataset']['root'],
                        images_file=cfg['dataset']['val_images'],
                        transform=val_transform,
                        smoothing=cfg['train']['smoothing'], 
                        max_d=max_d, pm=pm,
                        rot_crop=False,
                        split_validation=0.4)
    
    print(f"Validating on {len(valset)} images")
    
    valloader = torch.utils.data.DataLoader(dataset=valset,
                                            batch_size=cfg['val']['batch_size'],
                                            shuffle=True,
                                            num_workers=cfg['dataset']['num_workers'],
                                            pin_memory=True)
    
    return valloader

def find_mean(trainloader):
    pbar = tqdm(trainloader, total=len(trainloader), dynamic_ncols=True)
    x_max = []
    x_min = []
    y_max = []
    y_min = []
    z_max = []
    z_min = []
    for _, point_map, _ in pbar:
        max_val = point_map.max(2)[0]
        min_val = point_map.min(2)[0]
        
        
        lx = max_val[:, 0]
        sx = min_val[:, 0]
        
        ly = max_val[:, 1]
        sy = min_val[:, 1]
        
        lz = max_val[:, 2]
        sz = min_val[:, 2]
        
        x_max.append(lx)
        x_min.append(sx)
        
        y_max.append(ly)
        y_min.append(sy)
        
        z_max.append(lz)
        z_min.append(sz)
    
    lx = torch.cat(x_max, axis=0).max()
    sx = torch.cat(x_min, axis=0).min()
    
    ly = torch.cat(y_max, axis=0).max()
    sy = torch.cat(y_min, axis=0).min()
    
    lz = torch.cat(z_max, axis=0).max()
    sz = torch.cat(z_min, axis=0).min()
    
    max_d = max([lx - sx, ly - sy, lz - sz])
    pm = torch.tensor([((lx + sx)/2).item(), ((ly + sy)/2).item(),
                       ((lz + sz)/2).item()])
    return max_d, np.array(torch.unsqueeze(pm, axis=1))
    
#=========================================
#=================MAIN====================
#=========================================
def main():
    print("Starting training 3DPC_NET anti-spoofing")

    print("Pytorch Version:" + torch.__version__)
    print("Cuda Version:" + torch.version.cuda)

    #parsing arguments----------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", type=str, help="Path where to load model to continue training")
    parser.add_argument("--logs_path", type=str, default="experiments", help="Path where to save logs")
    parser.add_argument("--exp_name", type=str, default="exp_1", help="Name of experiment folder to save logs")
    parser.add_argument("--cfg", type=str, default="config/DPC3_NET_config.yaml", help="Path to config(.yaml) file")

    args = parser.parse_args()
    #---------------------------------------------------

    print("Reading config file....")
    cfg=read_cfg(cfg_file=args.cfg)
    print ("ok")

    device = get_device(cfg)
    print('Using {} device'.format(device))

    print("Load Network...")
    network=build_network(cfg)
    print("ok")
    print("model:" + cfg['model']['base'])

    print("Load Optimizer...")
    optimizer=get_optimizer(cfg,network)
    print("ok")
    print("optimizer:" + cfg['train']['optimizer'])

    print("Loss Funtion: Chamfer distance")
    
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.8)

    print("Finding normalization values in training data...")
    trainloader=training_data_RAW(cfg)
    max_d, pm = find_mean(trainloader)
    print("Ok")
    
    print("=========  Normalization values  =========")
    # TODO: Save normalization values on experiments folder
    print(f"Maximum distance: {max_d}")
    print(f"Medium point: {pm}")
    
    print("Load training data...")
    trainloader=training_data(cfg, max_d, pm)
    print("Ok")

    print("Load validation data...")
    valloader=val_data(cfg, max_d, pm)
    print("Ok")

    logs_full_path = os.path.join(args.logs_path, args.exp_name)
    if not os.path.exists(logs_full_path):
        os.makedirs(logs_full_path)
    
    save_cfg(cfg, os.path.join(logs_full_path, "train_config.yaml"))
    
    print("Starting TensorBoard.....")
    
    writer = SummaryWriter(f'{logs_full_path}/Tensoboard_logs')
    print("Ok")

    trainer= FASTrainer(cfg=cfg,
                        network=network,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        device=device,
                        trainloader=trainloader,
                        valloader=valloader,
                        writer=writer,
                        logs_path=logs_full_path,
                        model_path=args.load_model)
    
    print("Starting training...")
    trainer.train()
    print("Finish Training")
    writer.close()
    
#==============================================
if __name__=='__main__':
    main()
