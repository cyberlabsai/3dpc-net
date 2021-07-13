import torch
from torchvision import transforms
from tools.FASDataset import FASDataset
from trainer.FASTTester import FASTTester
import argparse
from utils.utils import read_cfg, get_device, build_network
import os
import numpy as np
from tqdm import tqdm
from tools.FASDataset_RAW import FASDataset_RAW
# from Train import find_mean, training_data_RAW


def testing_data_RAW(cfg):
    #train_transform = transforms.Compose([RandomGammaCorrection(max_gamma=cfg['dataset']['augmentation']['gamma_correction'][1],
    #                                      min_gamma=cfg['dataset']['augmentation']['gamma_correction'][0]),
    #                                      transforms.RandomResizedCrop(cfg['model']['input_size'][0]),
    #                                      transforms.RandomRotation(cfg['dataset']['augmentation']['rotation_range']),
    #                                      transforms.RandomHorizontalFlip(),

    train_transform = transforms.Compose ([transforms.Resize(cfg['model']['input_size']),
                                         transforms.ToTensor(),
                                         transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])])
    

    trainset = FASDataset_RAW(root_dir=cfg['dataset']['root'],
                          images_file=cfg['dataset']['test_images'],
                          transform=train_transform,
                          smoothing=cfg['train']['smoothing'])
    
    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=cfg['test']['batch_size'],
                                              shuffle=True,
                                              num_workers=2)
    return trainloader
    
def find_mean(trainloader):
    # Try to calculate max_d and pm per batch and then calculate the meanof batches
    pbar = tqdm(trainloader, total=len(trainloader), dynamic_ncols=True)
    x_max = []
    x_min = []
    y_max = []
    y_min = []
    z_max = []
    z_min = []
    
    all_x = []
    all_y = []
    all_z = []
    for _, point_map, _ in pbar:
        # Downsample 10k -> 2.5k ?
        max_val = point_map.max(2)[0]
        min_val = point_map.min(2)[0]
        
        # print(max_val.shape)
        
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
        
        all_x.append(point_map[:, 0])
        all_y.append(point_map[:, 1])
        all_z.append(point_map[:, 2])
        
    
    lx = torch.cat(x_max, axis=0).max()
    sx = torch.cat(x_min, axis=0).min()
    
    ly = torch.cat(y_max, axis=0).max()
    sy = torch.cat(y_min, axis=0).min()
    
    lz = torch.cat(z_max, axis=0).max()
    sz = torch.cat(z_min, axis=0).min()
    
    print(len(all_x))
    print(len(all_y))
    print(len(all_z))
    all_x = torch.cat(all_x, axis=0)
    all_y = torch.cat(all_y, axis=0)
    all_z = torch.cat(all_z, axis=0)
    
    print(f"Check shape: {all_z.shape}")
    
    
    max_d = max([lx - sx, ly - sy, lz - sz])
    pm = torch.tensor([((lx + sx)/2).item(), ((ly + sy)/2).item(),
                       ((lz + sz)/2).item()])
    pm = torch.tensor([torch.mean(all_x, dim=(0, 1)), torch.mean(all_y, dim=(0, 1)),
                       torch.mean(all_z, dim=(0, 1))])
    return max_d, np.array(torch.unsqueeze(pm, axis=1))

def test_data(cfg, max_d, pm):

    test_transform = transforms.Compose ([transforms.Resize(cfg['model']['input_size']),
                                         transforms.ToTensor(),
                                         transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])])
    

    testset = FASDataset(root_dir=cfg['dataset']['root'],
                          images_file=cfg['dataset']['test_images'],
                          transform=test_transform,
                          smoothing=False, 
                          max_d=max_d, pm=pm,
                          rot_crop=False)
    
    print(f"Testing on {len(testset)} images")
    
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=cfg['test']['batch_size'],
                                              shuffle=True,
                                              num_workers=cfg['dataset']['num_workers'],
                                              pin_memory=True)

    return testloader

def main():
    print("Starting testing 3DPC_NET anti-spoofing")

    print("Pytorch Version:" + torch.__version__)
    print("Cuda Version:" + torch.version.cuda)

    #parsing arguments----------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", type=str, help="Path where to load model to continue training")
    parser.add_argument("--exp_folder", type=str, default="experiments/exp_1", help="Experiment folder to where load files")
    parser.add_argument("--csv_path", type=str, default="Protocol1_results.csv", help="Path to where save csv results")
    args = parser.parse_args()
    #---------------------------------------------------

    print("Reading config file....")
    cfg_path = os.path.join(args.exp_folder, 'train_config.yaml')
    cfg = read_cfg(cfg_file=cfg_path)
    print ("ok")
    
    device = get_device(cfg)
    print('Using {} device'.format(device))

    print("Load Network...")
    network = build_network(cfg)
    print("ok")
    print("model:" + cfg['model']['base'])
    
    # Precisa carregar os valores de normalização utilizados no treino?
    #! Normalizar os dados com os valores achados no conjunto de teste?
    
    # print("Finding normalization values in training data...")
    # trainloader = testing_data_RAW(cfg)
    # max_d, pm = find_mean(trainloader)
    # print("Ok")
    
    # print("=========  Normalization values  =========")
    # # TODO: Save normalization values on experiments folder
    # print(f"Maximum distance: {max_d}")
    # print(f"Medium point: {pm}")
    
    # Remeber point cloud data is still summed by 0.5
    # Norm values for OULU Train Protocol 1 
    # max_d = torch.tensor(1598)
    max_d = 1598
    pm = np.array([[582.2], [736.4], [300.2]])
    # print(pm.shape)
    testloader = test_data(cfg, max_d=max_d, pm=pm)
    
    tester = FASTTester(cfg=cfg, network=network, device=device,
                        testloader=testloader, csv_path=args.csv_path,
                        exp_folder=args.exp_folder, model_path=args.load_model)
    
    tester.test()

    
if __name__=='__main__':
    main()
