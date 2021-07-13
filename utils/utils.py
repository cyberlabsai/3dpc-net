import yaml
import torch
from torch import optim
from tools.DPC3_NET import DPC3_NET

def read_cfg(cfg_file):
    """
    Read configurations from yaml file
    Args:
        cfg_file (.yaml): path to cfg yaml
    Returns:
        (dict): configuration in dict
    """
    with open(cfg_file, 'r') as rf:
        cfg = yaml.safe_load(rf)
        return cfg
    
def save_cfg(cfg_dict, save_path):
    '''
        Save cfg in yaml file
    '''
    with open(save_path, 'w') as yaml_file:
        yaml.dump(cfg_dict, yaml_file, default_flow_style=False)

#====================================================
def get_optimizer(cfg, network):
    """ Get optimizer based on the configuration
    Args:
        cfg (dict): a dict of configuration
        network: network to optimize
    Returns:
        optimizer 
    """
    optimizer = None
    if cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'])
    else:
        raise NotImplementedError

    return optimizer
#=====================================================
def get_device(cfg):
    """ Get device based on configuration
    Args: 
        cfg (dict): a dict of configuration
    Returns:
        torch.device
    """
    device = None
    if len(cfg['device']) <= 1:
        if cfg['device'] == []:
            device = torch.device("cpu")
        elif cfg['device'][0] == 0:
            device = torch.device("cuda:0")
        elif cfg['device'][0] == 1:
            device = torch.device("cuda:1")
    elif len(cfg['device']) > 1:
        # device = [torch.device(f"cuda:{cfg['device'][0]}"), torch.device(f"cuda:{cfg['device'][1]}")]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise NotImplementedError
    # print(f"GPUs found: {device}")
    return device
#====================================================
def build_network(cfg):
    """ Build the network based on the cfg
    Args:
        cfg (dict): a dict of configuration
    Returns:
        network (nn.Module) 
    """
    network = None

    if cfg['model']['base'] == 'DPC3-NET':
        network = DPC3_NET()
    else:
        raise NotImplementedError

    return network
