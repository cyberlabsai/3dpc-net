class BaseTrainer():
    def __init__(self, cfg, network, optimizer, lr_scheduler, device, trainloader, valloader, writer):
        self.cfg = cfg
        self.network = network
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.trainloader = trainloader
        self.valloader = valloader
        self.writer = writer
