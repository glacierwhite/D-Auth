import sys
if './' not in sys.path:
	sys.path.append('./')
     
import logging
import time
import os

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
	
from models.vision_transformer import VisionTransformer
from torch.utils.tensorboard import SummaryWriter
from options import Options
import torch
from running import setup, SupervisedRunner, NEG_METRICS, validate
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.loss import get_loss_module
from optimizers import get_optimizer
import utils

def main(config):

    total_epoch_time = 0
    total_eval_time = 0

    total_start_time = time.time()

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device.type == "cuda":
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Define transformations for preprocessing the images
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),   # resize images
        transforms.ToTensor(),           # convert PIL image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]) # normalize like ImageNet
    ])

    # Load dataset
    logger.info("Loading and preprocessing data ...")
    train_dataset = datasets.ImageFolder(root= config['data_dir']+"/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=config['data_dir']+"/val", transform=transform)

    logger.info("{} samples may be used for training".format(len(train_dataset)))
    logger.info("{} samples will be used for validation".format(len(val_dataset)))

    # Create model
    logger.info("Creating model ...")
    model = VisionTransformer(
            image_size=config['image_size'],
            patch_size=config['patch_size'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            hidden_dim=config['hidden_dim'],
            mlp_dim=config['mlp_dim'],
            num_classes=config['num_classes']
        )
    
    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))
    
    # Initialize optimizer

    if config['global_reg']:
        weight_decay = config['l2_reg']
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])
    optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`
    lr = config['lr']  # current learning step
    # Load model and optimizer state
    if args.load_model:
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_factor'])
    
    model.to(device)

    loss_module = get_loss_module(config)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=config['batch_size'],  # number of images per batch
                            shuffle=True,   # shuffle for training
                            drop_last=True,
                            num_workers=config['num_workers'])  # adjust for your CPU

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            drop_last=True,
                            num_workers=config['num_workers'])

    # Example: iterate over one batch
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break

    trainer = SupervisedRunner(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                                 print_interval=config['print_interval'], console=config['console'])
    
    val_evaluator = SupervisedRunner(model, val_loader, device, loss_module,
                                       print_interval=config['print_interval'], console=config['console'])
    
    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])

    best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf depending on key metric
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}

    # Evaluate on validation before training
    aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                          best_value, epoch=0)
    metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    metrics.append(list(metrics_values))

if __name__ == '__main__':
     
     args = Options().parse()
     config = setup(args)
     main(config)