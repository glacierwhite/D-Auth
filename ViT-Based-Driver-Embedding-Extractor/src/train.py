import sys
if './' not in sys.path:
	sys.path.append('./')
	
from models.vision_transformer import VisionTransformer
from options import Options
import torch
from running import setup
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def main(config):

    # Define the ViT model
    model = VisionTransformer(
            image_size=config['image_size'],
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=5
        )

    # # Define transformations for preprocessing the images
    # transform = transforms.Compose([
    #     transforms.Resize((config['iamge_size'], config['iamge_size'])),   # resize images
    #     transforms.ToTensor(),           # convert PIL image to PyTorch tensor
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                         std=[0.229, 0.224, 0.225]) # normalize like ImageNet
    # ])

    # # Load dataset
    # train_dataset = datasets.ImageFolder(root="data/train", transform=transform)
    # val_dataset = datasets.ImageFolder(root="data/val", transform=transform)

    # # Create DataLoaders
    # train_loader = DataLoader(dataset=train_dataset,
    #                         batch_size=32,  # number of images per batch
    #                         shuffle=True,   # shuffle for training
    #                         num_workers=4)  # adjust for your CPU

    # val_loader = DataLoader(dataset=val_dataset,
    #                         batch_size=32,
    #                         shuffle=False,
    #                         num_workers=4)

    # # Example: iterate over one batch
    # for images, labels in train_loader:
    #     print(f"Image batch shape: {images.shape}")
    #     print(f"Label batch shape: {labels.shape}")
    #     break

if __name__ == '__main__':
     
     args = Options().parse()
     config = setup(args)
     print(config)
     main(config)