import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from model import VGG11
from trainer import train_model, test_model

def setup(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def get_dataloaders(rank, world_size, batch_size, is_ddp):
    """Creates the training and test dataloaders."""
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Download only on the main process
    if rank == 0:
        datasets.CIFAR10(root="./data", train=True, download=True)
        datasets.CIFAR10(root="./data", train=False, download=True)
    
    # Wait for the main process to finish downloading
    if is_ddp:
        dist.barrier()

    training_set = datasets.CIFAR10(root="./data", train=True, transform=transform_train)
    test_set = datasets.CIFAR10(root="./data", train=False, transform=transform_test)

    train_sampler = None
    if is_ddp:
        train_sampler = DistributedSampler(training_set, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(training_set,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              shuffle=(train_sampler is None),
                              num_workers=4, # Can be tuned
                              pin_memory=True)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)
    
    return train_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description="PyTorch DDP CIFAR-10 Training")
    parser.add_argument('--epochs', default=15, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size per GPU')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    args = parser.parse_args()

    # DDP environment variables
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_ddp = world_size > 1

    if is_ddp:
        setup(rank, world_size)
        print(f"DDP setup on rank {rank} of {world_size}.")

    # Device
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    
    # Model
    model = VGG11().to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    # DataLoaders
    train_loader, test_loader = get_dataloaders(rank, world_size, args.batch_size, is_ddp)

    # Optimizer and Loss
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(args.epochs):
        if rank == 0:
            print(f"--- Starting Epoch {epoch+1}/{args.epochs} ---")
        train_model(rank, model, train_loader, optimizer, criterion, epoch)
        # Testing is typically done on one process to avoid redundancy
        if rank == 0:
            test_model(rank, model, test_loader, criterion)

    if is_ddp:
        cleanup()

if __name__ == '__main__':
    main()
