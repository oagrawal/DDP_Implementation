import time
import torch
from tqdm import tqdm

def train_model(rank, model, train_loader, optimizer, criterion, epoch):
    """
    Train the model for one epoch.
    """
    model.train()
    start_time = time.time()
    
    # In DDP, DistributedSampler handles shuffling, so we need to set the epoch
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, disable=(rank != 0))):
        if torch.cuda.is_available():
            data, target = data.to(rank), target.to(rank)

        # Forward pass
        pred = model(data)
        loss = criterion(pred, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0 and rank == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
    
    if rank == 0:
        total_time = time.time() - start_time
        print(f"Epoch {epoch} training time: {total_time:.2f} seconds")


def test_model(rank, model, test_loader, criterion):
    """
    Test the model.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, disable=(rank != 0)):
            if torch.cuda.is_available():
                data, target = data.to(rank), target.to(rank)
            
            pred = model(data)
            test_loss += criterion(pred, target).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
            total_samples += len(data)
    
    if rank == 0:
        avg_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total_samples
        print(f'\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)\n')