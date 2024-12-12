import torch
import numpy as np
import argparse
from util import load_data_n_model
import torch.nn as nn
import os
from attack import pgd_attack

def temporal_shuffling(data, interval_size):
    """
    Perform temporal shuffling on CSI spectrogram data within specified intervals.
    
    Args:
        data: Tensor of shape (batch_size, channels, frequency, time)
        interval_size: Number of time frames in each shuffling interval
        
    Returns:
        Shuffled tensor of the same shape
    """
    # Create a copy of the input data
    shuffled_data = data.clone()
    
    # Get the dimensions
    batch_size, channels, freq_dim, time_dim = data.shape
    
    # Calculate number of complete intervals
    num_intervals = time_dim // interval_size
    
    # Process each sample in the batch
    for b in range(batch_size):
        for c in range(channels):
            # Process each complete interval
            for i in range(num_intervals):
                start_idx = i * interval_size
                end_idx = (i + 1) * interval_size
                
                # Get the current interval
                interval = shuffled_data[b, c, :, start_idx:end_idx]
                
                # Generate random permutation indices for time dimension
                perm_indices = torch.randperm(interval_size)
                
                # Shuffle the time frames within the interval
                shuffled_data[b, c, :, start_idx:end_idx] = interval[:, perm_indices]
                
            # Handle remaining frames if time_dim is not perfectly divisible by interval_size
            if time_dim % interval_size != 0:
                start_idx = num_intervals * interval_size
                remaining_frames = time_dim - start_idx
                
                # Shuffle remaining frames
                interval = shuffled_data[b, c, :, start_idx:]
                perm_indices = torch.randperm(remaining_frames)
                shuffled_data[b, c, :, start_idx:] = interval[:, perm_indices]
    
    return shuffled_data

# Example usage with the PGD attack defense
def defend_with_temporal_shuffling(model, data, device, interval_size=5):
    """
    Apply temporal shuffling defense before model inference
    
    Args:
        model: The neural network model
        data: Input data tensor
        labels: Ground truth labels
        criterion: Loss function
        device: Computing device (CPU/GPU)
        interval_size: Size of temporal shuffling intervals
        
    Returns:
        model outputs after applying temporal shuffling defense
    """
    # Apply temporal shuffling
    data = data.to(device)
    shuffled_data = temporal_shuffling(data, interval_size)
    # shuffled_data = shuffled_data.to(device)
    
    # Model inference
    outputs = model(shuffled_data)
    
    return outputs

# Modified PGD attack test function incorporating the defense
def pgd_test_with_defense(model, tensor_loader, criterion, device, interval_size=5):
    correct = 0
    total = 0
    
    model.train()
    for data, labels in tensor_loader:
        # Apply PGD attack
        adversarial_images = pgd_attack(model, data, labels, criterion, device)
        
        # Apply temporal shuffling defense
        with torch.no_grad():
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            outputs = defend_with_temporal_shuffling(
                model, adversarial_images, device, interval_size
            )
            
            _, pred = torch.max(outputs, 1)
            total += len(labels)
            correct += (pred == labels).sum()
    
    print('PGD Attack Accuracy with Temporal Shuffling Defense: {:.4f}'.format(float(correct) / total))

def natural_test_with_defense(model, tensor_loader, criterion, device, interval_size=5):
    """
    Evaluate model performance on clean data with temporal shuffling defense
    
    Args:
        model: The neural network model
        tensor_loader: DataLoader containing test data
        criterion: Loss function
        device: Computing device (CPU/GPU)
        interval_size: Size of temporal shuffling intervals
        
    Returns:
        tuple of (accuracy, average loss)
    """
    correct = 0
    total = 0
    total_loss = 0.0
    
    model.eval()
    with torch.no_grad():
        for data, labels in tensor_loader:
            labels = labels.type(torch.LongTensor)
            data = data.to(device)
            labels = labels.to(device)
            
            # Apply temporal shuffling defense
            shuffled_data = temporal_shuffling(data, interval_size)
            
            # Forward pass
            outputs = model(shuffled_data)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Accumulate loss
            total_loss += loss.item()
    
    # Calculate average accuracy and loss
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(tensor_loader)
    
    print(f'Natural Test Accuracy with Defense: {accuracy:.2f}%')
    print(f'Average Loss with Defense: {avg_loss:.4f}')
    

# Example usage showing both natural and adversarial testing
def evaluate_model_with_defense(model, test_loader, criterion, device, interval_size=5):
    """
    Comprehensive evaluation of the model with temporal shuffling defense
    
    Args:
        model: The neural network model
        test_loader: DataLoader containing test data
        criterion: Loss function
        device: Computing device (CPU/GPU)
        interval_size: Size of temporal shuffling intervals
    """
    print("=== Evaluating Model with Temporal Shuffling Defense ===")
    print(f"Interval size: {interval_size}")
    print("\n1. Natural Test Performance:")
    natural_test_with_defense(
        model, test_loader, criterion, device, interval_size
    )
    
    print("\n2. PGD Attack Performance:")
    pgd_test_with_defense(
        model, test_loader, criterion, device, interval_size
    )


if __name__=='__main__':
    root = './Data/'
    model_save_pth = './models/'

    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices = ['UT_HAR_data','NTU-Fi-HumanID','NTU-Fi_HAR','Widar'])
    parser.add_argument('--model', choices = ['MLP','LeNet','ResNet18','ResNet50','ResNet101','RNN','GRU','LSTM','BiLSTM', 'CNN+GRU','ViT'])
    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model

    train_loader, test_loader, model, train_epoch = load_data_n_model(dataset_name, model_name, root)
    model.load_state_dict(torch.load(os.path.join(model_save_pth, f'{dataset_name}_{model_name}.pth'), map_location=torch.device('cpu') ))

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    evaluate_model_with_defense(model, test_loader, criterion, device, interval_size=20)

