import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
import os
from attack import pgd_attack
import torch.nn as nn
from util import load_data_n_model

# Please use below function to generate adversarial examples, and evaluate the detector model
# pgd_attack(model, data, labels, criterion, device, eps=64/255, alpha=2/255, iters=20)

def get_penultimate_layer_features(model, device):
    """
    Registers a forward hook to capture the input to the last layer (penultimate features).
    Returns the hook handle and a feature storage dictionary.
    """
    features = {}

    def hook(module, input, output):
        features['penultimate'] = input[0].detach().to(device)

    # Identify the last linear (classification) layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handle = module.register_forward_hook(hook)
            break

    return handle, features


def train_anomaly_detector(model, clean_loader, device):
    latent_representations = []

    model.train()
    # model.eval()

    handle, features = get_penultimate_layer_features(model, device)
    with torch.no_grad():
        for images, _ in clean_loader:
            images = images.to(device)
            # Extract features from an intermediate layer
            model(images)  
            clean_features = features['penultimate']
            latent_representations.append(clean_features.cpu().numpy())
    
    # Flatten and standardize representations
    latent_representations = np.concatenate(latent_representations, axis=0)
    scaler = StandardScaler().fit(latent_representations)
    latent_representations = scaler.transform(latent_representations)
    
    # Train an outlier detection model
    detector = IsolationForest(contamination=0.1, random_state=42)
    detector.fit(latent_representations)
    
    return detector, scaler

def detect_adversarial(detector, scaler, model, test_images, device):
    # model.eval()
    model.train()
    handle, features = get_penultimate_layer_features(model, device)
    with torch.no_grad():
        test_images = test_images.to(device)
        model(test_images)  # Replace `.features` with appropriate layer
        features = scaler.transform(features['penultimate'].cpu().numpy())
        
        # Predict using the anomaly detector
        is_outlier = detector.predict(features)  # -1 indicates outlier, 1 indicates inlier
        return is_outlier

# def eval_detector(model, detector, scaler, test_loader, device):
#     """
#     Evaluate the anomaly detector on clean and adversarial examples.
    
#     Returns:
#         precision: Precision of the detector.
#         recall: Recall of the detector.
#     """
#     clean_labels = []
#     adv_labels = []
#     predictions = []
    
#     criterion = nn.CrossEntropyLoss()

#     for data, labels in test_loader:
#         data, labels = data.to(device), labels.to(device)
        
#         # Clean example detection
#         clean_preds = detect_adversarial(detector, scaler, model, data, device)
#         clean_labels.extend([0] * len(clean_preds))  # 0: Clean
#         predictions.extend(clean_preds)
        
#         # Generate adversarial examples
#         adv_data = pgd_attack(model, data, labels, criterion, device, eps=64/255, alpha=2/255, iters=20)
#         adv_preds = detect_adversarial(detector, scaler, model, adv_data, device)
#         adv_labels.extend([1] * len(adv_preds))  # 1: Adversarial
#         predictions.extend(adv_preds)
    
#     # Combine clean and adversarial labels
#     true_labels = clean_labels + adv_labels
    
#     # Convert predictions to binary: -1 (outlier) -> 1 (adversarial), 1 (inlier) -> 0 (clean)
#     binary_predictions = [1 if pred == -1 else 0 for pred in predictions]
    
#     # Calculate precision and recall
#     true_positives = sum((p == 1) and (t == 1) for p, t in zip(binary_predictions, true_labels))
#     false_positives = sum((p == 1) and (t == 0) for p, t in zip(binary_predictions, true_labels))
#     false_negatives = sum((p == 0) and (t == 1) for p, t in zip(binary_predictions, true_labels))
    
#     precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
#     recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
#     return precision, recall

def eval_detector(model, detector, scaler, test_loader, device, clean_to_adv_ratio=1):
    """
    Evaluate the anomaly detector on clean and adversarial examples.
    
    Args:
        model: The neural network model.
        detector: Trained anomaly detection model (e.g., IsolationForest).
        scaler: Scaler used to standardize features.
        test_loader: DataLoader for the test dataset.
        device: Device to run the evaluation on.
        clean_to_adv_ratio: Ratio of clean to adversarial examples in evaluation.
    
    Returns:
        precision: Precision of the detector.
        recall: Recall of the detector.
    """
    clean_labels = []
    adv_labels = []
    predictions = []
    criterion = nn.CrossEntropyLoss()

    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        batch_size = data.size(0)

        # Number of clean and adversarial examples
        num_clean = int(batch_size * clean_to_adv_ratio / (clean_to_adv_ratio + 1))
        num_adv = batch_size - num_clean

        # Split the batch into clean and adversarial subsets
        clean_data, adv_data = data[:num_clean], data[num_clean:]
        clean_labels_batch, adv_labels_batch = labels[:num_clean], labels[num_clean:]

        # Detect clean examples
        clean_preds = detect_adversarial(detector, scaler, model, clean_data, device)
        clean_labels.extend([0] * len(clean_preds))  # 0: Clean examples
        predictions.extend(clean_preds)

        # Generate and detect adversarial examples
        adv_data = pgd_attack(model, adv_data, adv_labels_batch, criterion, device, eps=64/255, alpha=2/255, iters=20)
        adv_preds = detect_adversarial(detector, scaler, model, adv_data, device)
        adv_labels.extend([1] * len(adv_preds))  # 1: Adversarial examples
        predictions.extend(adv_preds)

    # Combine clean and adversarial labels
    true_labels = clean_labels + adv_labels

    # Convert predictions to binary: -1 (outlier) -> 1 (adversarial), 1 (inlier) -> 0 (clean)
    binary_predictions = [1 if pred == -1 else 0 for pred in predictions]

    # Calculate precision and recall
    true_positives = sum((p == 1) and (t == 1) for p, t in zip(binary_predictions, true_labels))
    false_positives = sum((p == 1) and (t == 0) for p, t in zip(binary_predictions, true_labels))
    false_negatives = sum((p == 0) and (t == 1) for p, t in zip(binary_predictions, true_labels))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall


if __name__ == '__main__':
    root = './Data/' 
    model_save_pth = './models/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['UT_HAR_data', 'NTU-Fi-HumanID', 'NTU-Fi_HAR', 'Widar'])
    parser.add_argument('--model', choices=['MLP', 'LeNet', 'ResNet18', 'ResNet50', 'ResNet101', 'RNN', 'GRU', 'LSTM', 'BiLSTM', 'CNN+GRU', 'ViT'])
    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model
    
    # Load data and model
    train_loader, test_loader, model, train_epoch = load_data_n_model(dataset_name, model_name, root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model.load_state_dict(torch.load(os.path.join(model_save_pth, f'{dataset_name}_{model_name}.pth')))
    model.to(device)

    # Train anomaly detector
    detector, scaler = train_anomaly_detector(model, train_loader, device)

    # Evaluate anomaly detector
    precision, recall = eval_detector(model, detector, scaler, test_loader, device)
    
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
