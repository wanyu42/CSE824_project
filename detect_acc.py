import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
import os
from attack import pgd_attack
import torch.nn as nn
from util import load_data_n_model

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
    model.train()  # or model.eval(), depending on the model architecture needs
    handle, features = get_penultimate_layer_features(model, device)
    with torch.no_grad():
        for images, _ in clean_loader:
            images = images.to(device)
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
    model.train()  # or model.eval(), depending on needs
    handle, features = get_penultimate_layer_features(model, device)
    with torch.no_grad():
        test_images = test_images.to(device)
        model(test_images)
        features = scaler.transform(features['penultimate'].cpu().numpy())
        
        # Predict using the anomaly detector
        is_outlier = detector.predict(features)  # -1 indicates outlier (adv), 1 indicates inlier (clean)
        return is_outlier


def eval_with_refusal(model, detector, scaler, test_loader, device, clean_to_adv_ratio=3):
    """
    Evaluate the anomaly detector integrated with a 'refuse' option.
    If detector marks an instance as adversarial (-1):
      - If it's actually adversarial, count as correct (true positive detection).
      - If it's actually clean, count as incorrect (false positive detection).
    If detector marks an instance as clean (1):
      - Classify normally:
        - If clean and classified correctly, correct.
        - If clean and misclassified, incorrect.
        - If adversarial and not detected:
          * If classified correctly (somehow), count as correct.
          * Otherwise, incorrect.

    Returns:
        accuracy: The overall accuracy considering the refusal mechanism.
    """
    criterion = nn.CrossEntropyLoss()
    correct_count = 0
    total_count = 0

    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        batch_size = data.size(0)

        # Number of clean and adversarial examples in the batch
        num_clean = int(batch_size * clean_to_adv_ratio / (clean_to_adv_ratio + 1))
        num_adv = batch_size - num_clean

        # Split the batch into clean and adversarial subsets
        clean_data, adv_data = data[:num_clean], data[num_clean:]
        clean_labels_batch, adv_labels_batch = labels[:num_clean], labels[num_clean:]

        # Detect on clean subset
        clean_preds = detect_adversarial(detector, scaler, model, clean_data, device)
        # clean_preds: -1 means we consider them adversarial and refuse classification,
        # 1 means consider them clean and classify normally.

        # For those predicted as adversarial (-1) but are actually clean => incorrect
        # For those predicted as clean (1), classify normally.
        clean_correct = 0
        # Classification only on those considered clean by detector
        clean_inliers_mask = (clean_preds == 1)
        if clean_inliers_mask.any():
            with torch.no_grad():
                outputs = model(clean_data[clean_inliers_mask])
            _, predicted = torch.max(outputs, 1)
            # Count how many of these are correctly classified
            clean_correct = (predicted == clean_labels_batch[clean_inliers_mask]).sum().item()

        # Add the correct classifications for the clean subset
        # Also, all wrongly flagged as adv are incorrect
        # correct for clean = correct classification on inliers
        # no addition for outliers since they are guaranteed incorrect
        correct_count += clean_correct

        # Now the adversarial subset
        # Generate adversarial examples
        adv_data_pgd = pgd_attack(model, adv_data, adv_labels_batch, criterion, device, eps=64/255, alpha=2/255, iters=20)
        adv_preds = detect_adversarial(detector, scaler, model, adv_data_pgd, device)

        # If adv_preds == -1 (detector says adversarial), and it is indeed adversarial, count as correct
        adv_outliers_mask = (adv_preds == -1)
        adv_correct_outliers = adv_outliers_mask.sum().item()  # all these are correct because they are true positives

        # For the remaining adv samples that are considered clean (1), we classify normally
        adv_inliers_mask = (adv_preds == 1)
        adv_correct_inliers = 0
        if adv_inliers_mask.any():
            with torch.no_grad():
                outputs = model(adv_data_pgd[adv_inliers_mask])
            _, predicted = torch.max(outputs, 1)
            # Count correct classifications of undetected adv examples
            adv_correct_inliers = (predicted == adv_labels_batch[adv_inliers_mask]).sum().item()

        correct_count += adv_correct_outliers + adv_correct_inliers

        # Total samples processed
        total_count += batch_size

    accuracy = correct_count / total_count
    return accuracy


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
    model.load_state_dict(torch.load(os.path.join(model_save_pth, f'{dataset_name}_{model_name}.pth'), map_location=torch.device('cpu')  ))
    model.to(device)

    # Train anomaly detector
    detector, scaler = train_anomaly_detector(model, train_loader, device)

    # Evaluate with refusal option
    accuracy = eval_with_refusal(model, detector, scaler, test_loader, device)
    
    print(f"Accuracy with refusal: {accuracy:.4f}")
