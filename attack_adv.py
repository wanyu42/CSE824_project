from util import load_data_n_model
import os
import torch
import torch.nn as nn
from run_dev import test
import argparse
import matplotlib.pyplot as plt

def pgd_attack(model, data, labels, criterion, device, eps=64/255, alpha=2/255, iters=20) :
    data = data.to(device)
    labels = labels.type(torch.LongTensor)
    labels = labels.to(device)
        
    ori_data = data.data
        
    for i in range(iters) :    
        data.requires_grad = True
        outputs = model(data)

        model.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        adv_data = data + alpha*data.grad.sign()
        eta = torch.clamp(adv_data - ori_data, min=-eps, max=eps)
        data = torch.clamp(ori_data + eta, min=data.data.min(), max=data.data.max()).detach_()
            
    return data


def pgd_test(model, tensor_loader, criterion, device):
    correct = 0
    total = 0

    model.train()
    # model.eval()
    for data, labels in tensor_loader:
        # import ipdb; ipdb.set_trace()

        images = pgd_attack(model, data, labels, criterion, device)

        with torch.no_grad():
            labels = labels.to(device)
            outputs = model(images)
            
            _, pre = torch.max(outputs, 1)

            total += len(labels)
            correct += (pre == labels).sum()
        
    print('PGD Attack Accuracy: {:.4f}'.format(float(correct) / total))


def save_examples(model, test_loader, criterion, device, num_show=2):
    model.eval()
    
    data_return = []  # To store original data for misclassified examples
    adv_return = []   # To store adversarial data for misclassified examples
    correct_labels = []  # To store the correct labels
    wrong_labels = []    # To store the wrong predictions
    
    for data, labels in test_loader:
        adv_images = pgd_attack(model, data, labels, criterion, device)
        
        with torch.no_grad():
            labels = labels.to(device)
            outputs = model(adv_images)
            _, preds = torch.max(outputs, 1)  # Get predictions
            
            misclassified = preds != labels
            if misclassified.any():
                data_return.append(data[misclassified].cpu())
                adv_return.append(adv_images[misclassified].cpu())
                correct_labels.append(labels[misclassified].cpu())
                wrong_labels.append(preds[misclassified].cpu())
            
            if sum(len(batch) for batch in data_return) >= num_show:
                break

    data_return = torch.cat(data_return, dim=0)[:num_show]
    adv_return = torch.cat(adv_return, dim=0)[:num_show]
    correct_labels = torch.cat(correct_labels, dim=0)[:num_show]
    wrong_labels = torch.cat(wrong_labels, dim=0)[:num_show]
    
    return data_return, adv_return, correct_labels, wrong_labels


def plot_examples(dataset_name, model_name, data_return, adv_return, correct_labels, wrong_labels, num_show=2):
    """
    Plots original and adversarial examples with correct and wrong labels.
    
    Args:
        data_return: Tensor of original data (N, C, H, W)
        adv_return: Tensor of adversarial examples (N, C, H, W)
        correct_labels: Tensor of correct labels
        wrong_labels: Tensor of wrong predictions
        num_show: Number of examples to display
    """
    num_show = min(num_show, data_return.size(0))  # Limit to available examples
    
    # Create a figure
    plt.figure()
    fig, axes = plt.subplots(2, num_show, figsize=(8 * num_show, 5))
    if num_show == 1:  # Adjust for a single row of subplots
        axes = np.expand_dims(axes, 0)
    
    for i in range(num_show):
        # Normalize data for visualization (scale to [0, 1])
        orig = data_return[i].clone()
        adv = adv_return[i].clone()
        orig -= orig.min()  # Shift to start at 0
        orig /= orig.max()  # Scale to 1
        adv -= adv.min()    # Shift to start at 0
        adv /= adv.max()    # Scale to 1

        # Original image
        ax = axes[0, i]
        ax.imshow(orig.permute(1, 2, 0).cpu().numpy())
        ax.set_title(f"Original\nLabel: {correct_labels[i].item()}")
        ax.axis("off")

        # Adversarial image
        ax = axes[1, i]
        ax.imshow(adv.permute(1, 2, 0).cpu().numpy())
        ax.set_title(f"Adversarial\nPrediction: {wrong_labels[i].item()}")
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(f'adv_{dataset_name}_{model_name}_{num_show}examples.png')
    plt.close()



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
    model.load_state_dict(torch.load(os.path.join(model_save_pth, f'adv_{dataset_name}_{model_name}.pth')))
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device= device
        )
    pgd_test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device= device
        )

    # data_return, adv_return, correct_labels, wrong_labels = save_examples(model, test_loader, criterion, device)
    # plot_examples(dataset_name, model_name, data_return, adv_return, correct_labels, wrong_labels)


    # import pdb; pdb.set_trace()