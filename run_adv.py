import numpy as np
import torch
import torch.nn as nn
import argparse
from util import load_data_n_model
import matplotlib.pyplot as plt
import os
from attack import pgd_attack
from torch.optim.lr_scheduler import MultiStepLR

def adv_train(model, tensor_loader, num_epochs, learning_rate, criterion, device, test_loader):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    acc_dict = {'train':[], 'test':[]}
    loss_dict = {'train':[], 'test':[]}
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for data in tensor_loader:
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)

            adv_inputs = pgd_attack(model, inputs, labels, criterion, device, eps=64/255, alpha=4/255, iters=10)
            
            optimizer.zero_grad()
            outputs = model(adv_inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs,dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        scheduler.step()
        epoch_loss = epoch_loss/len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy/len(tensor_loader)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))

        val_acc, val_loss = test(model, test_loader, criterion, device)
        acc_dict['train'].append(epoch_accuracy)
        acc_dict['test'].append(val_acc)
        loss_dict['train'].append(epoch_loss)
        loss_dict['test'].append(val_loss)
    
    return acc_dict, loss_dict


def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)
        
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        
        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)
    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))
    return test_acc, test_loss

    
def main():
    root = './Data/' 
    model_save_pth = './models/'
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices = ['UT_HAR_data','NTU-Fi-HumanID','NTU-Fi_HAR','Widar'])
    parser.add_argument('--model', choices = ['MLP','LeNet','ResNet18','ResNet50','ResNet101','RNN','GRU','LSTM','BiLSTM', 'CNN+GRU','ViT'])
    args = parser.parse_args()

    train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    acc_dict, loss_dict = adv_train(
        model=model,
        tensor_loader= train_loader,
        num_epochs= train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device,
        test_loader=test_loader
        )
    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device= device
        )
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(model_save_pth, f'adv_{args.dataset}_{args.model}.pth'))

    # plot the training info
    plt.figure()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
    axes[0].plot(range(len(acc_dict['train'])), acc_dict['train'], label="train")
    axes[0].plot(range(len(acc_dict['test'])), acc_dict['test'], label="test")
    axes[0].set_title("Train vs Test Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("Acc")
    axes[0].legend()

    axes[1].plot(range(len(loss_dict['train'])), loss_dict['train'], label="train")
    axes[1].plot(range(len(loss_dict['test'])), loss_dict['test'], label="test")
    axes[1].set_title("Train vs Test Loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    # Adjust layout for better appearance
    plt.tight_layout()
    plt.savefig(f'plots/adv_{args.dataset}_{args.model}.png')

    return


if __name__ == "__main__":
    main()
