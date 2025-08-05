
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import os
import cv2
import tqdm
import argparse
import gc
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, recall_score



from models.models import Baseline, Mymodel
from dataset.casme import casme_dataset, casme_surgical_dataset
from dataset.samm import samm_dataset
from dataset.casme_triplet import casme_triplet_dataset,casme_surgical_triplet_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    # model setting
    parser.add_argument('--embed_dim', default=768, type=int)
    parser.add_argument('--num_classes', default=5, type=int)
    parser.add_argument('--model', default='meln', type=str)

    # process
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--IR50_path', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument("--weight_save_path", type=str, default="./saved_result")
    parser.add_argument("--exp_name", type=str, default="exp")

    # training
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_step', default=[100], nargs='*', type=int)
    parser.add_argument('--lr_gamma', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0., type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--scheduler', default='multistep', type=str)

    # testing 
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--test_image', default='', type=str)
    
    args = parser.parse_args()
    return args



def train(model, train_loader, optimizer, device):
    print("Training")
    print("Learning Rate: {}".format(optimizer.param_groups[0]['lr']))

    model.train()
    train_loss, train_accuracy, train_f1_score =  0, 0, 0
    total_prediction = []
    total_true = []
    

    for _, (inputs, targets, neutral, neutral_B) in enumerate(train_loader):
        
        inputs, targets, neutral, neutral_B = inputs.to(device), targets.to(device), neutral.to(device), neutral_B.to(device)
        optimizer.zero_grad()
        if args.model == "meln":
            outputs, out_p, me_aux, x_aug, out_mask_aug, me_aux_aug = model(inputs, neutral, neutral_B,warm_up=False)
        else:
            outputs = model(inputs)
        
        loss_main = criterion(outputs, targets)
        loss_main_aug = criterion(x_aug, targets)

        N,B,C = me_aux.shape
        targets_aux = targets.repeat(1,N).view(-1)
        me_aux = me_aux.view(-1,C)
        me_aux_aug = me_aux_aug.view(-1,C)

        loss_aux = criterion(me_aux, targets_aux)
        mask_target = out_p.detach().clone()
        loss_aux_aug = l2_criterion(out_mask_aug, mask_target)
        loss = loss_main + 0.001*loss_aux + 0.01*(loss_main_aug+ 0.01*loss_aux_aug)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        # Compute the accuracy
        prediction = (outputs.argmax(-1) == targets)
        train_accuracy += prediction.sum().item() / targets.size(0)
        train_f1_score += f1_score(
            targets.view(-1).tolist(), outputs.argmax(-1).view(-1).tolist(), 
            average="macro"
        )


        total_prediction.extend(outputs.argmax(-1).view(-1).tolist())
        total_true.extend(targets.view(-1).tolist())
    
    
    print("Training Loss: {}".format(train_loss))
    print("Training Accuracy: {}".format(train_accuracy / len(train_loader)))
    print("Training F1 score: {}\n".format(train_f1_score / len(train_loader)))
    
    return train_accuracy / len(train_loader), train_f1_score / len(train_loader), total_prediction, total_true


def test(model, test_loader, device):
    print("Testing")
    model.eval()
    
    
    test_loss, test_accuracy, test_f1_score =  0, 0, 0
    total_prediction = []
    total_true = []
    
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.model == "meln":
                outputs = model(inputs, inputs, inputs)
            else:
                outputs = model(inputs)
            test_loss = criterion(outputs, targets)
            
            # Compute the accuracy
            prediction = (outputs.argmax(-1) == targets)
            test_accuracy += prediction.sum().item() / targets.size(0)
            test_f1_score += f1_score(
                targets.view(-1).tolist(), outputs.argmax(-1).view(-1).tolist(), 
                average="macro"
            )

            total_prediction.extend(outputs.argmax(-1).view(-1).tolist())
            total_true.extend(targets.view(-1).tolist())
            
    
    print("Testing Loss: {}".format(test_loss))
    print("Testing Accuracy: {}".format(test_accuracy / len(test_loader)))
    print("Testing F1 score: {}\n".format(test_f1_score / len(test_loader)))
    
    return test_accuracy / len(test_loader), test_f1_score / len(test_loader), total_prediction, total_true
    

    
def plot_confusion_matrix(data, label_mapping: dict, fig_name: str,):
    type_mapping = label_mapping.values()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.matshow(data, cmap="Blues")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Target")
    ax.set_xticks(np.arange(len(type_mapping)))
    ax.set_xticklabels(list(type_mapping))
    ax.yaxis.set_label_position("right")
    ax.set_yticks(np.arange(len(type_mapping)))
    ax.set_yticklabels(list(type_mapping))

    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, z, ha='center', va='center')
    plt.savefig(fig_name, dpi=1200)

if __name__ == "__main__":
    args = parse_args()
    device = args.device
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.cuda.set_device(6)
    
    # log file
    if os.path.isdir(os.path.join(args.weight_save_path, args.exp_name)):
        print("Experiment Directory Exists!!!")
        assert 0
    else: 
        os.makedirs(os.path.join(args.weight_save_path, args.exp_name))
    log_file = open(f"{args.weight_save_path}/{args.exp_name}/weight.log","w")
    log_file.write(f"Experiment Name: {args.exp_name}\n")
    log_file.write(f"Model: {args.model}\n")
    log_file.write(f"training_data: {args.train_path}\n")
    log_file.write(f"testing_data: {args.test_path}\n")
    # training settings 
    if args.optimizer == 'sgd':
        log_file.write(f"Optimizer: {args.optimizer}\n")
        log_file.write(f"Learning Rate: {args.lr}\n")
        log_file.write(f"Momentum: {args.momentum}\n")
        log_file.write(f"Weight Decay: {args.weight_decay}\n")
    else:
        print('No such optimizer option')
        assert 0
    if args.scheduler == 'multistep':
        log_file.write(f"Scheduler: {args.scheduler}\n")
        log_file.write(f"Milestone: {args.lr_step}\n")
    else:
        print('No such scheduler option')
        assert 0
    
    '''# Calculate the number of parameters in the model
    model = Mymodel()
    model.to(device)
    def compute_model_size(model):
        total_params = sum(p.numel() for p in model.parameters())
        total_size_in_bytes = total_params * 4  # Assuming float32 (4 bytes per parameter)
        total_size_in_mb = total_size_in_bytes / (1024**2)
        return total_size_in_mb

    model_size = compute_model_size(model)
    print(f"Model Size: {model_size} MB")
    # Calculate FLOPS
    import torchprofile

    # Assuming 'model' is your PyTorch model and 'input_tensor' is a dummy input for the model
    input_tensor = torch.randn(1, 3, 112, 112).to(device)
    input_tensor2 = torch.randn(1, 3, 112, 112).to(device)
    input_tensor3 = torch.randn(1, 3, 112, 112).to(device)
    flops = torchprofile.profile_macs(model, (input_tensor, input_tensor2,input_tensor3))

    print(f"FLOPs: {flops}") 
    assert 0'''


    # LOSO
    total_accuracy = 0
    total_f1 = 0
    total_prediction = []
    total_true = []
    sub_list = set(pd.read_csv(args.label_path,usecols=['Subject'])['Subject'].to_list())
    
    for sub_v in sub_list:
        print("subject %d for validation."%sub_v)

        # Prepare model
        if args.model == 'cnntrans':
            model = Baseline()

        elif args.model =='cnn':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(512, 5)
        elif args.model == 'meln':
            model = Mymodel()
        else:
            print('no this model option !')
            assert 0

        model.to(device)
        
        transform = None

        # Prepare Training Data
        #train_data = casme_dataset(data_path = args.train_path, label_path = args.label_path, sub_v = sub_v, split='train',transform=transform)
        train_data = casme_triplet_dataset(data_path = args.train_path, label_path = args.label_path, sub_v = sub_v, split='train',transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        #test_data = casme_dataset(data_path = args.test_path, label_path = args.label_path, sub_v = sub_v, split='test',transform=transform)
        test_data = casme_surgical_triplet_dataset(data_path = args.test_path, label_path = args.label_path, sub_v = sub_v, split='test',transform=transform)
        #test_data = samm_dataset(data_path = args.test_path , label_path = args.label_path, sub_v = sub_v, split='test')
        if len(test_data) ==0:
            continue;
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)
        print('Dataset is ready!')
        

        # training settings 
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            print('No such optimizer option')
            assert 0
        if args.scheduler == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_step, gamma=args.lr_gamma)
        else:
            print('No such scheduler option')
            assert 0
        
        #loss
        criterion = torch.nn.CrossEntropyLoss(weight=train_data.weight.to(device), label_smoothing=0.2)
        l2_criterion = torch.nn.MSELoss()
        #
        print ('start training!')
        best_test_acc = 0
        best_f1 = 0
        best_epoch = -1
        for epoch in range(args.start_epoch, args.epochs):
            #train
            print("\nEpoch: {}".format(epoch))
            train_acc, train_f1, train_prediction, train_true = train(model, train_loader, optimizer, device)
            scheduler.step()
            #test
            test_acc, test_f1, test_prediction, test_true = test(model, test_loader, device)
            if test_acc>best_test_acc:
                best_test_acc = test_acc
                best_f1 = test_f1
                best_epoch = epoch
                temp_prediction = test_prediction
                temp_true = test_true
                # save the model weight
                save_path = os.path.join(args.weight_save_path,args.exp_name,'sub%02d'%sub_v)
                torch.save({'sub': epoch,
                            'model':args.model,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, save_path)

        total_accuracy+=best_test_acc
        total_f1+=best_f1
        total_prediction.extend(temp_prediction)
        total_true.extend(temp_true)  
        print(f"In Subect{sub_v}, best Acc: {best_test_acc:.4f}, best f1, {best_f1:.4f}, in Epoch {best_epoch}\n")
        log_file.write(f"In Subect{sub_v}, best Acc: {best_test_acc:.4f}, best f1, {best_f1:.4f}, in Epoch {best_epoch}\n")
        

    count_correct = 0
    count_total = 0
    for idx in range(len(total_prediction)):
        if total_prediction[idx] == total_true[idx]:
            count_correct += 1
        count_total += 1
    double_total_f1_score = f1_score(total_true, total_prediction, average="macro")
    UAR = recall_score(total_true, total_prediction, average="macro")


    print('#############################################################################')
    print(f"Mean LOSO accuracy: {total_accuracy / 26:.4f}, f1-score: {total_f1 /26:.4f}")
    print(f"Unweighted LOSO accuracy: {count_correct / count_total:.4f}, f1-score: {double_total_f1_score:.4f}")
    print(f"UAR: {UAR:.4f}")

    log_file.write('#############################################################################\n')
    log_file.write(f"Mean LOSO accuracy: {total_accuracy / 26:.4f}, f1-score: {total_f1 /26:.4f}\n")
    log_file.write(f"Unweighted LOSO accuracy: {count_correct / count_total:.4f}, f1-score: {double_total_f1_score:.4f}\n")
    log_file.write(f"UAR: {UAR:.4f}")
    log_file.close()
    

