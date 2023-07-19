# Code adapted from (insert reference)
# Original code available at (insert GH link)

# Define options
network = 'lstm5' # 'EEGChannelNet' or 'lstm' or 'lstm5' or 'lstm10'
train_method = "5K"
chosen_dataset = "14_70"
batch_size = 16
epochs = 200

import argparse

if chosen_dataset == "raw":
    data_path = '/share/klab/datasets/EEG_Visual/eeg_signals_raw_with_mean_std.pth'
    split_path_200L = '/share/klab/datasets/EEG_Visual/block_splits_by_image_all.pth'
    split_path_5K = '/share/klab/datasets/EEG_Visual/NeuCube_Splits.pth'

elif chosen_dataset == "raw_quarter":
    data_path = '/share/klab/datasets/EEG_Visual/eeg_signals_raw_with_mean_std.pth'
    split_path_200L = '/share/klab/datasets/EEG_Visual/quarter_splits.pth'
    split_path_5K = '/share/klab/datasets/EEG_Visual/NeuCube_Splits_Quarter.pth'

elif chosen_dataset == "5_95":
    data_path = '/share/klab/datasets/EEG_Visual/eeg_5_95_std.pth'
    split_path_200L = '/share/klab/datasets/EEG_Visual/block_splits_by_image_all.pth'
    split_path_5K = '/share/klab/datasets/EEG_Visual/NeuCube_Splits.pth'

elif chosen_dataset == "5_95_quarter":
    data_path = '/share/klab/datasets/EEG_Visual/eeg_5_95_std.pth'
    split_path_200L = '/share/klab/datasets/EEG_Visual/quarter_splits.pth'
    split_path_5K = '/share/klab/datasets/EEG_Visual/NeuCube_Splits_Quarter.pth'

elif chosen_dataset == "14_70":
    data_path = '/share/klab/datasets/EEG_Visual/eeg_14_70_std.pth'
    split_path_200L = '/share/klab/datasets/EEG_Visual/block_splits_by_image_all.pth'
    split_path_5K = '/share/klab/datasets/EEG_Visual/NeuCube_Splits.pth'

elif chosen_dataset == "14_70_quarter":
    data_path = '/share/klab/datasets/EEG_Visual/eeg_14_70_std.pth'
    split_path_200L = '/share/klab/datasets/EEG_Visual/quarter_splits.pth'
    split_path_5K = '/share/klab/datasets/EEG_Visual/NeuCube_Splits_Quarter.pth'

elif chosen_dataset == "55_95":
    data_path = '/share/klab/datasets/EEG_Visual/eeg_55_95_std.pth'
    split_path_200L = '/share/klab/datasets/EEG_Visual/block_splits_by_image_all.pth'
    split_path_5K = '/share/klab/datasets/EEG_Visual/NeuCube_Splits.pth'

elif chosen_dataset == "55_95_quarter":
    data_path = '/share/klab/datasets/EEG_Visual/eeg_55_95_std.pth'
    split_path_200L = '/share/klab/datasets/EEG_Visual/quarter_splits.pth'
    split_path_5K = '/share/klab/datasets/EEG_Visual/NeuCube_Splits_Quarter.pth'

parser = argparse.ArgumentParser(description="Template")

# define the dataset - update file location!
parser.add_argument('-ed', '--eeg-dataset', default=data_path, help="EEG dataset path") # 5-95Hz filter

# split the data for all participants into blocks - update file location!
if train_method == "200L":
    parser.add_argument('-sp', '--splits-path', default=split_path_200L, help="splits path") # all subjects
    parser.add_argument('-sn', '--split-num', default=0, type=int, help="split number")  # should always be zero

elif train_method == "5K":
    parser.add_argument('-sp', '--splits-path', default=split_path_5K, help="splits path")  # all subjects

# select the data for all subjects
parser.add_argument('-sub','--subject', default=0, type=int, help="choose a subject from 1 to 6, default is 0 (all subjects)")

# use all time samples (from 20 to 460)
parser.add_argument('-tl', '--time_low', default=20, type=float, help="lowest time value")
parser.add_argument('-th', '--time_high', default=460,  type=float, help="highest time value")

# select the model to be used - lstm or EEGChannelNet
# - lstm is the model described in the paper "Deep Learning Human Mind for Automated Visual Classification”, in CVPR 2017
parser.add_argument('-mt','--model_type', default=network, help='specify which generator should be used: lstm|EEGChannelNet')
parser.add_argument('-mp','--model_params', default='', nargs='*', help='list of key=value pairs of model options')
parser.add_argument('--pretrained_net', default='', help="path to pre-trained net (to continue training)")

# set the usual training parameters
parser.add_argument("-b", "--batch_size", default=batch_size, type=int, help="batch size")
parser.add_argument('-o', '--optim', default="Adam", help="optimizer")
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help="learning rate")
parser.add_argument('-lrdb', '--learning-rate-decay-by', default=0.5, type=float, help="learning rate decay factor")
parser.add_argument('-lrde', '--learning-rate-decay-every', default=10, type=int, help="learning rate decay period")
parser.add_argument('-dw', '--data-workers', default=4, type=int, help="data loading workers")
parser.add_argument('-e', '--epochs', default=epochs, type=int, help="training epochs")

# add code to handle activation of this script in an environment without CUDA
parser.add_argument('--no-cuda', default=False, help="disable CUDA", action="store_true")

# parse arguments
opt = parser.parse_args()

# Imports
import sys
import os
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
from scipy.fftpack import fft, rfft, fftfreq, irfft, ifft, rfftfreq
from scipy import signal
import numpy as np
import com_models
import importlib
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score as accuracy_score

loss_fn = nn.CrossEntropyLoss()

# Dataset class
class EEGDataset:
    
    # Constructor
    def __init__(self, eeg_signals_path):

        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        if opt.subject!=0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==opt.subject]
        else:
            self.data=loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):

        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[opt.time_low:opt.time_high,:]

        if opt.model_type == "EEGChannelNet":
            eeg = eeg.t()
            eeg = eeg.view(1,128,opt.time_high-opt.time_low)
        
 	# Get label
        label = self.data[i]["label"]

        # Return
        return eeg, label

# Splitter class
class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):

        # Set EEG dataset
        self.dataset = dataset

        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]

        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]

        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):

        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]

        # Return
        return eeg, label

print(f'Training {network} with the following parameters: dataset {chosen_dataset}, train method {train_method}, epochs {epochs}, batch size {batch_size}.')

start_time = time.time()

if train_method == "200L":

    # Load dataset
    dataset = EEGDataset(opt.eeg_dataset)

    # Create loaders
    loaders = {split: DataLoader(Splitter(dataset, split_path = opt.splits_path, split_num = opt.split_num, split_name = split), batch_size = opt.batch_size, drop_last = True, shuffle = True) for split in ["train", "val", "test"]}

    # Load model
    model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value) for (key, value) in [x.split("=") for x in opt.model_params]}

    # Create discriminator model/optimizer
    module = importlib.import_module("com_models." + opt.model_type)
    model = module.Model(**model_options)
    optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr = opt.learning_rate)

    # Setup CUDA
    if not opt.no_cuda:
        model.cuda()
        print("Copied to CUDA")

    #initialize training,validation, test losses and accuracy list
    losses_per_epoch={"train":[], "val":[],"test":[]}
    accuracies_per_epoch={"train":[],"val":[],"test":[]}

    best_accuracy = 0
    best_accuracy_val = 0
    best_epoch = 0

    predictions_cm = {"train": [], "val": [], "test": []}
    targets_cm = {"train": [], "val": [], "test": []}

    for epoch in range(1, opt.epochs+1):
        # Initialize loss/accuracy variables
        losses = {"train": 0, "val": 0, "test": 0}
        accuracies = {"train": 0, "val": 0, "test": 0}
        counts = {"train": 0, "val": 0, "test": 0}

        # Adjust learning rate for SGD
        if opt.optim == "SGD":
            lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Process each split
        for split in ("train", "val", "test"):

            # Set network mode
            if split == "train":
                model.train()
                torch.set_grad_enabled(True)
            else:
                model.eval()
                torch.set_grad_enabled(False)

            # Process all split batches
            for i, (input, target) in enumerate(loaders[split]):

                # Check CUDA
                if not opt.no_cuda:
                    input = input.to("cuda")
                    target = target.to("cuda")

                # Forward step
                output = model(input)

                # Compute loss
                loss_backward = F.cross_entropy(output, target)

                # Compute accuracy
                _,pred = output.data.max(1)

                pred = pred.type(torch.float64)
                target = target.type(torch.float64)
                loss = loss_fn(pred.unsqueeze(0),target.unsqueeze(0))

                losses[split] += loss.item()

                pred_cm = pred.cpu()
                pred_cm = pred_cm.numpy()

                target_cm = target.cpu()
                target_cm = target_cm.numpy()

                predictions_cm[split].extend(pred_cm)
                targets_cm[split].extend(target_cm)

                correct = pred.eq(target.data).sum().item()
                accuracy = correct/input.data.size(0)
                accuracies[split] += accuracy
                counts[split] += 1

                # Backward step and optimization
                if split == "train":
                    optimizer.zero_grad()
                    loss_backward.backward()
                    optimizer.step()

        # Print info at the end of the epoch
        if accuracies["val"]/counts["val"] >= best_accuracy_val:
            best_accuracy_val = accuracies["val"]/counts["val"]
            best_accuracy = accuracies["test"]/counts["test"]
            best_epoch = epoch

        if opt.subject == 0:
            subject='All'
        else:
            subject = opt.subject

        TrL,TrA,VL,VA,TeL,TeA=  losses["train"]/counts["train"],accuracies["train"]/counts["train"],losses["val"]/counts["val"],accuracies["val"]/counts["val"],losses["test"]/counts["test"],accuracies["test"]/counts["test"]
        print("Model: {11} - Subject {12} - Time interval: [{9}-{10}]  [{9}-{10} Hz] - Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, TeL={5:.4f}, TeA={6:.4f}, TeA at max VA = {7:.4f} at epoch {8:d}".format(epoch,
                                                                                                             losses["train"]/counts["train"],
                                                                                                             accuracies["train"]/counts["train"],
                                                                                                             losses["val"]/counts["val"],
                                                                                                             accuracies["val"]/counts["val"],
                                                                                                             losses["test"]/counts["test"],
                                                                                                             accuracies["test"]/counts["test"],
                                                                                                             best_accuracy, best_epoch, opt.time_low,opt.time_high, opt.model_type,opt.subject))

        losses_per_epoch['train'].append(TrL)
        losses_per_epoch['val'].append(VL)
        losses_per_epoch['test'].append(TeL)
        accuracies_per_epoch['train'].append(TrA)
        accuracies_per_epoch['val'].append(VA)
        accuracies_per_epoch['test'].append(TeA)

    print(f'test accuracy is {accuracy_score(targets_cm["test"], predictions_cm["test"])}')
    print(f'val accuracy is {accuracy_score(targets_cm["val"], predictions_cm["val"])}')

    # Print the confusion matrices
    test_confusion_matrix = confusion_matrix(targets_cm['test'], predictions_cm['test'])
    val_confusion_matrix = confusion_matrix(targets_cm['val'], predictions_cm['val'])

    disp_test = ConfusionMatrixDisplay(confusion_matrix=test_confusion_matrix)
    fig, ax = plt.subplots(figsize=(20, 20))
    disp_test.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
    plt.savefig(f'{network}_{chosen_dataset}_{train_method}_confusion_matrix_test.png')

    disp_val = ConfusionMatrixDisplay(confusion_matrix=val_confusion_matrix)
    fig, ax = plt.subplots(figsize=(20, 20))
    disp_val.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
    plt.savefig(f'{network}_{chosen_dataset}_{train_method}_confusion_matrix_val.png')

    print('Confusion Matrices Saved')

    # graph the losses and accuracies
    x = [i for i in range(1, opt.epochs+1)]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x,
                             y=losses_per_epoch['train'],
                             mode='lines',
                             name='Train Losses per Epoch'))
    fig1.add_trace(go.Scatter(x=x,
                             y=accuracies_per_epoch['train'],
                             mode='lines',
                             name='Train Accuracies per Epoch'))
    fig1.update_layout(title=f"{network} Train Acc Loss")
    fig1.write_image(f"{network}_{chosen_dataset}_{train_method}_train_loss_acc.png")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x,
                             y=losses_per_epoch['test'],
                             mode='lines',
                             name='Test Losses per Epoch'))
    fig2.add_trace(go.Scatter(x=x,
                             y=accuracies_per_epoch['test'],
                             mode='lines',
                             name='Test Accuracies per Epoch'))
    fig2.update_layout(title=f"{network} Test Acc Loss")
    fig2.write_image(f"{network}_{chosen_dataset}_{train_method}_test_loss_acc.png")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=x,
                             y=losses_per_epoch['val'],
                             mode='lines',
                             name='Val Losses per Epoch'))
    fig3.add_trace(go.Scatter(x=x,
                             y=accuracies_per_epoch['val'],
                             mode='lines',
                             name='Val Accuracies per Epoch'))
    fig3.update_layout(title=f"{network} Val Acc Loss")
    fig3.write_image(f"{network}_{chosen_dataset}_{train_method}_val_loss_acc.png")

    print('Loss Acc Graphs Saved')

elif train_method == "5K":

    # Load dataset
    dataset = EEGDataset(opt.eeg_dataset)

    loaders = []

    # Create loaders
    for i in range(5):
        loader = {split: DataLoader(Splitter(dataset, split_path = opt.splits_path, split_num = i, split_name = split), batch_size = opt.batch_size, drop_last = True, shuffle = True) for split in ["train", "test"]}
        loaders.append(loader)

    #initialize training, test losses and accuracy list
    losses_per_split={"train":[], "test":[]}
    accuracies_per_split={"train":[],"test":[]}

    predictions_cm = {"train": [], "test": []}
    targets_cm = {"train": [], "test": []}

    for i in range(5):

        # Load model
        model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value) for
                         (key, value) in [x.split("=") for x in opt.model_params]}

        # Create discriminator model/optimizer
        module = importlib.import_module("com_models." + opt.model_type)
        model = module.Model(**model_options)
        optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr=opt.learning_rate)

        # Setup CUDA
        if not opt.no_cuda:
            model.cuda()
            print("Copied to CUDA")

        # Initialize loss/accuracy variables
        losses = {"train": 0, "test": 0}
        accuracies = {"train": 0, "test": 0}
        counts = {"train": 0, "test": 0}

        # Adjust learning rate for SGD
        if opt.optim == "SGD":
            lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Process each split
        for split in ("train", "test"):

            # Set network mode
            if split == "train":
                model.train()
                torch.set_grad_enabled(True)
            else:
                model.eval()
                torch.set_grad_enabled(False)

            # Process all split batches
            for a, (input, target) in enumerate(loaders[i][split]):

                # Check CUDA
                if not opt.no_cuda:
                    input = input.to("cuda")
                    target = target.to("cuda")

                # Forward step
                output = model(input)

                # Compute loss
                loss_backward = F.cross_entropy(output, target)

                # Compute accuracy
                _,pred = output.data.max(1)

                pred = pred.type(torch.float64)
                target = target.type(torch.float64)
                loss = loss_fn(pred.unsqueeze(0), target.unsqueeze(0))

                losses[split] += loss.item()

                pred_cm = pred.cpu()
                pred_cm = pred_cm.numpy()

                target_cm = target.cpu()
                target_cm = target_cm.numpy()

                predictions_cm[split].extend(pred_cm)
                targets_cm[split].extend(target_cm)

                correct = pred.eq(target.data).sum().item()
                accuracy = correct/input.data.size(0)
                accuracies[split] += accuracy
                counts[split] += 1

                # Backward step and optimization
                if split == "train":
                    optimizer.zero_grad()
                    loss_backward.backward()
                    optimizer.step()

        train_loss = losses["train"]/counts["train"]
        test_loss = losses["test"] / counts["test"]
        train_acc = accuracies["train"] / counts["train"]
        test_acc = accuracies["test"] / counts["test"]

        losses_per_split['train'].append(train_loss)
        accuracies_per_split['train'].append(train_acc)
        losses_per_split['test'].append(test_loss)
        accuracies_per_split['test'].append(test_acc)

        print(f'training loss with split {i} is {train_loss}')
        print(f'test loss with split {i} is {test_loss}')
        print(f'training accuracy with split {i} is {train_acc}')
        print(f'test accuracy with split {i} is {test_acc}')

    target_test = targets_cm['test']
    pred_test = predictions_cm['test']

    print(f'test accuracy is {accuracy_score(target_test,pred_test)}')

    # Print the confusion matrices
    test_confusion_matrix = confusion_matrix(targets_cm['test'], predictions_cm['test'])
    disp_test = ConfusionMatrixDisplay(confusion_matrix=test_confusion_matrix)
    fig, ax = plt.subplots(figsize=(20, 20))
    disp_test.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
    plt.savefig(f'{network}_{chosen_dataset}_{train_method}_confusion_matrix_test.png')

    print('Confusion Matrix Saved')

    # graph the losses and accuracies
    x = [i for i in range(1, 6)]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x,
                             y=losses_per_split['train'],
                             mode='lines',
                             name='Train Losses per Epoch'))
    fig1.add_trace(go.Scatter(x=x,
                             y=accuracies_per_split['train'],
                             mode='lines',
                             name='Train Accuracies per Epoch'))
    fig1.update_layout(title=f"{network}_Train Acc Loss")
    fig1.write_image(f"{network}_{chosen_dataset}_{train_method}_train_loss_acc.png")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x,
                             y=losses_per_split['test'],
                             mode='lines',
                             name='Test Losses per Epoch'))
    fig2.add_trace(go.Scatter(x=x,
                             y=accuracies_per_split['test'],
                             mode='lines',
                             name='Test Accuracies per Epoch'))
    fig2.update_layout(title=f"{network} Test Acc Loss")
    fig2.write_image(f"{network}_{chosen_dataset}_{train_method}_test_loss_acc.png")

    print('Loss Acc Graphs Saved')

end_time = time.time()

elapsed_time = end_time - start_time
elapsed_hours = int(elapsed_time/3600)
elapsed_minutes = int((elapsed_time % 3600)/60)

print (f'Training took {elapsed_hours} hours and {elapsed_minutes} minutes.')


