# Define options
train_method = "5K"
chosen_dataset = "5_95_quarter"
epochs = 200

from neucube import Reservoir
from neucube.encoder import Delta
from neucube.validation.learn_pipeline import learn_pipeline
from neucube.validation.pipeline import Pipeline
from neucube.sampler import SpikeCount, DeSNN
import torch

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score as accuracy_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import plotly.graph_objects as go

import time
import pickle

if chosen_dataset == "raw":
    data_path = '/share/klab/datasets/EEG_Visual/NeuCube_Format/Raw/'
    label_path = '/share/klab/datasets/EEG_Visual/NeuCube_Format/Raw/'
    train_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/split_train.csv')
    test_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/split_test.csv')
    val_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/split_val.csv')
    split_5K_train = '/share/klab/datasets/EEG_Visual/NeuCube_Format/train_k_splits.pkl'
    split_5K_test = '/share/klab/datasets/EEG_Visual/NeuCube_Format/test_k_splits.pkl'
    quarter = False

elif chosen_dataset == "raw_quarter":
    data_path = '/share/klab/datasets/EEG_Visual/NeuCube_Format/Raw/'
    label_path = '/share/klab/datasets/EEG_Visual/NeuCube_Format/Raw_Quarter/'
    train_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_train.csv')
    test_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_test.csv')
    val_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_val.csv')
    split_5K_train = '/share/klab/datasets/EEG_Visual/NeuCube_Format/train_k_splits_quarter.pkl'
    split_5K_test = '/share/klab/datasets/EEG_Visual/NeuCube_Format/test_k_splits_quarter.pkl'
    quarter = True

elif chosen_dataset == "5_95":
    data_path = '/share/klab/datasets/EEG_Visual/NeuCube_Format/5_95/'
    label_path = '/share/klab/datasets/EEG_Visual/NeuCube_Format/5_95/'
    train_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/split_train.csv')
    test_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/split_test.csv')
    val_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/split_val.csv')
    split_5K_train = '/share/klab/datasets/EEG_Visual/NeuCube_Format/train_k_splits.pkl'
    split_5K_test = '/share/klab/datasets/EEG_Visual/NeuCube_Format/test_k_splits.pkl'
    quarter = False

elif chosen_dataset == "5_95_quarter":
    data_path = '/share/klab/datasets/EEG_Visual/NeuCube_Format/5_95/'
    label_path = '/share/klab/datasets/EEG_Visual/NeuCube_Format/5_95_Quarter/'
    train_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_train.csv')
    test_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_test.csv')
    val_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_val.csv')
    split_5K_train = '/share/klab/datasets/EEG_Visual/NeuCube_Format/train_k_splits_quarter.pkl'
    split_5K_test = '/share/klab/datasets/EEG_Visual/NeuCube_Format/test_k_splits_quarter.pkl'
    quarter = True

elif chosen_dataset == "14_70":
    data_path = '/share/klab/datasets/EEG_Visual/NeuCube_Format/14_70/'
    label_path = '/share/klab/datasets/EEG_Visual/NeuCube_Format/14_70/'
    train_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/split_train.csv')
    test_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/split_test.csv')
    val_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/split_val.csv')
    split_5K_train = '/share/klab/datasets/EEG_Visual/NeuCube_Format/train_k_splits.pkl'
    split_5K_test = '/share/klab/datasets/EEG_Visual/NeuCube_Format/test_k_splits.pkl'
    quarter = False

elif chosen_dataset == "14_70_quarter":
    data_path = '/share/klab/datasets/EEG_Visual/NeuCube_Format/14_70/'
    label_path = '/share/klab/datasets/EEG_Visual/NeuCube_Format/14_70_Quarter/'
    train_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_train.csv')
    test_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_test.csv')
    val_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_val.csv')
    split_5K_train = '/share/klab/datasets/EEG_Visual/NeuCube_Format/train_k_splits_quarter.pkl'
    split_5K_test = '/share/klab/datasets/EEG_Visual/NeuCube_Format/test_k_splits_quarter.pkl'
    quarter = True

elif chosen_dataset == "55_95":
    data_path = '/share/klab/datasets/EEG_Visual/NeuCube_Format/55_95/'
    label_path = '/share/klab/datasets/EEG_Visual/NeuCube_Format/55_95/'
    train_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/split_train.csv')
    test_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/split_test.csv')
    val_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/split_val.csv')
    split_5K_train = '/share/klab/datasets/EEG_Visual/NeuCube_Format/train_k_splits.pkl'
    split_5K_test = '/share/klab/datasets/EEG_Visual/NeuCube_Format/test_k_splits.pkl'
    quarter = False

elif chosen_dataset == "55_95_quarter":
    data_path = '/share/klab/datasets/EEG_Visual/NeuCube_Format/55_95/'
    label_path = '/share/klab/datasets/EEG_Visual/NeuCube_Format/55_95_Quarter/'
    train_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_train.csv')
    test_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_test.csv')
    val_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_val.csv')
    split_5K_train = '/share/klab/datasets/EEG_Visual/NeuCube_Format/train_k_splits_quarter.pkl'
    split_5K_test = '/share/klab/datasets/EEG_Visual/NeuCube_Format/test_k_splits_quarter.pkl'
    quarter = True

print(f'Training with the following parameters: {epochs} epochs, train method {train_method}, dataset {chosen_dataset}.')
start_time = time.time()

train_split = train_split.transpose()
train_split = train_split.to_numpy()
train_split = train_split[0]

test_split = test_split.transpose()
test_split = test_split.to_numpy()
test_split = test_split[0]

val_split = val_split.transpose()
val_split = val_split.to_numpy()
val_split = val_split[0]

if train_method == "200L":

    filenameslist = ['sam' + str(i) + '_eeg.csv' for i in range(11965)]

    dfs = []
    for filename in filenameslist:
        df = pd.read_csv(data_path + filename, header=None)
        df = df.iloc[:, 20:460]
        df = df.transpose()
        dfs.append(df)

    fulldf = pd.concat(dfs)

    labels = pd.read_csv(f'{data_path}tar_class_labels.csv', header=None)
    y = labels.values.flatten()

    print(fulldf.shape)

    X = torch.tensor(fulldf.values.reshape(11965, 440, 128))
    encoder = Delta(threshold=0.8)
    X = encoder.encode_dataset(X)
    y = labels.values.flatten()
    print(X.shape)
    print(y.shape)

    m1 = Reservoir(inputs=128)
    out = m1.simulate(X)
    print("\n\nSummary of Reservoir")
    m1.summary()

    # state vector based on spike count
    sampler1 = SpikeCount()
    state_vec1 = sampler1.sample(out)

    # state vector based on deSNN
    sampler2 = DeSNN()
    state_vec2 = sampler2.sample(out)

    loss_fn = nn.CrossEntropyLoss()

    res = Reservoir(inputs=128)
    sam = SpikeCount()
    clf = LogisticRegression(solver='liblinear')
    pipe = learn_pipeline(res, sam, clf)

    # initialize training, test losses and accuracy list
    losses_per_epoch= {"train": [], "test": [], "val": []}
    accuracies_per_epoch = {"train": [], "test": [], "val": []}

    y_test_target_total, y_test_pred_total = [], []
    y_val_target_total, y_val_pred_total = [], []

    for epoch in range(1, epochs+1):

        x_train = X[train_split]
        y_train = y[train_split]

        res, clf = pipe.fit(x_train, y_train)
        pred = pipe.predict(x_train)

        pred = torch.from_numpy(pred)
        y_train = torch.from_numpy(y_train)

        pred = pred.type(torch.float64)
        y_train = y_train.type(torch.float64)

        train_loss = loss_fn(pred.unsqueeze(0), y_train.unsqueeze(0))
        correct = pred.eq(y_train).sum().item()
        train_acc = correct/y_train.size(0)

        x_val = X[val_split]
        y_val = y[val_split]

        pred = pipe.predict(x_val)

        pred = torch.from_numpy(pred)
        y_val = torch.from_numpy(y_val)

        pred = pred.type(torch.float64)
        y_val = y_val.type(torch.float64)

        val_loss = loss_fn(pred.unsqueeze(0), y_val.unsqueeze(0))
        correct = pred.eq(y_val).sum().item()
        val_acc = correct / y_val.size(0)

        y_val_pred_total.extend(pred)
        y_val_target_total.extend(y_val)

        if epoch == epoch:

            x_test = X[test_split]
            y_test = y[test_split]

            pred = pipe.predict(x_test)

            pred = torch.from_numpy(pred)
            y_test = torch.from_numpy(y_test)

            pred = pred.type(torch.float64)
            y_test = y_test.type(torch.float64)

            test_loss = loss_fn(pred.unsqueeze(0), y_test.unsqueeze(0))
            correct = pred.eq(y_test).sum().item()
            test_acc = correct / y_test.size(0)

            y_test_pred_total.extend(pred)
            y_test_target_total.extend(y_test)

            print('tested')

        pipe = learn_pipeline(res, sam, clf)

        losses_per_epoch['train'].append(train_loss.item())
        accuracies_per_epoch['train'].append(train_acc)
        losses_per_epoch['val'].append(val_loss.item())
        accuracies_per_epoch['val'].append(val_acc)

        if epoch == epoch:

            losses_per_epoch['test'].append(test_loss.item())
            accuracies_per_epoch['test'].append(test_acc)

            print('recorded test results')

        print(f'finished epoch {epoch}')


    print(f'test accuracy is {accuracy_score(y_test_target_total, y_test_pred_total)}')
    print(f'val accuracy is {accuracy_score(y_val_target_total, y_val_pred_total)}')

    test_confusion_matrix = confusion_matrix(y_test_target_total, y_test_pred_total)
    val_confusion_matrix = confusion_matrix(y_val_target_total, y_val_pred_total)

    disp_test = ConfusionMatrixDisplay(confusion_matrix=test_confusion_matrix)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp_test.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
    plt.savefig(f'neucube_confusion_matrix_test_{chosen_dataset}_{train_method}.png')

    disp_val = ConfusionMatrixDisplay(confusion_matrix=val_confusion_matrix)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp_val.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
    plt.savefig(f'neucube_confusion_matrix_val_{chosen_dataset}_{train_method}.png')

    print('Confusion Matrices Saved')

    print(f'The final overall test accuracy is {accuracy(y_test_target_total, y_test_pred_total)}.')
    print(f'The final overall val accuracy is {accuracy(y_val_target_total, y_val_pred_total)}.')

    # graph the losses and accuracies
    x = [i for i in range(1, epochs + 1)]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x,
                              y=losses_per_epoch['train'],
                              mode='lines',
                              name='Train Losses per Epoch'))
    fig1.add_trace(go.Scatter(x=x,
                              y=accuracies_per_epoch['train'],
                              mode='lines',
                              name='Train Accuracies per Epoch'))
    fig1.update_layout(title="NeuCube Train Acc Loss")
    fig1.write_image("neucube_train_loss_acc.png")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x,
                              y=losses_per_epoch['test'],
                              mode='lines',
                              name='Test Losses per Epoch'))
    fig2.add_trace(go.Scatter(x=x,
                              y=accuracies_per_epoch['test'],
                              mode='lines',
                              name='Test Accuracies per Epoch'))
    fig2.update_layout(title="NeuCube Test Acc Loss")
    fig2.write_image("neucube_test_loss_acc.png")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=x,
                              y=losses_per_epoch['val'],
                              mode='lines',
                              name='Val Losses per Epoch'))
    fig3.add_trace(go.Scatter(x=x,
                              y=accuracies_per_epoch['val'],
                              mode='lines',
                              name='Val Accuracies per Epoch'))
    fig3.update_layout(title="NeuCube Val Acc Loss")
    fig3.write_image("neucube_val_loss_acc.png")

    print('Loss Acc Graphs Saved')

elif train_method == "5K":

    if quarter == True:

        idx = train_split.tolist() + test_split.tolist() + val_split.tolist()

        idx.sort()

        filenameslist = ['sam' + str(idx) + '_eeg.csv' for idx in idx]

        dfs = []
        for filename in filenameslist:
            df = pd.read_csv(data_path + filename, header=None)
            df = df.iloc[:, 20:460]
            df = df.transpose()
            dfs.append(df)

        fulldf = pd.concat(dfs)

        labels = pd.read_csv(f'{label_path}tar_class_labels.csv', header=None)
        y = labels.values.flatten()

        print(fulldf.shape)

        X = torch.tensor(fulldf.values.reshape(2993, 440, 128))
        encoder = Delta(threshold=0.8)
        X = encoder.encode_dataset(X)
        y = labels.values.flatten()
        print(X.shape)
        print(y.shape)

    elif quarter == False:

        filenameslist = ['sam' + str(i) + '_eeg.csv' for i in range(11965)]

        dfs = []
        for filename in filenameslist:
            df = pd.read_csv(data_path + filename, header=None)
            df = df.iloc[:, 20:460]
            df = df.transpose()
            dfs.append(df)

        fulldf = pd.concat(dfs)

        labels = pd.read_csv(f'{data_path}tar_class_labels.csv', header=None)
        y = labels.values.flatten()

        print(fulldf.shape)

        X = torch.tensor(fulldf.values.reshape(11965, 440, 128))
        encoder = Delta(threshold=0.8)
        X = encoder.encode_dataset(X)
        y = labels.values.flatten()
        print(X.shape)
        print(y.shape)

    m1 = Reservoir(inputs=128)
    out = m1.simulate(X)
    print("\n\nSummary of Reservoir")
    m1.summary()

    # state vector based on spike count
    sampler1 = SpikeCount()
    state_vec1 = sampler1.sample(out)

    # state vector based on deSNN
    sampler2 = DeSNN()
    state_vec2 = sampler2.sample(out)

    loss_fn = nn.CrossEntropyLoss()

    res = Reservoir(inputs=128)
    sam = SpikeCount()
    clf = LogisticRegression(solver='liblinear')
    pipe = Pipeline(res, sam, clf)

    # With original method

    with open(split_5K_train, 'rb') as f:
        train_index = pickle.load(f)

    with open(split_5K_test, 'rb') as f:
        test_index = pickle.load(f)

    y_total, pred_total = [], []

    losses_per_fold = {"train": [], "test": []}
    accuracies_per_fold = {"train": [], "test": []}

    for i in range(5):

        X_train, X_test = X[train_index[i]], X[test_index[i]]
        y_train, y_test = y[train_index[i]], y[test_index[i]]
        
        pipe.fit(X_train, y_train)

        pred_train = pipe.predict(X_train)
        pred_test = pipe.predict(X_test)

        pred_train_l = torch.from_numpy(pred_train)
        pred_test_l = torch.from_numpy(pred_test)
        y_train_l = torch.from_numpy(y_train)
        y_test_l = torch.from_numpy(y_test)

        pred_train_l = pred_train_l.type(torch.float64)
        pred_test_l = pred_test_l.type(torch.float64)
        y_train_l = y_train_l.type(torch.float64)
        y_test_l = y_test_l.type(torch.float64)

        train_loss = loss_fn(pred_train_l.unsqueeze(0), y_train_l.unsqueeze(0))
        correct = pred_train_l.eq(y_train_l).sum().item()
        train_accuracy = correct / (y_train_l.size(0))

        test_loss = loss_fn(pred_test_l.unsqueeze(0), y_test_l.unsqueeze(0))
        correct = pred_test_l.eq(y_test_l).sum().item()
        test_accuracy = correct / (y_test_l.size(0))

        losses_per_fold['train'].append(train_loss.item())
        losses_per_fold['test'].append(test_loss.item())
        accuracies_per_fold['train'].append(train_accuracy)
        accuracies_per_fold['test'].append(test_accuracy)

        y_total.extend(y_test)
        pred_total.extend(pred_test)

        print(f'finished with {i}')

    print(f'The final accuracy is {accuracy_score(y_total, pred_total)}.')

    confusion_matrix = confusion_matrix(y_total, pred_total)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    fig, ax = plt.subplots(figsize=(20, 20))
    disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
    plt.savefig(f'neucube_confusion_matrix_5K.png')

    print('Confusion Matrix Saved')

    x = [i for i in range(1,6)]

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=x,
                             y=losses_per_fold['train'],
                             mode='lines',
                             name='Losses per Split'))
    fig1.add_trace(go.Scatter(x=x,
                             y=accuracies_per_fold['train'],
                             mode='lines',
                             name='Accuracies per Split'))

    fig1.update_layout(title="NeuCube Train Loss Acc")
    fig1.write_image(f"neucube_5K_train_loss_acc.png")

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=x,
                              y=losses_per_fold['test'],
                              mode='lines',
                              name='Losses per Split'))
    fig2.add_trace(go.Scatter(x=x,
                              y=accuracies_per_fold['test'],
                              mode='lines',
                              name='Accuracies per Split'))

    fig2.update_layout(title="NeuCube Test Loss Acc")
    fig2.write_image(f"neucube_5K_test_loss_acc.png")

    print('Loss and Accuracy Graphs Saved')

end_time = time.time()

elapsed_time = end_time - start_time
elapsed_hours = int(elapsed_time/3600)
elapsed_minutes = int((elapsed_time % 3600)/60)

print (f'Training took {elapsed_hours} hours and {elapsed_minutes} minutes.')