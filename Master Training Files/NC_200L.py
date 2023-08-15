'''
This file trains NeuCube on selected data. The basic training code comes from the original NeuCube code, with modifications made around it to apply the network in the
epoch learning paradigm and to implement loss/acc graphing.
'''

# Define which training paradigm and dataset are to be used, as well as how many epochs to train for
train_method = "200L"
chosen_dataset = "5_95_quarter"
epochs = 50

# import the NeuCube files and all needed supplementary packages
from neucube import Reservoir
from neucube.encoder import Delta
from neucube.validation.pipeline import Pipeline
from neucube.sampler import SpikeCount, DeSNN
import torch
import pandas as pd
from sklearn.metrics import accuracy_score as accuracy_score
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.graph_objects as go
import time
import pickle
import csv
import os

# define the various splits and data paths based on the dataset selected
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

# print a note of the parameters the training will be using
print(f'Training NeuCube with the following parameters: {epochs} epochs, train method {train_method}, dataset {chosen_dataset}.')

# record the start time of the training
start_time = time.time()

# load the train, test, and val indices based on the locations defined above, which depend on the dataset selected
train_split = train_split.transpose()
train_split = train_split.to_numpy()
train_split = train_split[0]

test_split = test_split.transpose()
test_split = test_split.to_numpy()
test_split = test_split[0]

val_split = val_split.transpose()
val_split = val_split.to_numpy()
val_split = val_split[0]

# define the training procedure for the epoch training paradigm
if train_method == "200L":

    # record the start time of the DS load
    ds_start = time.time()

    # lines 132 - 143 from the original NeuCube training code, lightly modified to select the appropriate time range in the EEG

    # create a list of the file names for the full DS
    filenameslist = ['sam' + str(i) + '_eeg.csv' for i in range(11965)]

    # load the EEG segments into a pandas dataframe, selecting 20-460ms after stimulus onset
    dfs = []
    for filename in filenameslist:
        df = pd.read_csv(data_path + filename, header=None)
        df = df.iloc[:, 20:460]
        df = df.transpose()
        dfs.append(df)

    fulldf = pd.concat(dfs)

    # load the label file
    # data path versus label path because with 200L, NC always works with the whole DS as the indices are designed for all files being present
    labels = pd.read_csv(f'{data_path}tar_class_labels.csv', header=None)
    y = labels.values.flatten()

    # check the ds size
    print(fulldf.shape)

    # lines 155 - 158 from the original NeuCube code, lightly modified to the parameters of the current dataset
    # load the dataset into a tensor of size files x segment length x features
    X = torch.tensor(fulldf.values.reshape(11965, 440, 128))
    encoder = Delta(threshold=0.8)
    X = encoder.encode_dataset(X)
    y = labels.values.flatten()
    print(X.shape)
    print(y.shape)

    # record how long loading the ds took
    ds_end = time.time()
    elapsed_time = ds_end - ds_start
    elapsed_hours = int(elapsed_time/3600)
    elapsed_minutes = int((elapsed_time % 3600)/60)
    print (f'ds load took {elapsed_hours} hours and {elapsed_minutes} minutes.')

    # record how long preparing the reservoir and samplers takes
    prep_start = time.time()

    # prepare the reservoir and the samplers
    # lines 174 - 185 from the original NeuCube code
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

    # define the loss function to be used
    loss_fn = nn.CrossEntropyLoss()

    # define the training pipeline
    # lines 193 -196 from the original NeuCube code
    # max_iter on clf increased from original code due to convergence failure errors in early runs
    res = Reservoir(inputs=128)
    sam = SpikeCount()
    clf = LogisticRegression(solver='liblinear', max_iter = 10000)
    pipe = Pipeline(res, sam, clf)

    # record how long preparation took
    prep_end = time.time()
    elapsed_time = prep_end - prep_start
    elapsed_hours = int(elapsed_time/3600)
    elapsed_minutes = int((elapsed_time % 3600)/60)
    print (f'Prep took {elapsed_hours} hours and {elapsed_minutes} minutes.')

    # instantiate variables for the best validation accuracy, best epoch, and best test accuracy at best validation accuracy
    best_val = 0
    best_epoch = 0
    test_best_val = 0

    # the code for the application of NeuCube to the data in this section largely follows the original code, though updated to allow training across epochs and with a
    # train/val/test split versus the train/test split used in the original code
    # iterate over the epochs
    for epoch in range(1, epochs+1):

        # record the epoch start time
        epoch_start = time.time()

        # load the train data and label splits and then fit the pipeline to them
        # lines 218 - 222 from the original NeuCube code, though pipe.fit is now called on each epoch to allow NeuCube to train across epochs
        x_train = X[train_split]
        y_train = y[train_split]

        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_train)

        # convert the predictions and true labels to torch tensors and calculate the loss and accuracy for this epoch
        pred = torch.from_numpy(pred)
        y_train = torch.from_numpy(y_train)
        pred = pred.type(torch.float64)
        y_train = y_train.type(torch.float64)

        print('pred is')
        print(pred)
        print(pred.unsqueeze(0))
        print('y_train is')
        print(y_train)
        print(y_train.unsqueeze(0))

        train_loss = loss_fn(pred.unsqueeze(0), y_train.unsqueeze(0))
        correct = pred.eq(y_train).sum().item()
        train_acc = correct/y_train.size(0)

        # repeat the same procedure for the validation split, this time saving the predictions and targets into files to save them per epoch and across all epochs
        # for later construction of the confusion matrices
        x_val = X[val_split]
        y_val = y[val_split]

        pred = pipe.predict(x_val)

        for value in pred:
            with open(f'nc_200L_predictions_cm_val.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([value])

        for value in pred:
            with open(f'nc_pred_val_{epoch}.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([value])

        for value in y_val:
            with open(f'nc_200L_targets_cm_val.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([value])

        for value in y_val:
            with open(f'nc_target_val_{epoch}.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([value])

        pred = torch.from_numpy(pred)
        y_val = torch.from_numpy(y_val)
        pred = pred.type(torch.float64)
        y_val = y_val.type(torch.float64)

        val_loss = loss_fn(pred.unsqueeze(0), y_val.unsqueeze(0))
        correct = pred.eq(y_val).sum().item()
        val_acc = correct / y_val.size(0)

        # repeat the same procedure with the test split
        x_test = X[test_split]
        y_test = y[test_split]

        pred = pipe.predict(x_test)

        for value in pred:
            with open(f'nc_200L_predictions_cm_test.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([value])

        for value in pred:
            with open(f'nc_pred_test_{epoch}.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([value])

        for value in y_test:
            with open(f'nc_200L_targets_cm_test.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([value])

        for value in y_test:
            with open(f'nc_target_test_{epoch}.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([value])

        pred = torch.from_numpy(pred)
        y_test = torch.from_numpy(y_test)
        pred = pred.type(torch.float64)
        y_test = y_test.type(torch.float64)

        test_loss = loss_fn(pred.unsqueeze(0), y_test.unsqueeze(0))
        correct = pred.eq(y_test).sum().item()
        test_acc = correct / y_test.size(0)

        # if the validation accuracy on this epoch is better than the previous best val acc, update the tracking variables
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            test_best_val = test_acc

        # record the losses and accs for this epoch
        with open(f'nc_200L_losses_per_epoch_train.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([train_loss.item()])

        with open(f'nc_200L_losses_per_epoch_val.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([val_loss.item()])

        with open(f'nc_200L_losses_per_epoch_test.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([test_loss.item()])

        with open(f'nc_200L_acc_per_epoch_train.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([train_acc])

        with open(f'nc_200L_acc_per_epoch_val.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([val_acc])

        with open(f'nc_200L_acc_per_epoch_test.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([test_acc])

        # print the current best test accuracy at the best validation accuracy as of this epoch
        print(f'Test acc at best val acc - {best_val} - is {test_best_val}, achieved on epoch {best_epoch} .')

        # print the time the epoch took
        epoch_end = time.time()
        elapsed_time = epoch_end - epoch_start
        elapsed_hours = int(elapsed_time/3600)
        elapsed_minutes = int((elapsed_time % 3600)/60)
        print (f'Epoch took {elapsed_hours} hours and {elapsed_minutes} minutes.')

    # load the predictions and targets for creating the confusion matrices
    with open('nc_200L_predictions_cm_test.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        predictions_cm_test = [int(float(item)) for sublist in data for item in sublist]

    with open('nc_200L_predictions_cm_val.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        predictions_cm_val = [int(float(item)) for sublist in data for item in sublist]

    with open('nc_200L_targets_cm_test.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        targets_cm_test = [int(float(item)) for sublist in data for item in sublist]

    with open('nc_200L_targets_cm_val.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        targets_cm_val = [int(float(item)) for sublist in data for item in sublist]

    # print the current test loss and accuracy on this epoch
    print(f'test loss at {test_loss}')
    print(f'test acc at {test_acc}')

    # print the confusion matrix (note that this is adapted from the method used in the original NeuCube code
    test_confusion_matrix = confusion_matrix(targets_cm_test, predictions_cm_test)
    val_confusion_matrix = confusion_matrix(targets_cm_val, predictions_cm_val)

    disp_test = ConfusionMatrixDisplay(confusion_matrix=test_confusion_matrix)
    fig, ax = plt.subplots(figsize=(30, 30))
    disp_test.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
    plt.savefig(f'neucube_confusion_matrix_test_{chosen_dataset}_200L.png')

    disp_val = ConfusionMatrixDisplay(confusion_matrix=val_confusion_matrix)
    fig, ax = plt.subplots(figsize=(30, 30))
    disp_val.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
    plt.savefig(f'neucube_confusion_matrix_val_{chosen_dataset}_200L.png')

    print('Confusion Matrices Saved')

    # load the accs and losses
    with open('nc_200L_losses_per_epoch_train.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        losses_per_epoch_train = [int(float(item)) for sublist in data for item in sublist]

    with open('nc_200L_losses_per_epoch_val.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        losses_per_epoch_val = [int(float(item)) for sublist in data for item in sublist]

    with open('nc_200L_losses_per_epoch_test.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        losses_per_epoch_test = [int(float(item)) for sublist in data for item in sublist]

    with open('nc_200L_acc_per_epoch_train.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        acc_per_epoch_train = [int(float(item)) for sublist in data for item in sublist]

    with open('nc_200L_acc_per_epoch_val.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        acc_per_epoch_val = [int(float(item)) for sublist in data for item in sublist]

    with open('nc_200L_acc_per_epoch_test.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        acc_per_epoch_test = [int(float(item)) for sublist in data for item in sublist]

    # graph the losses and accuracies
    x = [i for i in range(1, epochs + 1)]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x,
                              y=losses_per_epoch_train,
                              mode='lines',
                              name='Train Losses per Epoch'))
    fig1.add_trace(go.Scatter(x=x,
                              y=acc_per_epoch_train,
                              mode='lines',
                              name='Train Accuracies per Epoch'))
    fig1.update_layout(title=f"NeuCube Epochs {chosen_dataset} Train Acc Loss")
    fig1.write_image(f"neucube_train_loss_acc_200L_{chosen_dataset}.png")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x,
                              y=losses_per_epoch_val,
                              mode='lines',
                              name='Val Losses per Epoch'))
    fig2.add_trace(go.Scatter(x=x,
                              y=acc_per_epoch_val,
                              mode='lines',
                              name='Val Accuracies per Epoch'))
    fig2.update_layout(title=f"NeuCube Epochs {chosen_dataset} Val Acc Loss")
    fig2.write_image(f"neucube_val_loss_acc_200L_{chosen_dataset}.png")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=x,
                              y=losses_per_epoch_test,
                              mode='lines',
                              name='Val Losses per Epoch'))
    fig3.add_trace(go.Scatter(x=x,
                              y=acc_per_epoch_test,
                              mode='lines',
                              name='Val Accuracies per Epoch'))
    fig3.update_layout(title=f"NeuCube Epochs {chosen_dataset} Test Acc Loss")
    fig3.write_image(f"neucube_test_loss_acc_200L_{chosen_dataset}.png")

    print('Loss Acc Graphs Saved')

    # assemble the losses and accs in a master file and remove the unneeded csv files
    with open('nc_200L_losses_per_epoch_train.csv', mode='r') as file1, \
        open('nc_200L_losses_per_epoch_val.csv', mode='r') as file2, \
        open('nc_200L_losses_per_epoch_test.csv', mode='r') as file3, \
        open('nc_200L_acc_per_epoch_train.csv', mode='r') as file4, \
        open('nc_200L_acc_per_epoch_val.csv', mode='r') as file5, \
        open('nc_200L_acc_per_epoch_test.csv', mode='r') as file6, \
        open(f'nc_200L_{chosen_dataset}_{epochs}epochs_variable_printout.csv', mode='w', newline='') as output:

        reader1 = csv.reader(file1)
        reader2 = csv.reader(file2)
        reader3 = csv.reader(file3)
        reader4 = csv.reader(file4)
        reader5 = csv.reader(file5)
        reader6 = csv.reader(file6)

        writer = csv.writer(output)

        header = ['Epoch Losses Train',
                  'Epoch Losses Val',
                  'Epoch Losses Test',
                  'Epoch Accs Train',
                  'Epoch Accs Val',
                  'Epoch Accs Test']

        writer.writerow(header)

        for value1, value2, value3, value4, value5, value6 in zip(reader1, reader2, reader3, reader4, reader5, reader6):
            writer.writerow(value1 + value2 + value3 + value4+ value5 + value6)

    os.remove('nc_200L_losses_per_epoch_train.csv')
    os.remove('nc_200L_losses_per_epoch_val.csv')
    os.remove('nc_200L_losses_per_epoch_test.csv')
    os.remove('nc_200L_acc_per_epoch_train.csv')
    os.remove('nc_200L_acc_per_epoch_val.csv')
    os.remove('nc_200L_acc_per_epoch_test.csv')

# define the training procedure for the cross validation paradigm
elif train_method == "5K":

    # define the dataset when working with the quartered DS (with the quartered DS in the 5K paradigm, the index values are meant for the DS already in its quartered form)
    if quarter == True:

        # load the segments and create the dataframe (same mix of original NeuCube code and custom code)
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

    # define the dataset when working with the full DS
    elif quarter == False:

        # load the segments and create the dataframe (same mix of original NeuCube code and custom code)
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

    # the following code mirrors the code used for the epoch learning paradigm, with the same mix of original and custom code, with any additions noted
    # note that this code follows the original code for a train/test split
    prep_start = time.time()

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

    # load the 5K splits from the appropriate pkl file
    with open(split_5K_train, 'rb') as f:
        train_index = pickle.load(f)

    with open(split_5K_test, 'rb') as f:
        test_index = pickle.load(f)

    prep_end = time.time()
    elapsed_time = prep_end - prep_start
    elapsed_hours = int(elapsed_time/3600)
    elapsed_minutes = int((elapsed_time % 3600)/60)
    print (f'Prep took {elapsed_hours} hours and {elapsed_minutes} minutes.')

    # run NeuCube on the splits for each fold
    for i in range(5):

        pipe_start = time.time()

        #max_iter on clf increased from original code due to convergence failure errors in early runs
        res = Reservoir(inputs=128)
        sam = SpikeCount()
        clf = LogisticRegression(solver='liblinear', max_iter = 10000)
        pipe = Pipeline(res, sam, clf)

        pipe_end = time.time()
        elapsed_time = pipe_end - pipe_start
        elapsed_hours = int(elapsed_time/3600)
        elapsed_minutes = int((elapsed_time % 3600)/60)
        print (f'Pipe took {elapsed_hours} hours and {elapsed_minutes} minutes.')

        fold_start = time.time()

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

        for value in pred_test:
            with open(f'nc_5K_predictions_cm_test.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([value])

        for value in y_test:
            with open(f'nc_5K_targets_cm_test.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([value])

        train_loss = loss_fn(pred_train_l.unsqueeze(0), y_train_l.unsqueeze(0))
        correct = pred_train_l.eq(y_train_l).sum().item()
        train_acc = correct / y_train_l.size(0)

        test_loss = loss_fn(pred_test_l.unsqueeze(0), y_test_l.unsqueeze(0))
        correct = pred_test_l.eq(y_test_l).sum().item()
        test_acc = correct / y_test_l.size(0)

        with open(f'nc_5K_losses_per_fold_train.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([train_loss.item()])

        with open(f'nc_5K_losses_per_fold_test.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([test_loss.item()])

        with open(f'nc_5K_acc_per_fold_train.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([train_acc])

        with open(f'nc_5K_acc_per_fold_test.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([test_acc])

        print(f'test acc at {test_acc} in fold {i}')

        fold_end = time.time()
        elapsed_time = fold_end - fold_start
        elapsed_hours = int(elapsed_time/3600)
        elapsed_minutes = int((elapsed_time % 3600)/60)
        print (f'Fold took {elapsed_hours} hours and {elapsed_minutes} minutes.')

    with open('nc_5K_targets_cm_test.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        targets_cm_test = [int(float(item)) for sublist in data for item in sublist]

    with open('nc_5K_predictions_cm_test.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        predictions_cm_test = [int(float(item)) for sublist in data for item in sublist]

    print(f'Average test accuracy is {accuracy_score(targets_cm_test, predictions_cm_test)}.')

    confusion_matrix = confusion_matrix(targets_cm_test, predictions_cm_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    fig, ax = plt.subplots(figsize=(30, 30))
    disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
    plt.savefig(f'neucube_confusion_matrix_{chosen_dataset}_5K.png')

    print('Confusion Matrix Saved')

    with open('nc_5K_losses_per_fold_train.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        losses_per_fold_train = [int(float(item)) for sublist in data for item in sublist]

    with open('nc_5K_losses_per_fold_test.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        losses_per_fold_test = [int(float(item)) for sublist in data for item in sublist]

    with open('nc_5K_acc_per_fold_train.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        acc_per_fold_train = [int(float(item)) for sublist in data for item in sublist]

    with open('nc_5K_acc_per_fold_test.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        acc_per_fold_test = [int(float(item)) for sublist in data for item in sublist]

    x = [i for i in range(1,6)]

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=x,
                             y=losses_per_fold_train,
                             mode='lines',
                             name='Losses per Fold'))
    fig1.add_trace(go.Scatter(x=x,
                             y=acc_per_fold_train,
                             mode='lines',
                             name='Accuracies per Fold'))

    fig1.update_layout(title=f"NeuCube 5K {chosen_dataset} Train Loss Acc")
    fig1.write_image(f"neucube_5K_train_loss_acc_{chosen_dataset}.png")

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=x,
                              y=losses_per_fold_test,
                              mode='lines',
                              name='Losses per Fold'))
    fig2.add_trace(go.Scatter(x=x,
                              y=acc_per_fold_test,
                              mode='lines',
                              name='Accuracies per Fold'))

    fig2.update_layout(title=f"NeuCube 5K {chosen_dataset} Test Loss Acc")
    fig2.write_image(f"neucube_5K_test_loss_acc_{chosen_dataset}.png")

    print('Loss and Accuracy Graphs Saved')

    with open('nc_5K_losses_per_fold_train.csv', mode='r') as file1, \
        open('nc_5K_losses_per_fold_test.csv', mode='r') as file2, \
        open('nc_5K_acc_per_fold_train.csv', mode='r') as file3, \
        open('nc_5K_acc_per_fold_test.csv', mode='r') as file4, \
        open(f'nc_5K_{chosen_dataset}_variable_printout.csv', mode='w', newline='') as output:

        reader1 = csv.reader(file1)
        reader2 = csv.reader(file2)
        reader3 = csv.reader(file3)
        reader4 = csv.reader(file4)

        writer = csv.writer(output)

        header = ['Fold Losses Train',
                  'Fold Losses Test',
                  'Fold Accs Train',
                  'Fold Accs Test']

        writer.writerow(header)

        for value1, value2, value3, value4 in zip(reader1, reader2, reader3, reader4):
            writer.writerow(value1 + value2 + value3 + value4)

    os.remove('nc_5K_losses_per_fold_train.csv')
    os.remove('nc_5K_losses_per_fold_test.csv')
    os.remove('nc_5K_acc_per_fold_train.csv')
    os.remove('nc_5K_acc_per_fold_test.csv')

# record how long training took
end_time = time.time()

elapsed_time = end_time - start_time
elapsed_hours = int(elapsed_time/3600)
elapsed_minutes = int((elapsed_time % 3600)/60)

print (f'Training took {elapsed_hours} hours and {elapsed_minutes} minutes.')