'''
This file calculates performance metrics and saves the confusion matrices for all networks in both training paradigms, for ease of review.
It assumes results from each network run have been placed in an appropriately labelled subdirectory of a master results directory.
'''

import csv
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# define the master path to the results files
master_path = "/share/klab/ppowell/EEG_CN_Classification/results_5_95_R2"

print()

# set the parameters for pulling the results for EEGChannelNet run on the full dataset in the cross validation training paradigm
# ECN Full 5K
network = 'EEGChannelNet'
dataset = '5_95'
ref = 'ecn_full_5K'
best_epoch = 'N/A'

# pull the predictions and targets from the appropriate offline csv file for creation of the confusion matrices
with open(f'{master_path}/{ref}/{network}_{dataset}_5K_predictions_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_{dataset}_5K_targets_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

# print the confusion matrix (in this case across the five folds)
# confusion matrix code modelled after that found in the original NeuCube code
cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

# use the classification_report function to generate a report on the F1 score, prediction, and recall
# additionally calculate the overall classification accuracy and print it
print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

# list the classification accuracies by class
for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over the arrays for the predictions and targets
    for x, y in zip(pred, target):

        # if both values are equal to the given (class), increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count

    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    # if the network never guessed a certain class, the formula results in division by zero, so we set the results for this class to undefined
    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")

print()

### repeat the same procedure for the remaining network/dataset/training paradigm combinations ###

# for full learning paradigms, the epoch at which the model achieved its highest validation accuracy must be entered to ensure printing of the appropriate confusion matrix
# the best epoch can be determined from the printouts for the respective training run

# ECN Full Learning

network = 'EEGChannelNet'
dataset = '5_95'
ref = 'ecn_full_200L'
best_epoch = 45

with open(f'{master_path}/{ref}/{network}_{dataset}_200L_predictions_cm_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_{dataset}_200L_targets_cm_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count

    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")

print()

# ECN Quarter 5K

network = 'EEGChannelNet'
dataset = '5_95_quarter'
ref = 'ecn_q_5K'
best_epoch = 'N/A'

with open(f'{master_path}/{ref}/{network}_{dataset}_5K_predictions_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_{dataset}_5K_targets_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count

    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")

print()

# ECN Quarter Learning

network = 'EEGChannelNet'
dataset = '5_95_quarter'
ref = 'ecn_q_200L'
best_epoch = 40

with open(f'{master_path}/{ref}/{network}_{dataset}_200L_predictions_cm_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_{dataset}_200L_targets_cm_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count
    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")


#############################################################


print()

# lstm Full 5K
network = 'lstm'
dataset = '5_95'
ref = 'lstm_full_5K'
best_epoch = 'N/A'

with open(f'{master_path}/{ref}/{network}_{dataset}_5K_predictions_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_{dataset}_5K_targets_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count

    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")

print()

# ECN Full Learning

network = 'lstm'
dataset = '5_95'
ref = 'lstm_full_200L'
best_epoch = 21

with open(f'{master_path}/{ref}/{network}_{dataset}_200L_predictions_cm_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_{dataset}_200L_targets_cm_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count

    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")

print()

# ECN Quarter 5K

network = 'lstm'
dataset = '5_95_quarter'
ref = 'lstm_q_5K'
best_epoch = 'N/A'

with open(f'{master_path}/{ref}/{network}_{dataset}_5K_predictions_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_{dataset}_5K_targets_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count

    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")

print()

# ECN Quarter Learning

network = 'lstm'
dataset = '5_95_quarter'
ref = 'lstm_q_200L'
best_epoch = 33

with open(f'{master_path}/{ref}/{network}_{dataset}_200L_predictions_cm_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_{dataset}_200L_targets_cm_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count
    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")


#########################################################################

print()

# ECN Full 5K
network = 'lstm5'
dataset = '5_95'
ref = 'lstm5_full_5K'
best_epoch = 'N/A'

with open(f'{master_path}/{ref}/{network}_{dataset}_5K_predictions_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_{dataset}_5K_targets_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count

    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")

print()

# ECN Full Learning

network = 'lstm5'
dataset = '5_95'
ref = 'lstm5_full_200L'
best_epoch = 34

with open(f'{master_path}/{ref}/{network}_{dataset}_200L_predictions_cm_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_{dataset}_200L_targets_cm_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count

    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")

print()

# ECN Quarter 5K

network = 'lstm5'
dataset = '5_95_quarter'
ref = 'lstm5_q_5K'
best_epoch = 'N/A'

with open(f'{master_path}/{ref}/{network}_{dataset}_5K_predictions_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_{dataset}_5K_targets_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count

    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")

print()

# ECN Quarter Learning

network = 'lstm5'
dataset = '5_95_quarter'
ref = 'lstm5_q_200L'
best_epoch = 41

with open(f'{master_path}/{ref}/{network}_{dataset}_200L_predictions_cm_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_{dataset}_200L_targets_cm_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count
    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")


################################################################

print()

# ECN Full 5K
network = 'lstm10'
dataset = '5_95'
ref = 'lstm10_full_5K'
best_epoch = 'N/A'

with open(f'{master_path}/{ref}/{network}_{dataset}_5K_predictions_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_{dataset}_5K_targets_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count

    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")

print()

# ECN Full Learning

network = 'lstm10'
dataset = '5_95'
ref = 'lstm10_full_200L'
best_epoch = 5

with open(f'{master_path}/{ref}/{network}_{dataset}_200L_predictions_cm_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_{dataset}_200L_targets_cm_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count

    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")

print()

# ECN Quarter 5K

network = 'lstm10'
dataset = '5_95_quarter'
ref = 'lstm10_q_5K'
best_epoch = 'N/A'

with open(f'{master_path}/{ref}/{network}_{dataset}_5K_predictions_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_{dataset}_5K_targets_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count

    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")

print()

# ECN Quarter Learning

network = 'lstm10'
dataset = '5_95_quarter'
ref = 'lstm10_q_200L'
best_epoch = 2

with open(f'{master_path}/{ref}/{network}_{dataset}_200L_predictions_cm_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_{dataset}_200L_targets_cm_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count
    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")


####################################################################

print()

# ECN Full 5K
network = 'nc'
dataset = '5_95'
ref = 'nc_full_5K'
best_epoch = 'N/A'

with open(f'{master_path}/{ref}/{network}_5K_predictions_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_5K_targets_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count

    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")

print()

# ECN Full Learning

network = 'nc'
dataset = '5_95'
ref = 'nc_full_200L'
best_epoch = 4

with open(f'{master_path}/{ref}/{network}_pred_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_target_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count

    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")

print()

# ECN Quarter 5K

network = 'nc'
dataset = '5_95_quarter'
ref = 'nc_q_5K'
best_epoch = 'N/A'

with open(f'{master_path}/{ref}/{network}_5K_predictions_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_5K_targets_cm_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count

    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")

print()

# ECN Quarter Learning

network = 'nc'
dataset = '5_95_quarter'
ref = 'nc_q_200L'
best_epoch = 7

with open(f'{master_path}/{ref}/{network}_pred_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    pred = [int(float(item)) for sublist in data for item in sublist]

with open(f'{master_path}/{ref}/{network}_target_test_{best_epoch}.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    target = [int(float(item)) for sublist in data for item in sublist]

print(f'Generating reports for {ref}, best epoch {best_epoch}:')

print()

cm = confusion_matrix(target, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(30, 30))
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
plt.savefig(f'{ref}.png')

print(classification_report(target,pred))
print(f'Overall accuracy is: {accuracy_score(target,pred)}')

for i in range(40):

    # initialize a counter variable
    count = 0
    value = i

    # loop over both arrays using zip()
    for x, y in zip(pred, target):

        # if both values are equal to the given value, increment the counter
        if x == value and y == value:
            count += 1

    true_pos = count
    if pred.count(value) != 0:
        accuracy = count / pred.count(value)

    else:
        accuracy = 'Undefined'

    print(f"Accuracy for class {i}: {accuracy}")

print('All Reports Generated and Saved')





