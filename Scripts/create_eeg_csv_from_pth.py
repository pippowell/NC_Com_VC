import torch
import pandas as pd
import glob
import csv

path = "/share/klab/datasets/EEG_Visual/"
a = torch.load(f"{path}eeg_signals_raw_with_mean_std.pth")

# Separate out the EEG data
for i in range(len(a['dataset'])):
    image = a['dataset'][i]['image']
    label = a['dataset'][i]['label']
    subject = a['dataset'][i]['subject']
    eeg_data = a['dataset'][i]['eeg']
    df = pd.DataFrame(eeg_data)
    df.to_csv(f"{path}EEG_CSV/Raw/eeg_raw{i}_image{image}_class{label}_subject{subject}.csv", index=False)

with open(f'{path}EEG_CSV/Raw/tar_class_labels.csv','w') as f:
    pass

for i in range(0,11965):
    path = f'{path}/EEG_CSV/Raw/*eeg_raw{i}_*.csv'
    files = glob.glob(path)
    file = files[0]
    with open(file,'r') as f:
        reader = csv.reader(f)
        data = list(reader)[1:]
    with open(f'{path}/EEG_CSV/Raw/sam{i}_eeg.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    class_from_filename = re.search(r'(?<=class)(\d+)', file)
    label = class_from_filename.group(1)
    label = [label]
    with open(f'{path}EEG_CSV/Raw/tar_class_labels.csv','a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(label)

print('Conversion Successful')