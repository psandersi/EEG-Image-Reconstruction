import numpy as np
import os
import warnings
os.chdir('..')
warnings.filterwarnings('ignore')

# Firstly import the class of dataset
from Scripts.Data_Loader import EIRDataset
# function for getting samples on choosen subject id's
# if get_pixel in format (x, y) specified will be returned only label for choosen pixel instead of full picture
def get_sample(EIR_Dataset: EIRDataset, subj_id: int | list = None, get_pixel: bool | tuple = False):
    meta = []
    y = []
    X = []
    labels = []
    for i in range(len(EIR_Dataset)): 
        eeg_sample, eye_sample, metadata, label, img = EIR_Dataset[i]
        if subj_id == None:
            meta.append(metadata)
            y.append(img.flatten())
            labels.append(label)
            X.append(eeg_sample.get_data())
        if isinstance(subj_id, list):
            if metadata['subject_id'] in subj_id:
                meta.append(metadata)
                y.append(img.flatten())
                labels.append(label)
                X.append(eeg_sample.get_data())
        if isinstance(subj_id, int):
            if metadata['subject_id'] == subj_id:
                meta.append(metadata)
                y.append(img.flatten())
                labels.append(label)
                X.append(eeg_sample.get_data())
    if get_pixel:
        y = [t[get_pixel[0]*6 + get_pixel[1]] for t in y]
    X = np.stack(X)
    y = np.array(y)
    labels = np.array(labels)
    return X, y, labels

# choose subject and trials 
def get_sample_choosen_trial(EIR_Dataset: EIRDataset, subj_id: list = None, choosen_trial: list = None):
    meta = []
    y = []
    X = []
    labels = []
    for i in range(len(EIR_Dataset)): 
        eeg_sample, eye_sample, metadata, label, img = EIR_Dataset[i]
        for i in range(len(subj_id)):
            if metadata['subject_id'] == subj_id[i] and metadata['trial_id'] == choosen_trial[i]:
                meta.append(metadata)
                y.append(img.flatten())
                labels.append(label)
                X.append(eeg_sample.get_data())
    X = np.stack(X)
    y = np.array(y)
    labels = np.array(labels)
    return X, y, labels

# get all subjects excluding specified. Also can be done using get_sample, of course
def get_sample_exclude(EIR_Dataset: EIRDataset, subj_id: int):
    meta = []
    y = []
    X = []
    labels = []
    for i in range(len(EIR_Dataset)): 
        eeg_sample, eye_sample, metadata, label, img = EIR_Dataset[i]
        if metadata['subject_id'] != subj_id:
            meta.append(metadata)
            y.append(img.flatten())
            labels.append(label)
            X.append(eeg_sample.get_data())
    X = np.stack(X)
    y = np.array(y)
    labels = np.array(labels)
    return X, y, labels

# Get all label-image pairs
def get_target_pairs(EIR_Dataset: EIRDataset):
    targs = {}
    for i in range(len(EIR_Dataset)):
        eeg_sample, eye_sample, metadata, label, img = EIR_Dataset[i]
        if label in targs.keys():
            if not np.array_equal(targs[label], img): print('ахтунг')
        else:
            targs[label] = img
    return targs