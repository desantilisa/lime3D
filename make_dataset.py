#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:24:11 2023

@author: lisadesanti
"""

import argparse
import os
import math
import numpy as np
import random
import pandas 
import pydicom
import SimpleITK as sitk

import torch
from torch import Tensor
import torch.optim
import torch.utils.data
from torch.utils.data import WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

    

def get_ppmi_datscan_brains_paths(
        dataset_path = "/home/lisadesanti/DeepLearning/PPMI/DaTSCAN/",
        metadata_path = "/home/lisadesanti/DeepLearning/PPMI/DaTScan_Visual_Interpretation_Results_10Jul2024.csv",
        dic_classes = {"positive":0, "negative":1},
        set_type = 'train',
        shuffle = True,
        n_fold = 5, 
        current_fold = 1,
        test_split = 0.2,
        seed = 42):
    
    """ Get a list with directories and labels of the Training/Validation/Test 
    set splitted performing 5-fold cross-validation
    
    Args:
        
        - dataset_path: Path of folder containing input images
        - metadata_path: Path of folder containing metadata (including image labels) 
        - dic_classes: Dictionary of classes considered, "class_name":label
        - set_type: str used to identify Training/Validation/Test set:
            set_type \in {"train", "val", "test"}
        - current_fold: Current fold 
        - n_fold: n° fold in kfold validation
        - test_split: % of split used
    
    Returns: 
        
        - ndarray of Training/Validation/Test/All data directories
        - ndarray of image labels
        - dict containing dataset information
    """
    
    metadata_paths = pandas.read_csv(metadata_path)
    subjs_id = os.listdir(dataset_path)
    img_directiories_dic = []
    labels = []
    
    for subj_id in subjs_id:
        
        subj_path = os.path.join(dataset_path, subj_id)
        process_types = os.listdir(subj_path)
        
        for process_type in process_types:
            
            process_path = os.path.join(subj_path, process_type)
            acq_dates = os.listdir(process_path)
            
            for acq_date in acq_dates:
                
                acq_path = os.path.join(process_path, acq_date)
                img_id = os.listdir(acq_path)[0]
                img_folder = os.path.join(acq_path, img_id)
                img_file = os.listdir(img_folder)[0]
                img_directory = os.path.join(img_folder, img_file)
                
                # Select the DaTSCAN visual interpretation(s) of current subject
                datscan_vis_int_patno = metadata_paths.loc[metadata_paths['PATNO'] == int(subj_id)]
                
                if len(datscan_vis_int_patno)>0:
                    
                    # Select the DaTSCAN visual interpretation(s) of the current acquisition
                    selected_date = acq_date[5:7] + '/' + acq_date[0:4]
                    datscan_vis_int_patno_date = datscan_vis_int_patno.loc[datscan_vis_int_patno['DATSCAN_DATE'] == selected_date]
                    
                    if len(datscan_vis_int_patno_date)>0:

                        label = datscan_vis_int_patno_date['DATSCAN_VISINTRP']
                        label = label.to_numpy()[0]
                        img_directory_dic = {"ROOT":dataset_path, "LABEL":label, "SUBJ":subj_id, "PREPROC":process_type, "DATE":acq_date, "EXAM_ID":img_id, "FILENAME":img_file}
                        
                        if label in dic_classes.keys():
                            # Save only image directory of the visually inspected DaTSCAN
                            img_directiories_dic.append(img_directory_dic)
                            
                        else:
                            # Unexpected label
                            continue
                    else:
                        # The current DaTSCAN acquisition was not visually inspected
                        continue
                else:
                    # The current subject does not have a visually inspected DaTSCAN
                    continue
    
    # Dataframe of ADNI directories
    directory_dataframe = pandas.DataFrame(img_directiories_dic)
    #all_subj = list(set(directory_dataframe["SUBJ"]))
    labels = directory_dataframe["LABEL"]
    img_num = len(img_directiories_dic)
    
    # Split Dataset into Training(+ Valid) and Test set 
    # Shuffle (reproducible) and select the last 20% of the dataset 
    X_train_val_df, X_test_df, y_train_val, y_test = train_test_split(
        directory_dataframe, 
        labels, 
        test_size = test_split, 
        shuffle = True, 
        random_state = seed, 
        stratify = labels
        ); # with shuffle False stratify is not support
    
    # Check to not have data (exams) from the same subjects both in the 
    # training and validation sets
    subj_train = np.array(X_train_val_df["SUBJ"])
    subj_test = np.array(X_test_df["SUBJ"])
    dup_subjects = np.intersect1d(subj_train, subj_test)

    # If a subjects has data in both sets move data to the training set
    for dup_subj in dup_subjects:

        dup_subj_test = X_test_df.loc[X_test_df["SUBJ"]==dup_subj]
        id_dup_subj_test = np.array(dup_subj_test.index)
        to_train = X_test_df.loc[id_dup_subj_test]

        # Test set (without duplicated subjects)
        X_test_df = X_test_df.drop(id_dup_subj_test)
        X_test_df = X_test_df.sort_values("SUBJ")
        y_test = X_test_df["LABEL"]
        
        # Training+Validation set (without duplicated subjects)
        X_train_val_df = pandas.concat([X_train_val_df, to_train], ignore_index=True)
        X_train_val_df = X_train_val_df.sort_values("SUBJ")
        y_train_val = X_train_val_df["LABEL"]
        
    # Perform k-fold crossvalidation on the Training + Validation
    
    # Create a new index
    new_index = range(0, 0 + len(X_train_val_df))
    # Reindex the DataFrame
    X_train_val_df.index = new_index
    y_train_val.index = new_index
    skf = StratifiedKFold(n_splits=n_fold, random_state=None, shuffle=False)
    
    kfold_generator = skf.split(X_train_val_df, y_train_val)

    for i in range(current_fold):
        
        # Split into Training and Validation set
        train_index, val_index = next(kfold_generator)
        X_train_df  = X_train_val_df.loc[train_index]
        X_val_df = X_train_val_df.loc[val_index]
        y_train = y_train_val[train_index]
        y_val = y_train_val[val_index]
   
        # Check to not have data (exams) of the same subjects both in the training and test sets
        subj_train = np.array(X_train_df["SUBJ"])
        subj_val = np.array(X_val_df["SUBJ"])
        dup_subjects = np.intersect1d(subj_train, subj_val)
   
        # If a subjects has data in both sets move data to the training set
        # (this is an arbitrary choice)
        for dup_subj in dup_subjects:
   
            dup_subj_val = X_val_df.loc[X_val_df["SUBJ"]==dup_subj]
            id_dup_subj_val = np.array(dup_subj_val.index)
            to_train = X_val_df.loc[id_dup_subj_val]
   
            # Validation set (without duplicated subjects)
            X_val_df = X_val_df.drop(id_dup_subj_val)
            X_val_df = X_val_df.sort_values("SUBJ")
            y_val = X_val_df["LABEL"]
            
            # Training set (without duplicated subjects)
            X_train_df = pandas.concat([X_train_df, to_train], ignore_index=True)
            X_train_df = X_train_df.sort_values("SUBJ")
            y_train = X_train_df["LABEL"]

    # Check to not have data (exams) from the same subjects both in the 
    # training and validation sets
    subj_train = np.array(X_train_df["SUBJ"])
    subj_val = np.array(X_val_df["SUBJ"])
    dup_subjects = np.intersect1d(subj_train, subj_val)

    # If a subjects has data in both sets move data to the training set
    # (this is an arbitrary choice)
    for dup_subj in dup_subjects:

        dup_subj_val = X_val_df.loc[X_val_df["SUBJ"]==dup_subj]
        id_dup_subj_val = np.array(dup_subj_val.index)
        to_train = X_val_df.loc[id_dup_subj_val]

        # Vslidation set (without duplicated subjects)
        X_val_df = X_val_df.drop(id_dup_subj_val)
        X_val_df = X_val_df.sort_values("SUBJ")
        y_val = X_val_df["LABEL"]
        
        # Training+Validation set (without duplicated subjects)
        X_train_df = pandas.concat([X_train_df, to_train], ignore_index=True)
        X_train_df = X_train_df.sort_values("SUBJ")
        y_train = X_train_df["LABEL"]
    
    subj_train = np.array(X_train_df["SUBJ"])
    n_train = len(X_train_df)
    train_sn = y_train.tolist().count('negative')
    train_sp = y_train.tolist().count('positive')
    
    subj_val = np.array(X_val_df["SUBJ"])
    n_val = len(X_val_df)
    val_sn = y_val.tolist().count('negative')
    val_sp = y_val.tolist().count('positive')
    
    subj_test = np.array(X_test_df["SUBJ"])
    n_test = len(X_test_df)
    test_sn = y_test.tolist().count('negative')
    test_sp = y_test.tolist().count('positive')
    
    # dizionario con info 
    dataset_info = {
        "train_subj":subj_train.tolist(), 
        "n_train":n_train, "train_sn":train_sn, "train_sp":train_sp,
        "val_subj":subj_val.tolist(),
        "n_val":n_val, "val_sn":val_sn, "val_sp":val_sp,
        "test_subj":subj_test.tolist(),
        "n_test":n_test, "test_sn":test_sn, "test_sp":test_sp}
    
    dup_subjects_train_val = np.intersect1d(subj_train, subj_val)
    dup_subjects_train_test = np.intersect1d(subj_train, subj_test)
    dup_subjects_val_test = np.intersect1d(subj_val, subj_test)
    
    # Check data leackage issue
    if len(dup_subjects_train_val) or len(dup_subjects_train_test) or len(dup_subjects_val_test):
        print('Data Leackage occurred!! ')
        return
    
    X = np.array(directory_dataframe['ROOT']) + np.array(['/']*img_num) + \
        np.array(directory_dataframe['SUBJ']) +  np.array(['/']*img_num) + \
        np.array(directory_dataframe['PREPROC']) + np.array(['/']*img_num) + \
        np.array(directory_dataframe['DATE']) + np.array(['/']*img_num) + \
        np.array(directory_dataframe['EXAM_ID']) + np.array(['/']*img_num) + \
        np.array(directory_dataframe['FILENAME'])
    y = np.array(labels)
    
    X_train = np.array(X_train_df['ROOT']) + np.array(['/']*n_train) + \
              np.array(X_train_df['SUBJ']) + np.array(['/']*n_train) + \
              np.array(X_train_df['PREPROC']) + np.array(['/']*n_train) + \
              np.array(X_train_df['DATE']) + np.array(['/']*n_train) + \
              np.array(X_train_df['EXAM_ID']) + np.array(['/']*n_train) + \
              np.array(X_train_df['FILENAME'])
    y_train = np.array(y_train)
    y_train = np.array([dic_classes[yi] for yi in y_train])
    
    X_val = np.array(X_val_df['ROOT']) + np.array(['/']*n_val) + \
            np.array(X_val_df['SUBJ']) + np.array(['/']*n_val) + \
            np.array(X_val_df['PREPROC']) + np.array(['/']*n_val) + \
            np.array(X_val_df['DATE']) +  np.array(['/']*n_val) + \
            np.array(X_val_df['EXAM_ID']) + np.array(['/']*n_val) + \
            np.array(X_val_df['FILENAME'])
    y_val = np.array(y_val)
    y_val = np.array([dic_classes[yi] for yi in y_val])
    
    X_test = np.array(X_test_df['ROOT']) + np.array(['/']*n_test) + \
             np.array(X_test_df['SUBJ']) + np.array(['/']*n_test) + \
             np.array(X_test_df['PREPROC']) + np.array(['/']*n_test) + \
             np.array(X_test_df['DATE']) + np.array(['/']*n_test) + \
             np.array(X_test_df['EXAM_ID']) + np.array(['/']*n_test) + \
             np.array(X_test_df['FILENAME'])
    y_test = np.array(y_test)
    y_test = np.array([dic_classes[yi] for yi in y_test])

    # Data shuffling 
    if shuffle:
        
        rng = np.random.default_rng(seed)
        shuffled_index = np.arange(n_train)
        rng.shuffle(shuffled_index)
        # Shuffled dataset
        X_train = X_train[shuffled_index]
        y_train = y_train[shuffled_index]
    
        rng = np.random.default_rng(seed)
        shuffled_index = np.arange(n_val)
        rng.shuffle(shuffled_index)
        # Shuffled dataset
        X_val = X_val[shuffled_index]          
        y_val = y_val[shuffled_index]
        
        rng = np.random.default_rng(seed)
        shuffled_index = np.arange(n_test)
        rng.shuffle(shuffled_index)
        # Shuffled dataset
        X_test = X_test[shuffled_index] 
        y_test = y_test[shuffled_index]
    
    if set_type == 'train':
        return X_train, y_train, dataset_info
    elif set_type == 'val':
        return X_val, y_val, dataset_info
    elif set_type == 'test':
        return X_test , y_test, dataset_info
    else:
        return X, y, dataset_info    


def get_ppmi_datscan_parkinson_paths(
        dataset_path = "/home/lisadesanti/DeepLearning/PPMI/DaTSCAN/",
        metadata_path = "/home/lisadesanti/DeepLearning/PPMI/PPMI_DaTSCAN_7_11_2024.csv",
        dic_classes = {'Control':0, 'GenReg PD':1, 'PD':2, 'SWEDD':3},
        set_type = 'train',
        shuffle = True,
        n_fold = 5, 
        current_fold = 1,
        test_split = 0.2,
        seed = 42):
    
    """ Get a list with directories and labels of the Training/Validation/Test 
    set splitted performing 5-fold cross-validation
    
    Args:
        
        - dataset_path: Path of folder containing input images
        - metadata_path: Path of folder containing metadata (including image labels) 
        - dic_classes: Dictionary of classes considered, "class_name":label
        - set_type: str used to identify Training/Validation/Test set:
            set_type \in {"train", "val", "test"}
        - current_fold: Current fold 
        - n_fold: n° fold in kfold validation
        - test_split: % of split used
    
    Returns: 
        
        - ndarray of Training/Validation/Test/All data directories
        - ndarray of image labels
        - dict containing dataset information
    """
    
    metadata_paths = pandas.read_csv(metadata_path)
    subjs_id = os.listdir(dataset_path)
    img_directiories_dic = []
    labels = []
    
    for subj_id in subjs_id:
        
        subj_path = os.path.join(dataset_path, subj_id)
        process_types = os.listdir(subj_path)
        
        for process_type in process_types:
            
            process_path = os.path.join(subj_path, process_type)
            acq_dates = os.listdir(process_path)
            
            for acq_date in acq_dates:
                
                acq_path = os.path.join(process_path, acq_date)
                img_id = os.listdir(acq_path)[0]
                img_folder = os.path.join(acq_path, img_id)
                img_file = os.listdir(img_folder)[0]
                img_directory = os.path.join(img_folder, img_file)
                dicom_file = pydicom.dcmread(img_directory)
                volume = dicom_file.pixel_array
                
                label = metadata_paths.loc[metadata_paths["Image Data ID"]==img_id]['Group']
                label = label.to_numpy()[0]
                img_directory_dic = {"ROOT":dataset_path, "LABEL":label, "SUBJ":subj_id, "PREPROC":process_type, "DATE":acq_date, "EXAM_ID":img_id, "FILENAME":img_file}
                
                if label in dic_classes.keys() and volume.dtype == 'int16' and volume.shape == (91,109,91):
                    # Save only image directory of the desidered DaTSCAN
                    # (there are two abnormal scans, uint16 and uncorred shape)
                    img_directiories_dic.append(img_directory_dic)
                    
                else:
                    # Unexpected label
                    continue

    
    # Dataframe of ADNI directories
    directory_dataframe = pandas.DataFrame(img_directiories_dic)
    all_subj = list(set(directory_dataframe["SUBJ"]))
    labels = directory_dataframe["LABEL"]
    img_num = len(img_directiories_dic)
    
    # Split Dataset into Training(+ Valid) and Test set 
    # Shuffle (reproducible) and select the last 20% of the dataset 
    X_train_val_df, X_test_df, y_train_val, y_test = train_test_split(
        directory_dataframe, 
        labels, 
        test_size = test_split, 
        shuffle = True, 
        random_state = seed, 
        stratify = labels
        ); # with shuffle False stratify is not support
    
    # Check to not have data (exams) from the same subjects both in the 
    # training and validation sets
    subj_train = np.array(X_train_val_df["SUBJ"])
    subj_test = np.array(X_test_df["SUBJ"])
    dup_subjects = np.intersect1d(subj_train, subj_test)

    # If a subjects has data in both sets move data to the training set
    for dup_subj in dup_subjects:

        dup_subj_test = X_test_df.loc[X_test_df["SUBJ"]==dup_subj]
        id_dup_subj_test = np.array(dup_subj_test.index)
        to_train = X_test_df.loc[id_dup_subj_test]

        # Test set (without duplicated subjects)
        X_test_df = X_test_df.drop(id_dup_subj_test)
        X_test_df = X_test_df.sort_values("SUBJ")
        y_test = X_test_df["LABEL"]
        
        # Training+Validation set (without duplicated subjects)
        X_train_val_df = pandas.concat([X_train_val_df, to_train], ignore_index=True)
        X_train_val_df = X_train_val_df.sort_values("SUBJ")
        y_train_val = X_train_val_df["LABEL"]
        
    # Perform k-fold crossvalidation on the Training + Validation
    
    # Create a new index
    new_index = range(0, 0 + len(X_train_val_df))
    # Reindex the DataFrame
    X_train_val_df.index = new_index
    y_train_val.index = new_index
    skf = StratifiedKFold(n_splits=n_fold, random_state=None, shuffle=False)
    
    kfold_generator = skf.split(X_train_val_df, y_train_val)

    for i in range(current_fold):
        
        # Split into Training and Validation set
        train_index, val_index = next(kfold_generator)
        X_train_df  = X_train_val_df.loc[train_index]
        X_val_df = X_train_val_df.loc[val_index]
        y_train = y_train_val[train_index]
        y_val = y_train_val[val_index]
   
        # Check to not have data (exams) of the same subjects both in the training and test sets
        subj_train = np.array(X_train_df["SUBJ"])
        subj_val = np.array(X_val_df["SUBJ"])
        dup_subjects = np.intersect1d(subj_train, subj_val)
   
        # If a subjects has data in both sets move data to the training set
        # (this is an arbitrary choice)
        for dup_subj in dup_subjects:
   
            dup_subj_val = X_val_df.loc[X_val_df["SUBJ"]==dup_subj]
            id_dup_subj_val = np.array(dup_subj_val.index)
            to_train = X_val_df.loc[id_dup_subj_val]
   
            # Validation set (without duplicated subjects)
            X_val_df = X_val_df.drop(id_dup_subj_val)
            X_val_df = X_val_df.sort_values("SUBJ")
            y_val = X_val_df["LABEL"]
            
            # Training set (without duplicated subjects)
            X_train_df = pandas.concat([X_train_df, to_train], ignore_index=True)
            X_train_df = X_train_df.sort_values("SUBJ")
            y_train = X_train_df["LABEL"]

    # Check to not have data (exams) from the same subjects both in the 
    # training and validation sets
    subj_train = np.array(X_train_df["SUBJ"])
    subj_val = np.array(X_val_df["SUBJ"])
    dup_subjects = np.intersect1d(subj_train, subj_val)

    # If a subjects has data in both sets move data to the training set
    # (this is an arbitrary choice)
    for dup_subj in dup_subjects:

        dup_subj_val = X_val_df.loc[X_val_df["SUBJ"]==dup_subj]
        id_dup_subj_val = np.array(dup_subj_val.index)
        to_train = X_val_df.loc[id_dup_subj_val]

        # Vslidation set (without duplicated subjects)
        X_val_df = X_val_df.drop(id_dup_subj_val)
        X_val_df = X_val_df.sort_values("SUBJ")
        y_val = X_val_df["LABEL"]
        
        # Training+Validation set (without duplicated subjects)
        X_train_df = pandas.concat([X_train_df, to_train], ignore_index=True)
        X_train_df = X_train_df.sort_values("SUBJ")
        y_train = X_train_df["LABEL"]
    
    subj_train = np.array(X_train_df["SUBJ"])
    n_train = len(X_train_df)
    {'Control':0, 'GenReg PD':1, 'PD':2, 'SWEDD':3}
    train_ct = y_train.tolist().count('Control')
    train_gr = y_train.tolist().count('GenReg PD')
    train_pd = y_train.tolist().count('PD')
    train_swedd = y_train.tolist().count('SWEDD')
    
    subj_val = np.array(X_val_df["SUBJ"])
    n_val = len(X_val_df)
    val_ct = y_val.tolist().count('Control')
    val_gr = y_val.tolist().count('GenReg PD')
    val_pd = y_val.tolist().count('PD')
    val_swedd = y_val.tolist().count('SWEDD')
    
    subj_test = np.array(X_test_df["SUBJ"])
    n_test = len(X_test_df)
    test_ct = y_test.tolist().count('Control')
    test_gr = y_test.tolist().count('GenReg PD')
    test_pd = y_test.tolist().count('PD')
    test_swedd = y_test.tolist().count('SWEDD')
    
    # dizionario con info 
    dataset_info = {
        "train_subj":subj_train.tolist(), 
        "n_train":n_train, 
        "train_ct":train_ct, 
        "train_gr":train_gr,
        "train_pd":train_pd, 
        "train_swedd":train_swedd,
        "val_subj":subj_val.tolist(),
        "n_val":n_val, 
        "val_ct":val_ct, 
        "val_gr":val_gr,
        "val_pd":val_pd, 
        "val_swedd":val_swedd,
        "test_subj":subj_test.tolist(),
        "n_test":n_test, 
        "test_ct":test_ct, 
        "test_gr":test_gr,
        "test_pd":test_pd, 
        "test_swedd":test_swedd}
    
    dup_subjects_train_val = np.intersect1d(subj_train, subj_val)
    dup_subjects_train_test = np.intersect1d(subj_train, subj_test)
    dup_subjects_val_test = np.intersect1d(subj_val, subj_test)
    
    # Check data leackage issue
    if len(dup_subjects_train_val) or len(dup_subjects_train_test) or len(dup_subjects_val_test):
        print('Data Leackage occurred!! ')
        return
    
    X = np.array(directory_dataframe['ROOT']) + np.array(['/']*img_num) + \
        np.array(directory_dataframe['SUBJ']) +  np.array(['/']*img_num) + \
        np.array(directory_dataframe['PREPROC']) + np.array(['/']*img_num) + \
        np.array(directory_dataframe['DATE']) + np.array(['/']*img_num) + \
        np.array(directory_dataframe['EXAM_ID']) + np.array(['/']*img_num) + \
        np.array(directory_dataframe['FILENAME'])
    y = np.array(labels)
    
    X_train = np.array(X_train_df['ROOT']) + np.array(['/']*n_train) + \
              np.array(X_train_df['SUBJ']) + np.array(['/']*n_train) + \
              np.array(X_train_df['PREPROC']) + np.array(['/']*n_train) + \
              np.array(X_train_df['DATE']) + np.array(['/']*n_train) + \
              np.array(X_train_df['EXAM_ID']) + np.array(['/']*n_train) + \
              np.array(X_train_df['FILENAME'])
    y_train = np.array(y_train)
    y_train = np.array([dic_classes[yi] for yi in y_train])
    
    X_val = np.array(X_val_df['ROOT']) + np.array(['/']*n_val) + \
            np.array(X_val_df['SUBJ']) + np.array(['/']*n_val) + \
            np.array(X_val_df['PREPROC']) + np.array(['/']*n_val) + \
            np.array(X_val_df['DATE']) +  np.array(['/']*n_val) + \
            np.array(X_val_df['EXAM_ID']) + np.array(['/']*n_val) + \
            np.array(X_val_df['FILENAME'])
    y_val = np.array(y_val)
    y_val = np.array([dic_classes[yi] for yi in y_val])
    
    X_test = np.array(X_test_df['ROOT']) + np.array(['/']*n_test) + \
             np.array(X_test_df['SUBJ']) + np.array(['/']*n_test) + \
             np.array(X_test_df['PREPROC']) + np.array(['/']*n_test) + \
             np.array(X_test_df['DATE']) + np.array(['/']*n_test) + \
             np.array(X_test_df['EXAM_ID']) + np.array(['/']*n_test) + \
             np.array(X_test_df['FILENAME'])
    y_test = np.array(y_test)
    y_test = np.array([dic_classes[yi] for yi in y_test])

    # Data shuffling 
    if shuffle:
        
        rng = np.random.default_rng(seed)
        shuffled_index = np.arange(n_train)
        rng.shuffle(shuffled_index)
        # Shuffled dataset
        X_train = X_train[shuffled_index]
        y_train = y_train[shuffled_index]
    
        rng = np.random.default_rng(seed)
        shuffled_index = np.arange(n_val)
        rng.shuffle(shuffled_index)
        # Shuffled dataset
        X_val = X_val[shuffled_index]          
        y_val = y_val[shuffled_index]
        
        rng = np.random.default_rng(seed)
        shuffled_index = np.arange(n_test)
        rng.shuffle(shuffled_index)
        # Shuffled dataset
        X_test = X_test[shuffled_index] 
        y_test = y_test[shuffled_index]
    
    if set_type == 'train':
        return X_train, y_train, dataset_info
    elif set_type == 'val':
        return X_val, y_val, dataset_info
    elif set_type == 'test':
        return X_test , y_test, dataset_info
    else:
        return X, y, dataset_info    
    
    
class BrainDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset_type: str,
            dic_classes: dict,
            transform = None,
            set_type = 'train',
            n_fold = 5,
            current_fold = 1,
            test_split = 0.2,
            seed = 42):
        
        self.set_type = set_type      # tranformation's type
        
        if dataset_type == "PPMI_DaTSCAN_Parkinson" or dataset_type == "PPMI_DaTSCAN_Parkinson_8slices":
            
            # Predict clinical diagnosis (Parkinson)
            dataset = get_ppmi_datscan_parkinson_paths(
                dic_classes = dic_classes,
                set_type = self.set_type,
                n_fold = n_fold,
                current_fold = current_fold,
                test_split = test_split,
                seed = seed)
        else:
            
            # Predict SPECT visual interpretation
            dataset = get_ppmi_datscan_brains_paths(
                dic_classes = dic_classes,
                set_type = self.set_type,
                n_fold = n_fold,
                current_fold = current_fold,
                test_split = test_split,
                seed = seed) 
            
        self.dataset_type = dataset_type
        
        self.img_dir = dataset[0]      # ndarray with images directories
        self.img_labels = dataset[1]   # ndarray with images labels
        self.dataset_info = dataset[2] # dictionary with dataset info
        
        self.transform = transform[set_type] # images transformation
        self.classes = list(dic_classes.keys())
        self.class_to_idx = dic_classes
        
        
    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        
        img_path = self.img_dir[idx]
        label = self.img_labels[idx]
        
        # PET volume, DICOM
        dicom_file = pydicom.dcmread(img_path)
        volume = dicom_file.pixel_array
        # numpy -> tensor
        volume = torch.tensor(volume)
        volume = torch.unsqueeze(volume,0)
        volume = volume.float()
        
        if self.dataset_type == "PPMI_DaTSCAN_Parkinson_8slices":
            volume = volume[:,37:45,:,:]
        
        
        if self.transform:
            volume = self.transform(volume)
            img_min = volume.min()
            img_max = volume.max()
            volume = (volume-img_min)/(img_max-img_min)

        return volume, label

    


