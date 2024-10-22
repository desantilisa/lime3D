#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:24:28 2024

@author: lisadesanti
"""


import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

from monai.transforms import (
    Compose,
    Resize,
    RandRotate,
    Affine,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandZoom,
    RandFlip,
    RepeatChannel,
    RandSpatialCrop,
)

from make_dataset import BrainDataset
from torch.utils.data import DataLoader

from model_builder import load_trained_blackbox
from test_model import eval_blackbox

try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image
from lime import lime_image3D

from plot_utils import plot_3d_slices


#%%

current_fold = 1
trained_model_key = 2

backbone_dic = {1:"resnet3D_18_kin400", 2:"resnet3D_18", 3:"convnext3D_tiny_imgnet", 4:"convnext3D_tiny"}
net = backbone_dic[1]

trained_model_dic = {1:"train_binary_vis_int_datscan/", 
                     2:"train_binary_parkinson_datscan/", 
                     3:"train_binary_parkinson_datscan_8slices/", 
                     4:"train_multiclass_parkinson_datscan/"}
trained_model = trained_model_dic[trained_model_key]

dic_classes_dic = {1: {"positive":0, "negative":1}, 
                   2: {'Control':0, 'PD':1}, 
                   3: {'Control':0, 'PD':1},
                   4: {'Control':0, 'PD':1, 'SWEDD':2}}
dic_classes = dic_classes_dic[trained_model_key]
num_classes = len(dic_classes)
out_shape = num_classes
dataset_type_dic = {1: "PPMI_DaTSCAN_visint", 
                    2: "PPMI_DaTSCAN_Parkinson", 
                    3:"PPMI_DaTSCAN_Parkinson_8slices",
                    4: "PPMI_DaTSCAN_Parkinson",}  
dataset_type = dataset_type_dic[trained_model_key]

net_dic = {"resnet3D_18_kin400":3, "resnet3D_18":1, "convnext3D_tiny_imgnet":3, "convnext3D_tiny":1} # 
n_fold = 5           # Number of fold
test_split = 0.2
seed = 42            # seed for reproducible shuffling
downscaling = 1
rows = int(109/downscaling)
cols = int(91/downscaling)
slices = int(91/downscaling) #int(91/downscaling)


root_folder = "/home/lisadesanti/DeepLearning/Parkinson/ResNet18_DaTSCAN_PPMI"
model_folder = os.path.join(root_folder, "models/PPMI_DaTSCAN", trained_model, net)


batch_size = 10 
lr = 0.5*0.0001
weight_decay = 0.1
gamma = 0.1
step_size = 7
epochs = 100

img_shape = (slices, rows, cols)
channels = net_dic[net]

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)


#%% Get Dataloaders for the current_fold

#%% Create Image Dataset and Get Dataloaders

# Data augmentation (on-the-fly) parameters
aug_prob = 0.5
rand_rot = 10                       # random rotation range [deg]
rand_rot_rad = rand_rot*math.pi/180 # random rotation range [rad]
rand_noise_std = 0.01               # std random Gaussian noise
rand_shift = 5                      # px random shift
min_zoom = 0.9
max_zoom = 1.1

# Define on-the-fly data augmentation transformation
data_transforms = {
    'train': Compose([
        Resize(spatial_size=img_shape),
        RandRotate(range_x=rand_rot_rad, range_y=rand_rot_rad, range_z=rand_rot_rad, prob=aug_prob),
        RandGaussianNoise(std=rand_noise_std, prob=aug_prob),
        Affine(translate_params=(rand_shift, rand_shift, rand_shift), image_only=True),
        RandZoom(min_zoom=min_zoom, max_zoom=max_zoom, prob=aug_prob),
        RepeatChannel(repeats=channels),]),
    'val': Compose([Resize(spatial_size = img_shape), RepeatChannel(repeats=channels),]),
    'test': Compose([Resize(spatial_size = img_shape), RepeatChannel(repeats=channels),]),
    }


# Load Data utilizzando la classe Parkinson_Dataset
train_dataset = BrainDataset(
    dataset_type = dataset_type,
    dic_classes = dic_classes,
    set_type='train', 
    transform = data_transforms,
    n_fold = n_fold, 
    current_fold = current_fold)

val_dataset = BrainDataset(
    dataset_type = dataset_type,
    dic_classes = dic_classes,
    set_type='val', 
    transform = data_transforms,
    n_fold = n_fold, 
    current_fold = current_fold)

test_dataset = BrainDataset(
    dataset_type = dataset_type,
    dic_classes = dic_classes,
    set_type='test', 
    transform = data_transforms,
    n_fold = n_fold, 
    current_fold = current_fold)


# dizionario che contiene le dimensioni dei set
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
dataloaders = {
    'train': DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=None),
    'val': DataLoader(dataset=val_dataset, batch_size=1, sampler=None),
    'test': DataLoader(dataset=test_dataset, batch_size=1, sampler=None)}

dataiter = iter(dataloaders['test'])
data = next(dataiter)
image, labels = data


net = load_trained_blackbox(net, channels, img_shape, out_shape, model_folder, current_fold)
net.eval()

# Inference
predictions, targets, report = eval_blackbox(
    dataloaders['test'],
    net, 
    dic_classes, 
    experiment_folder="",
    save=False)

print(report, flush = True)

cm = confusion_matrix(predictions, targets)
tp = cm[0][0]
fn = cm[0][1]
fp = cm[1][0]
tn = cm[1][1]

sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
acc = accuracy_score(predictions, targets)
bal_acc = balanced_accuracy_score(predictions, targets)
f1 = f1_score(predictions, targets)

print("Accuracy: ", acc, "\n",
      "Sensitivity: ", sensitivity, "\n",
      "Specificity: ", specificity, "\n",
      "Balanced Accuracy:", bal_acc, "\n",
      "f1-score", f1, flush = True) 

#%% LIME - Explain predictions

dataiter = iter(dataloaders['test'])
data = next(dataiter)
image, labels = data

# Immagine del quale vogliamo generare la spiegazione della predizione
X = image[0,0,:,:,:]

# Istanzio la classe che viene utilizzata per generare la spiegazione
explainer = lime_image3D.LimeImage3DExplainer()

# Classification function 
def predict_image(image):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    image = image[0]
    rep = RepeatChannel(repeats=channels)
    image = rep(image.unsqueeze(dim=0))
    image = image.unsqueeze(dim=0)
    image = image.to(device)
    net.eval()
    y_logit = net(image)
    y_pred_prob = F.softmax(y_logit, dim=-1) # predicted probability
    return y_pred_prob.detach().cpu().numpy()

            

# Genero la spiegazione alla predizione delle classi predette
# hide_color è il valore che viene sostituito ai superpixel posti a off, 
# num_samples sono il numero dei campioni nell'intorno di X da generare quando
# si va a perturbare tale istanza.
explanation = explainer.explain_instance(
    X, 
    classifier_fn = predict_image, 
    top_labels = 2, 
    hide_color = 0, 
    num_samples = 1000
    )

#%% Plot explanations

# Genero l'immagine e la maschera che costituiscono la spiegazione alla prima 
# delle 2 classi e visualizzo solo i super-voxel risultati avere un 
# contributo significativo e positivo ai fini della determinazione della 
# classe.
# num_features sono il numero dei superpixel da includere nella spiegazione.
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[labels.item()],
    positive_only = True,
    negative_only = False,
    num_features = 10, 
    hide_rest = True);
# Plot dell'explaination
plot_3d_slices(np.array(temp), cmap='jet', title='Top 10 positive contribution')
#plot_3d_slices(mask)


temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[labels.item()],
    positive_only = False,
    negative_only = True,
    num_features = 10, 
    hide_rest = True);
# Plot dell'explaination
plot_3d_slices(np.array(temp), cmap='jet', title='Top 10 negative contribution')
#plot_3d_slices(mask)



