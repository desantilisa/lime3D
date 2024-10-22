#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:18:26 2024

@author: lisadesanti
"""

import os
import sys
import xml.etree.ElementTree as ET
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
from lime.segmentation3D import segmentation3D_atlas
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
atlas_labels_path = "/home/lisadesanti/DeepLearning/PPMI/Atlases/HarvardOxford-Subcortical.xml"


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


#%% Create Image Dataset and Get Dataloaders for current fold

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


atlas = segmentation3D_atlas()
num_features = len(np.unique(atlas))
atlas_features = np.unique(atlas)

mean_subj_i_lime_ct_as_ct = []  # list containing the average LIME maps of every ct image predicted as ct
std_subj_i_lime_ct_as_ct = []   # list containing the variance LIME maps of every ct image predicted as ct
mean_subj_i_lime_ct_as_pd = []  # list containing the average LIME maps of every ct image predicted as pd
std_subj_i_lime_ct_as_pd = []   # list containing the variance LIME maps of every ct image predicted as pd
mean_subj_i_lime_pd_as_pd = []  # list containing the average LIME maps of every pd image predicted as pd
std_subj_i_lime_pd_as_pd = []   # list containing the variance LIME maps of every pd image predicted as pd
mean_subj_i_lime_pd_as_ct = []  # list containing the average LIME maps of every pd image predicted as ct
std_subj_i_lime_pd_as_ct = []   # list containing the variance LIME maps of every pd image predicted as ct

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_explanations = 100 # n° LIME explanations produced for every subject

with torch.no_grad():
    
    for i, (image, label) in enumerate(dataloaders['test']): 
        
        print("Subject ", i)
        image = image.to(device)
        net.eval()
        y_logit = net(image)
        y_pred_prob = F.softmax(y_logit, dim=-1) # predicted probability
        y_pred_class = torch.argmax(y_pred_prob, dim=-1) # predicted class label
        
        lime_subj_i = [] # list containing a number of num_explanations LIME 
                         # maps of image i

        for j in range(num_explanations):
            # Generate a number of "num_explanations" LIME maps for every 
            # image
        
            # Image to explain
            X = image[0,0,:,:,:]
        
            # Explainer
            explainer = lime_image3D.LimeImage3DExplainer(random_state=j)
        
            # Generate explanation of the predicted class
            #   hide_color: pixel intensity substituited to the superpixel 
            #   num_samples: #samples in the neighborhood of X
            explanation = explainer.explain_instance(X, classifier_fn = predict_image, top_labels = 2, hide_color = 0, num_samples = 100, random_seed=j, progress_bar=False)
            exp_prediction = explanation.local_exp[explanation.top_labels[0]] # LIME explanation of predicted class (top_labels[0])
            exp_prediction.sort(key=lambda x: x[0]) # sort the LIME features by atlas array value (ascending)
            lime_subj_i.append(np.array([features[1] for features in exp_prediction]))
            
            # # Plot explanations
            # # Generate explanation of top 10 image features with a positive  
            # # contribution for the predicted class
            # top10_pos, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only = True, negative_only = False, num_features = 10, hide_rest = True);
            # # Plot dell'explaination
            # plot_3d_slices(np.array(top10_pos), cmap='jet', title='Top 10 positive contribution')
            # # Generate explanation of top 10 image features with a negative  
            # # contribution for the predicted class
            # top10_neg, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only = False, negative_only = True, num_features = 10, hide_rest = True);
            # # Plot dell'explaination
            # plot_3d_slices(np.array(top10_neg), cmap='jet', title='Top 10 negative contribution')

        lime_subj_i = np.array(lime_subj_i) # row_j = j-th LIME maps of image i
                                            # column_k = brain region of value k in the atlas array
        
        if label.item() == 0 and y_pred_class.item() == 0:
            # LIME explanation of control subjects predicted as control
            mean_subj_i_lime_ct_as_ct.append(np.mean(lime_subj_i, axis = 0))
            std_subj_i_lime_ct_as_ct.append(np.std(lime_subj_i, axis = 0))
        
        elif label.item() == 0 and y_pred_class.item() == 1:
            # LIME explanation of control subjects predicted as Parkinson
            mean_subj_i_lime_ct_as_pd.append(np.mean(lime_subj_i, axis = 0))
            std_subj_i_lime_ct_as_pd.append(np.std(lime_subj_i, axis = 0))
            
        elif label.item() == 1 and y_pred_class.item() == 1:
            # LIME explanation of Parkinson subjects predicted as Parkinson
            mean_subj_i_lime_pd_as_pd.append(np.mean(lime_subj_i, axis = 0))
            std_subj_i_lime_pd_as_pd.append(np.std(lime_subj_i, axis = 0))

        elif label.item() == 1 and y_pred_class.item() == 0:
            # LIME explanation of Parkinson subjects predicted as control
            mean_subj_i_lime_pd_as_ct.append(np.mean(lime_subj_i, axis = 0))
            std_subj_i_lime_pd_as_ct.append(np.std(lime_subj_i, axis = 0))
            

# For every array:
#   row_i = i-th image
#   column_k = brain region of value k in the atlas array
mean_subj_i_lime_ct_as_ct = np.array(mean_subj_i_lime_ct_as_ct) 
std_subj_i_lime_ct_as_ct = np.array(std_subj_i_lime_ct_as_ct)
mean_subj_i_lime_ct_as_pd = np.array(mean_subj_i_lime_ct_as_pd)
std_subj_i_lime_ct_as_pd = np.array(std_subj_i_lime_ct_as_pd)
mean_subj_i_lime_pd_as_pd = np.array(mean_subj_i_lime_pd_as_pd)
std_subj_i_lime_pd_as_pd = np.array(std_subj_i_lime_pd_as_pd)
mean_subj_i_lime_pd_as_ct = np.array(mean_subj_i_lime_pd_as_ct)
std_subj_i_lime_pd_as_ct = np.array(std_subj_i_lime_pd_as_ct)

# Compute the Group-level Mean and Variance LIME importance maps
# (Von Neumann statistics)
mean_lime_ct_as_ct = np.mean(mean_subj_i_lime_ct_as_ct/(std_subj_i_lime_ct_as_ct**2), axis=0)/np.mean(1/(std_subj_i_lime_ct_as_ct**2), axis=0)
std_lime_ct_as_ct = 1/np.mean(1/(std_subj_i_lime_ct_as_ct**2), axis=0)
mean_lime_ct_as_pd = np.mean(mean_subj_i_lime_ct_as_pd/(std_subj_i_lime_ct_as_pd**2), axis=0)/np.mean(1/(std_subj_i_lime_ct_as_pd**2), axis=0)
std_lime_ct_as_pd = 1/np.mean(1/(std_subj_i_lime_ct_as_pd**2), axis=0)
mean_lime_pd_as_pd = np.mean(mean_subj_i_lime_pd_as_pd/(std_subj_i_lime_pd_as_pd**2), axis=0)/np.mean(1/(std_subj_i_lime_pd_as_pd**2), axis=0)
std_lime_pd_as_pd = 1/np.mean(1/(std_subj_i_lime_pd_as_pd**2), axis=0)
mean_lime_pd_as_ct = np.mean(mean_subj_i_lime_pd_as_ct/(std_subj_i_lime_pd_as_ct**2), axis=0)/np.mean(1/(std_subj_i_lime_pd_as_ct**2), axis=0)
std_lime_pd_as_ct = 1/np.mean(1/(std_subj_i_lime_pd_as_ct**2), axis=0)


# Select the regions with the highest positive and negative contribution for 
# group prediction
pos_roi_ct_as_ct = np.where(mean_lime_ct_as_ct > 0)[0] # brain region values in the atlas array with positive LIME features
neg_roi_ct_as_ct = np.where(mean_lime_ct_as_ct < 0)[0] # brain region values in the atlas array with negative LIME features
top_pos_roi_ct_as_ct = pos_roi_ct_as_ct[np.argsort(-mean_lime_ct_as_ct[pos_roi_ct_as_ct])][:10] # sort indexes of positive values (descending).
top_neg_roi_ct_as_ct = neg_roi_ct_as_ct[np.argsort( mean_lime_ct_as_ct[neg_roi_ct_as_ct])][:10] # sort indexes of negative values (ascending).

pos_roi_ct_as_pd = np.where(mean_lime_ct_as_pd > 0)[0]
neg_roi_ct_as_pd = np.where(mean_lime_ct_as_pd < 0)[0]
top_pos_roi_ct_as_pd = pos_roi_ct_as_pd[np.argsort(-mean_lime_ct_as_pd[pos_roi_ct_as_pd])][:10]
top_neg_roi_ct_as_pd = neg_roi_ct_as_pd[np.argsort( mean_lime_ct_as_pd[neg_roi_ct_as_pd])][:10]
        
pos_roi_pd_as_pd = np.where(mean_lime_pd_as_pd > 0)[0]
neg_roi_pd_as_pd = np.where(mean_lime_pd_as_pd < 0)[0]
top_pos_roi_pd_as_pd = pos_roi_pd_as_pd[np.argsort(-mean_lime_pd_as_pd[pos_roi_pd_as_pd])][:10]
top_neg_roi_pd_as_pd = neg_roi_pd_as_pd[np.argsort( mean_lime_pd_as_pd[neg_roi_pd_as_pd])][:10]

pos_roi_pd_as_ct = np.where(mean_lime_pd_as_ct > 0)[0]
neg_roi_pd_as_ct = np.where(mean_lime_pd_as_ct < 0)[0]
top_pos_roi_pd_as_ct = pos_roi_pd_as_ct[np.argsort(-mean_lime_pd_as_ct[pos_roi_pd_as_ct])][:10]
top_neg_roi_pd_as_ct = neg_roi_pd_as_ct[np.argsort( mean_lime_pd_as_ct[neg_roi_pd_as_ct])][:10]
    

# Statistically relevant brain regions
# \mu_{LIME}/\sigma_{LIME}
roi_ct_as_ct = mean_lime_ct_as_ct/np.sqrt(std_lime_ct_as_ct)
roi_ct_as_pd = mean_lime_ct_as_pd/np.sqrt(std_lime_ct_as_pd)
roi_pd_as_pd = mean_lime_pd_as_pd/np.sqrt(std_lime_pd_as_pd)
roi_pd_as_ct = mean_lime_pd_as_ct/np.sqrt(std_lime_pd_as_ct)    


# Load XML files
tree = ET.parse(atlas_labels_path)
root = tree.getroot() # get the xml root
array_label_dic = {} # key: brain region value in the numpy array atlas 
                     # value: brain region name
# Itera su tutti gli elementi 'item' nel file XML
for item in root.findall('data'):
    for id_label, label in enumerate(item.findall('label')):
        brain_label = label.text
        x = int(label.attrib['x'])
        y = int(label.attrib['y'])
        z = int(label.attrib['z'])
        atlas_value = atlas[z,y,x]
        array_label_dic[atlas_value] = brain_label

# Add the label of the background region
array_label_dic[0.] = 'Background'




 
