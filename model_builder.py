#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:19:24 2023

@author: lisadesanti
"""

import sys
from copy import deepcopy
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn

from utils import get_model_layers
from utils import set_device
from resnet_features import resnet_18_3d
from convnext_features import convnext_tiny_3d


def load_blackbox(net, channels, img_shape, out_shape):
    
    """ Instantiate the desidered model """
    
    if net == "resnet3D_18_kin400":
        # ResNet3D-18 pretrained on Kinetics400, 3-channel 3D input
        weights = models.video.R3D_18_Weights.DEFAULT # best available weights 
        model = models.video.r3d_18(weights=weights)
        # Recreate the classifier layer and seed it to the target device
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), 
            torch.nn.Linear(in_features=512, out_features=out_shape, bias=True))

    elif net == "resnet3D_18":
        # ResNet3D-18 pretrained on Medical Images, 1-channel 3D input
        model = resnet_18_3d(
            pretrained=True, 
            sample_input_D = img_shape[0],
            sample_input_H = img_shape[1],
            sample_input_W = img_shape[2],
            num_classes = out_shape)
                
    elif net == "convnext3D_tiny_imgnet":
        # ConvNeXt3D pretrained on ImageNet + weight inflation, 3-channel 3D input
        pretrained_path = "/home/lisadesanti/DeepLearning/Lisa/pretrained_backbones/convnext3d_pretrained/saved_models/"
        pretrained_mode = "imagenet"
        model = convnext_tiny_3d(
                    pretrained = True,
                    added_dim = 2,
                    init_mode = 'two_g',
                    ten_net = 0,
                    in_chan=channels,
                    use_transformer = False,
                    pretrained_path = pretrained_path,
                    pretrained_mode = pretrained_mode,
                    drop_path = 0.0,
                    datasize = 256 # Shape (HxWxD) of the data is 256x256x256
                )
        
    elif net == "convnext3D_tiny":
        # ConvNeXt3D pretrained on Medical Images (STOIC dataset), 1-channel 3D input
        pretrained_path = "/home/lisadesanti/DeepLearning/Lisa/pretrained_backbones/convnext3d_pretrained/saved_models/"
        pretrained_mode = "multitaskECCV"
        model = convnext_tiny_3d(
                    pretrained = True,
                    added_dim = 2,
                    init_mode = 'two_g',
                    ten_net = 0,
                    in_chan = channels,
                    use_transformer = False,
                    pretrained_path = pretrained_path,
                    pretrained_mode = pretrained_mode,
                    drop_path = 0.0,
                    datasize = 256 # Shape (HxWxD) of the data is 256x256x256
                )
        
    return model


def load_trained_blackbox(net, channels, img_shape, out_shape, model_folder, current_fold):
    
    """ NB: You need to copy the weights of trained model you want to 
    instantiate into the specified models_folder before! """
    

    models_folder = model_folder
    model_path = models_folder + "/fold" + str(current_fold) + ".pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_blackbox(net, channels, img_shape, out_shape)
    net.load_state_dict(torch.load(model_path)) 
    net.to(device)
    net.eval()
    
    return net


    
    