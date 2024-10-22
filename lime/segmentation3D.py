# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 15:42:24 2021

@author: User
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from skimage import measure
import SimpleITK as sitk
import monai.transforms as transforms

"""

Attraverso algoritmi di segmentazione l'immagine viene divisa in super-pixel 
(nel nostro caso super-voxel), i quali verranno sfruttati per passare ad una
rappresentazione dell'immagine in un "dominio interpretabile" e 
successivamente verranno posti a on (l'intensità dei voxel appartenenti al 
super-voxel viene lasciata inalterata) o off (i voxel appartenenti al 
super-voxel vengono tutti settati ad un valore di default es. 0) in modo e in 
numero casuale.

Gli autori che hanno sviluppato il package lime hanno fornito la possibilità 
di sfruttare le seguenti tecniche di segmentazione (dal package 
skimage.segmentation):
    
    - skimage.segmentation.felzenszwalb: Computes Felsenszwalb’s efficient 
        graph based image segmentation (solo x immagini 2D);
    - skimage.segmentation.slic: Segments image using k-means clustering in 
        Color-(x,y,z) space (sarebbe stata ok anche per immagine 3D a 1 
        canale);
    - skimage.segmentation.quickshift: Segments image using quickshift 
        clustering in Color-(x,y) space.
        
Il numero dei super-voxel nell'immagine costituisce il numero delle features 
nella rappresentazione interpretabile dell'immagine.

"""
def segmentation3D_kmeans(X):
    
    """
    Segmentazione dell'immagine X (ndarray 3D, imamgine volumetrica a 1 
    canale) ho implemetato l'algoritmo di clustering kmeans seguito da un 
    algoritmo di labeling per separare i cluster topologicamente disgiunti
    
    NB: Ancora non è stato determinato quale sia il numero di 
    super-pixel(voxel) nel quale suddividere l'immagine, teoricamente 
    l'algoritmo potrebbe funzionare anche facendo coincidere il super-voxel 
    con il singolo voxel (e in effetti quando lime viene applicato ad esempio
    a testi le parole che lo compongono non vengono aggregate come viene fatto
    coi voxel dell'immagine), viene scelto di procedere in questo modo con le 
    immagini in quanto perturbando l'immagine facendo variare il singolo voxel
    potrebbe portare a non avere variazioni significative nella predizione 
    restituita dal modello black-box, causando potenziali problemi nella 
    determinazione dell'importanza delle features negli step successivi 
    dell'algoritmo.
    """
    
    size = X.shape
    Nz = size[0]
    Ny = size[1]
    Nx = size[2]
    X_vect = X.reshape(Nx*Ny*Nz,1);
    kmeans = KMeans(random_state=0,n_clusters=20).fit(X_vect);
    labels = kmeans.labels_; # immagine segmentata vettorizzata 
    segmented = labels.reshape(Nz,Ny,Nx); # immagine segmentata array vol
    segmented = measure.label(segmented, connectivity=3) # componenti connesse
    return segmented


def segmentation3D_atlas(atlas_type = "sub"):
    
    if atlas_type == "cort":
        atlas_cort_path = "/home/lisadesanti/DeepLearning/PPMI/Atlases/HarvardOxford-cort-maxprob-thr25-2mm.nii"
        atlas_cort_image = sitk.ReadImage(atlas_cort_path)
        atlas_cort = sitk.GetArrayFromImage(atlas_cort_image)
        atlas = atlas_cort.copy()
    
    elif atlas_type == "sub":
        atlas_sub_path = "/home/lisadesanti/DeepLearning/PPMI/Atlases/HarvardOxford-sub-maxprob-thr25-2mm.nii"
        atlas_sub_image = sitk.ReadImage(atlas_sub_path)
        atlas_sub = sitk.GetArrayFromImage(atlas_sub_image)
        atlas = atlas_sub.copy()

    atlas = atlas.astype(np.int32)
    
    return atlas