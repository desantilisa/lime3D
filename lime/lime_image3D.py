# -*- coding: utf-8 -*-
"""
Created on Tue Aug 3 09:52:40 2021

@author: User
"""

import copy
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.utils import check_random_state
from tqdm.auto import tqdm
from lime import lime_base, lime_image
from lime.lime_image import LimeImageExplainer, ImageExplanation
from lime.segmentation3D import segmentation3D_atlas
from skimage.segmentation import mark_boundaries


class Image3DExplanation(ImageExplanation):
    
    """
    ImageExplaination extension
    Class storing the LIME explanation of the predicted output of current 
    instance
    
    ImageExplaination
    Attributes:
        - image: Input for which we want to explain the model's prediction 
        - segments: Input image segmented into superpixels
        - intercept: Intercept of local linear model
        - local_exp: List of tuples. 
                     Every tuple stores the feature id with its weight in the 
                     linear model
        - local_pred: Prediction to explain
        - score: R2 coeff of the linear model
    Methods:
    - get_image_and_mask: Metodo utilizzato per ottenere la spiegazione alla 
        label passata.
    
    """
    
    def __init__(self, image, segments):
        ImageExplanation.__init__(self, image, segments)

    def get_image_and_mask(self, label, positive_only=True, negative_only=False, hide_rest=False, num_features=5, min_weight=0.):
        """
        Method overriding
        Returns:
            temp: Heatmaps image of importance supervoxels
            mask: Mask of important supervoxels
        """

        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
            
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
            
        segments = self.segments
        image = self.image
        exp = self.local_exp[label] # list [(feat_1,w_1),...,(feat_n,w_n)] of selected label
        mask = np.zeros(segments.shape, segments.dtype)
        
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.clone()
            
        if positive_only:
            print("Select positive only image features")
            fs = [x[0] for x in exp if x[1] > 0 and x[1] > min_weight][:num_features]
            
        if negative_only:
            print("Select negative only image features")
            fs = [x[0] for x in exp if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
            
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].detach().cpu().clone()
                mask[segments == f] = 1
            return temp, mask
        
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = -1 if w < 0 else 1
                temp[segments == f] = image[segments == f].detach().cpu().clone()
                #temp[segments == f, c] = np.max(image)
            return temp, mask

class LimeImage3DExplainer(LimeImageExplainer):
    
    """
    LimeImageExplainer extension
    Generates explanations to the predictions returned by the black-box model
    The explanation of the prediction returned by the model for an input images
    is an heatmap highlighting the voxels with the highest contribution in 
    prediciton
    
    --------------------------------------------------------------------------
    --- Class LimeImageExplainer ---
    Classe utilizzata per generare la spiegazione alle predizioni delle 
    immagini.
    Numerical features are perturbed sampling from a normal distribution and 
    performing the inverse operation of centering to the mean and scaling based 
    on the mean and standard deviation of the training data
    
    Attributies
        - random_state
        - feature_selection
        - base: LimeBase class instance
            ------------------------------------------------------------------
            --- Classe LimeBase ---
            Class which learns a sparse local linear model from perturbed data
            in their interpretable representation
            Applies a features selection and fits a local linear model 
            Class used for every input data (text, tabular, images)
            
            Attributes:
                - kernel_fn: Function whichtakes a distance array and returns 
                    a proximity array, \pi_x
                    Default:
                        \sqrt{e^{-\frac{d^2}{{kernel_dim}^2}}}
                - verbose: Print values of the local linear model
                - random_state: Seed used to generate randoom number
            Methods:
            	- generate_lars_path
                    K-features selection criteria 
                    Generates the LASSO path of the proximity-weighted data to 
                    select K features of the linear model
            	- forward_selection
                    Modality selection of the K features
                    Iteratively add features to the linear model
            	- feature_selection
                    Modality selection of the K features
                    Iteratively add features to the linear model
            	- explain_instance_with_data
            ------------------------------------------------------------------     
    
    Mathods:
    	- explain_instance: 
            Generates explanation to a model's prediction stored into an 
            ImageExplaination instance
            For single-channel images, generates an RGB 3-channel version 
            Applies a segmentation algorithm to identify image supervoxels
            Generates the perturbation images (used to turn off supervoxels)
            Calls "data_labels" method to sample neighbourhood images and their 
            respective labels
            Computes distances for the neighbours.
            Instantiates the ImageExplanation class and calls the 
            explain_instance_with_data method from LimeBase

        - data_labels
            Generates images and predictions in the neightborhood of the input
            image
            Parameters:
                fudged_image: Image to replace the original image when a 
                superpixel is turned "off"
                n_features: Number of superpixels identified by the 
                segmentation algorithm
                data: n_samples*n_features matrix of dimensions of 0s and 1s 
                randomly generated
                rows: Same as data
                    row: Vector of length equal to the number of superpixels.
                    row = row[index1], row[index2], ..., row[indexn_features]
                zeros: A vector containing the indices of the row vector where 
                the elements are 0
                mask: A boolean mask which identifies the superpixels that will 
                be turned off
                temp: Original image in which the superpixels identified by the 
                mask are turned off
                imgs: Images list with the superpixels turned off

    --------------------------------------------------------------------------
    
    The LIME explanation of the 3D image is computed as follows:
        - Segment image into super-voxels (using segmentation3D) identified 
            with a label.
        - Generate perturbed images in the neightborhood of the instance to 
            explain using an interpretable representation. 
            The interpretable representation is a binary vector indicating the 
            presence of the super-voxel in the image (super-voxel label = index 
            of the vector element in its interpretable representation)
            Note: The interpretable representation of the original image is a 
            vector of all 1s.
        - Evaluation of the output of the black-box model for the perturbed 
            images
        - Compute the distance between the perturbed and the original images
        - Features selection with the k-lasso
        - Fits the linear model on the selected features
        
    """
    
    def __init__(self, kernel_width=.25, kernel=None, verbose=False, feature_selection='auto', random_state=None):
        
        LimeImageExplainer.__init__(self, kernel_width=kernel_width, kernel=kernel, verbose=verbose, feature_selection=feature_selection, random_state=random_state)
    
    # Override del metodo ereditato dalla classe padre (LimeImageExplainer)
    def explain_instance(self, image, classifier_fn, labels=(1,), hide_color=None, top_labels=5, num_features=100000, num_samples=1000, batch_size=1, segmentation_fn=None, distance_metric='cosine', model_regressor=None, random_seed=None, progress_bar=True):

        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000) # serviva per segmentare dall'immagine 2D

        if segmentation_fn is None:
            segmentation_fn = segmentation3D_atlas
            
        segments = segmentation_fn()  # segment image into supervoxels
        fudged_image = image.clone()  # image to replace the turned-off supervoxels
        if hide_color is None:
            for x in np.unique(segments):
                mask3D = segments == x;
                fudged_image[mask3D] = np.mean(image[mask3D])
        else:
            fudged_image[:] = hide_color

        top = labels

        """
        data_labels
            - Generates images and predictions in the neighborhood of the 
              passed image, and returns these samples in their interpretable 
              representation
              The number of features equals the number of superpixels
            - Generates a two-dimensional array, data, with dimensions 
              num_samples * num_features of random numbers from 0 to 1. 
              The i-th row represents the i-th sample, and the element (i, j) 
              represents the j-th superpixel value of the i-th perturbed
              sample. For 0 element, the corresponding superpixel will be off; 
              if it is 1, it will be on. 
              The first row corresponds to the original image (all 1s).
            - Generates various images with specific superpixels set to off 
              and produces the predicted label for each image using the 
              classifier.
            - The function returns the matrix of 0s and 1s (data) and the 
              labels corresponding to the perturbed samples (labels)
        """
        data, labels = self.data_labels(image, fudged_image, segments, classifier_fn, num_samples, batch_size);
        
        """
        pairwaise_distance  
        Computes distance metrics between original and perturbed images
        """
        distances = sklearn.metrics.pairwise_distances(data, data[0].reshape(1,-1), metric=distance_metric).ravel()

        ret_exp = Image3DExplanation(image, segments);
        
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        
        """
        For each label in the list for which we want to provide an explanation, 
        we call the explain_instance_with_data method with these arguments: 
        the matrix of 0s and 1s (interpretable form of the original and 
        perturbed images), labels for the images, the distance vector, the 
        label to explain, the number of features, the regression model, and 
        the feature selection method.

        explain_instance_with_data
        (Method of the LimeBase class)
        
        Arguments:
        
            Interpretable data: 2D array (n_samples * n_features), where each 
            row represents a perturbed instance with superpixels as 0 (off) or 
            1 (on). The first row is all 1s (original instance).
            Labels to explain
            Distances of perturbed data
            Maximum number of features in the explanation
            Feature selection method:
                forward_selection: Adds features iteratively
                highest_weights: Chooses features with the highest absolute 
                weight * original data points
                lasso_path: Selects features based on the LASSO path
                none: Uses all features
                auto
            Regression model (default: Ridge) to derive the linear model
            
        Returns:
            Model parameters
            R² coefficient
            Prediction for the original instance
            Weights are computed by applying the kernel to the distance vector.
            After selecting features, we fit the linear regression model 
            (default: Ridge from sklearn) using the fit method. 
            We then return the R² coefficient and the prediction.

        """
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                 data, labels, distances, label, num_features,
                 feature_selection=self.feature_selection,
                 model_regressor=model_regressor)
        return ret_exp
    

def plot_explaination(data, mask, num_columns=10):
        
        """
        Plot a montage of slices
        
        Funzione utilizzata per plottare tutte le slices (fusion di mask e 
        data) che costituiscono la spiegazione della classe dell'immagine 3D
        predetta dalla CNN.
        
        Il numero di colonne può essere scelto dall'utente, il numero di righe
        viene determinato di conseguenza, andando ad aggiungere immagine 
        fittizie di tutte 0 se necessario.
        """
        data = np.array(data)
        depth = data.shape[0]
        width = data.shape[1]
        height = data.shape[2]
        
        data_min = data.min()
        data_max = data.max()
        
        r, num_rows = math.modf(depth/num_columns)
        num_rows = int(num_rows)
        if num_rows == 0:
            num_columns = int(r*num_columns)
            num_rows +=1
            r = 0
        elif r > 0:
            new_im = int(num_columns-(depth-num_columns*num_rows))
            add = np.zeros((new_im, width, height), dtype=type(data[0,0,0]))
            data = np.concatenate((data, add), axis=0)
            mask = np.concatenate((mask, add), axis=0)
            num_rows +=1
        
        data = np.reshape(data, (num_rows, num_columns, width, height))
        mask = np.reshape(mask, (num_rows, num_columns, width, height))

        rows_data, columns_data = data.shape[0], data.shape[1]
        heights = [slc[0].shape[0] for slc in data]
        widths = [slc.shape[1] for slc in data[0]]
        fig_width = 12.0
        fig_height = fig_width * sum(heights) / sum(widths)
        
        f, axarr = plt.subplots(rows_data,columns_data,figsize=(fig_width, fig_height),gridspec_kw={"height_ratios": heights},);
            
        for i in range(rows_data):
            for j in range(columns_data):
                if rows_data > 1:
                    img = axarr[i, j].imshow(mark_boundaries(np.squeeze(data[i][j]), np.squeeze(mask[i][j])), cmap="gray", vmin=data_min, vmax=data_max)
                    axarr[i, j].axis("off");
                else:
                    img = axarr[j].imshow(mark_boundaries(np.squeeze(data[i][j]), np.squeeze(mask[i][j])), cmap="gray", vmin=data_min, vmax=data_max)
                    axarr[j].axis("off");

        plt.subplots_adjust(wspace=0, hspace=0, left=0, right = 0.9, bottom=0, top=0.9)
        plt.show()