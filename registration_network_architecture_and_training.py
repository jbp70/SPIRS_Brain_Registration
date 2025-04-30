#Final 128 sized model with three different optimizers
#Using 0-1 normalization since it leads to better MSE due to the strong edge boundaries (with unsigned NCC)
#Reduce LR/WD by factor of 100
#Increase translation by factor of two to account for doubling of size
#num_patches_per_patient set to 1 for 64/128 models
#Freeze batch norm layers due to reduced batch size
#Only doing one run of segmentation (since segmenting lower quality scans will not help appreciably, especially for the task of segmenting micro-metastases) 
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import time
import shutil
import numpy as np
import neurite as ne
import nibabel as nib
import tensorflow as tf
import hiddenlayer as hl
import matplotlib.pyplot as plt
import hiddenlayer.transforms as ht
import tensorflow_addons.layers as tfa_layers
import tensorflow_addons.optimizers as tfa_optimizers

from copy import deepcopy
from train_cluster import *
from predict_cluster import *
from itertools import product
from sorcery import unpack_keys
from load_data import DataGenerator, nested_folder_filepaths

train_32 = False
train_64 = False
train_128 = True

params_dict = {}
#project directory
params_dict['scratch_dir'] = str(os.environ['SLURM_JOB_SCRATCHDIR']) + '/'
params_dict['job_id'] = str(os.environ['SLURM_JOB_ID'])
#train/val/test directories
params_dict['data_dir_train'] = params_dict['scratch_dir'] + 'set1_nii/Train/'
params_dict['data_dir_val'] = params_dict['scratch_dir'] + 'set1_nii/Val/'
params_dict['data_dir_test'] = params_dict['scratch_dir'] + 'set1_nii/Test/'
params_dict['data_dirs_predict'] = [params_dict['data_dir_train'], params_dict['data_dir_val'], params_dict['data_dir_test']]
#model file names
params_dict['model_outputs_dir'] = params_dict['scratch_dir'] + 'model_outputs/'                            #directory where all outputs of the model will be saved (i.e. saved model weights, optimizer, text file, etc.)
params_dict['model_weights_save_path'] = params_dict['model_outputs_dir'] + 'tf_ckpts'                      #model weights name used when saving/loading a model to/from memory
params_dict['max_number_checkpoints_keep'] = 25                                                             #number of most recent models to save in tensorflow checkpoint format (models saved based off of validation monitor value)
params_dict['output_file'] = params_dict['model_outputs_dir'] + params_dict['job_id'] + '.txt'              #name of text file containing epoch loss/metrics
params_dict['tensorboard_dir'] = params_dict['model_outputs_dir'] + params_dict['job_id']                   #name of logs directory for tensorboard
params_dict['model_image_save_path'] = params_dict['model_outputs_dir'] + 'model_image'                     #name of model image that will be saved out
#train params
params_dict['sample_background_prob'] = 0.99                            #probability of sampling background class during patching
params_dict['learning_rate'] = [0.01 / 100, 0.1 / 100, 0.1 / 100]                         #initial learning rate 
params_dict['weight_decay'] = [0.0000375 / 100, 0.00002 / 100, 0.00002 / 100]             #initial weight decay (only used for optimizers that accept weight decay parameter -> AdamW and SGDW)
params_dict['num_patches_per_patient'] = [1, 1]                         #number of patches to extract from each patient in the training set and validation set
params_dict['adaptive_full_image_patching'] = [False, False]            #whether to train on full size images, and whether to validate on full size images (in order to get accurate validation set dice score)
params_dict['run_augmentations_every_epoch'] = [True, 1]                #whether to run augmentations every epoch; if False, how many epochs before refreshing data cache; only possible if "cache_volumes" option is true
params_dict['probability_augmentation'] = [0.5,0.5,0.0,0.9,0.9,0.9,0.9]                                   #probability of applying each of the requested transforms (if supplying list of probabilities, must be of length 7 (one probability for flip, gamma, deform, scale, rotate, shear, translate))
params_dict['augmentation_scheduler'] = [False, 25]                                         #whether to ramp up augmentation probability over many epochs; number of epochs to reach full augmentation probability
params_dict['left_right_flip'] = [True, (0,1,2)]                                            #whether to apply left/right patch flipping augmentation, and axes along which to perform flip
params_dict['gamma_correction'] = [True, [.75, 1.25]]                                       #whether to apply gamma correction to input channels, and range for gamma factor
params_dict['deform_mesh'] = [True, [10.0, 15], 1.25, 4, 16]                                #whether to apply deformation augmentation, scale factors for normal/abnormal, sigma of gaussian smoother, truncate filter sigma, spline order for interpolation of volume and label 
params_dict['scale_mesh'] = [True, [.9, 1.1], False]                                        #whether to apply scaling augmentation, and the bounds for scaling operation (less than 1 zooms in; greater than 1 zooms out); whether scaling should be isotropic or not
params_dict['rotate_mesh'] = [True, np.array([-20, 20]) * np.pi / 180]                      #whether to apply rotation augmentation, and the bounds for rotating operation (in radians)
params_dict['shear_mesh'] = [True, [-.05, .05], False]                                      #whether to apply shear augmentation, and the bounds for shear operation (values close to zero will produce sensible shears); whether to shear only along one direction, or to allow shearing along all valid axes
params_dict['translate_mesh'] = [True, [-20, 20]]                                           #whether to apply translation augmentation, and the bounds for translating operation (in voxels)
params_dict['erode_dilate_conn_comp'] = [[False, .10, 3], [False, .10, 3], [False, .05]]    #only applicable if passing in binary mask as input; whether to use erosion augmentation, probability of erosion, and max number of iterations; whether to use dilation augmentation, probability of dilation, and max number of iterations; whether to delete connected components at random, and probability of deletion
params_dict['randomize_order_augmentations'] = False                                        #whether to randomize the order of composable affine transforms (scale, rotate, shear) or to use the default order every time
params_dict['spline_order'] = [3, 0, 1]                                                     #spline order for applying deformations to volumes and label maps (and distance maps if applicable)
params_dict['rescale_data_range'] = [True, 0, 1, [True, 5], [True, 5]]                      #whether to rescale the input data range; the lower and upper bounds of the rescaling range; whether to apply bottom and top clipping and the standard deviation clips
params_dict['num_epochs'] = 5000                                                            #number of epochs to train the model
params_dict['verbose'] = 1                                      #0 shows no output, 1 shows training progress bar (not recommended for non-interactive jobs), 2 shows one line per epoch (also not recommended since have callbacks enabled)
params_dict['workers'] = 10                                     #number of parallel processes to generate to handle data loading
params_dict['max_queue_size'] = 10                              #maximum number of entries in the generator queue
params_dict['cache_volumes'] = True                             #whether to cache the volumes/labels into a dictionary (recommended if have enough memory to store entire dataset in memory)
params_dict['enable_xla'] = True                                #whether to enable xla compilation (will see performance boost, but compilation will fail unless carefully construct network to utilize only multiples of 2 and 8 everywhere)
params_dict['mixed_precision'] = True                           #whether to use mixed_precision
params_dict['custom_fit'] = [True, False, [False, 2.], False]   #whether to override model.fit; whether to apply weight decay to all variables (or just apply it to kernels, leaving bias/scale un-regularized); whether to increase learning rate of bias/scale terms to speed up convergence of those terms; whether to use gradient centralization
#optional custom learning rate schedule parameters
params_dict['lr_schedule_type'] = 'CosineAnneal'                        #learning rate schedule to use (choose from 'None' (which will default to reducing learning rate on plateau), 'StepDecay', 'CosineAnneal', 'Exponential')
params_dict['use_lr_warmup'] = [True, 5, 1000]                          #whether to use learning rate warmup, how many epochs to warmup for, factor smaller than actual learning rate to start warmup period at
params_dict['cycle'] = True                                             #whether to make schedule cyclic
params_dict['num_epochs_cycle'] = 250                                   #number of epochs in the first cycle
params_dict['factor_decrease_learning_rate'] = 250                      #amount to decrease learning rate by the end of the cycle
params_dict['cycle_multiplier'] = 1.5                                   #how much to increase cycle length with each new cycle
params_dict['factor_decrease_max_rate_per_cycle'] = 10                  #how much to decay max learning rate at start of new cycle (closer to 1 decays less)
params_dict['factor_decrease_min_rate_per_cycle'] = 15                  #how much to decay min learning rate at start of new cycle (closer to 1 decays less)
#unet params
params_dict['network_name'] = 'LearnAffineMatrixNetwork'                                                     #name of network to use during training (choose between 'Unet' and 'ClassificationNetwork'); 'ClassificationNetwork' can be used as a generic ResNet/DenseNet by modulating the options below
params_dict['optimizer1'] = ['SGDW', params_dict['weight_decay'][0], params_dict['learning_rate'][0], .9]    #optimizer along with parameters (i.e. SGDW with weight decay, learning rate, and momentum)
params_dict['optimizer2'] = ['SGDW', params_dict['weight_decay'][1], params_dict['learning_rate'][1], .9]    #optimizer along with parameters (i.e. SGDW with weight decay, learning rate, and momentum)
params_dict['optimizer3'] = ['SGDW', params_dict['weight_decay'][2], params_dict['learning_rate'][2], .9]    #optimizer along with parameters (i.e. SGDW with weight decay, learning rate, and momentum)
params_dict['normalization'] = ['BatchNormalization', False]                                                 #normalization (choose from 'BatchNormalization', 'LayerNormalization', 'GroupNormalization', 'InstanceNormalization'); if using BatchNormalization, specify whether want to train batch statistics; if using GroupNormalization, specify number of channels per group and a minimum number of groups (both must be able to evenly divide filter_num_per_level parameter)
############################################################################################################
#extra useless parameters that can be ignored
params_dict['use_conn_comp_patching'] = True                            #whether to patch from set of connected components (only set to True if have binary task with many distinct lesions)
params_dict['patch_inside_volume'] = True                               #whether to patch inside the volume region for normal patches (which should help stabalize training by using patches with limited added padding in it)
params_dict['patch_center_of_tumor_for_classification'] = False         #whether to create a patch using the center of the tumor segmentation (only usable if tumor segmentation given as input to the network)
params_dict['require_positive_class_per_batch'] = False                 #whether all batches are required to have an example of the positive class present (will help stabilize training especially when training at small batch sizes)
params_dict['balanced_batches_for_classification'] = False              #whether to balance classes such that each batch has approximiately the same percentage of positive to negative examples
params_dict['balanced_epochs_for_classification'] = [False, 20]         #whether to balance classes such that each epoch has the same number of total examples, but has n number of negatives randomly swapped out for duplicated positives
params_dict['voxel_spacing'] = [True, np.array([1,1,1])]                #whether your data has isotropic voxel spacing (if it does, you can ignore the second option; otherwise put voxel spacing to ensure augmentations are performed correctly)
params_dict['downsample_for_augmentation'] = [False, 2]                 #whether to downsample the meshgrid for faster augmentation (helpful if augmentation is bottle-necking training script), and if so, the factor by which to downsample the grid
params_dict['run_augmentations_in_2D'] = False                          #whether to augment 2D slices in 3D volume instead of using native 3D augmentations (if using highly anisotropic data, augmentations can take lots of time to run since need to resample data (and can be subotimal due to the anisotropic nature of the data))
params_dict['crop_input_image_for_augmentation'] = [False, 10, False]   #whether to crop the input images when augmenting the images (will speed up augmentation, but may cause artifacts in patches); amount of voxels to keep around patch to limit artifacts; whether to account for voxel spacing and use the downsample_for_augmentation parameter (suggested to be False)
params_dict['early_stopping_threshold'] = [1000, 30, 5.]                #number of epochs before breaking from training loop if no improvement in validation monitor, number of epochs to drop learning rate (if not using LR schedule), factor by which to drop learning rate
#params for optional step decay learning rate decay schedule
params_dict['step_epoch_nums'] = [30, 50, 70, 85]                       #epochs at which learning rate decay will occur (make sure to take into account learning rate warmup if using that)
params_dict['step_factors'] = [1, 5, 25, 125, 625]                      #factor by how much initial learning rate should be decayed at corresponding steps (must be list of length one greater than length of 'step_epoch_nums')
#params for optional exponential learning rate decay schedule
params_dict['save_at_cycle_end'] = False                                #whether to save the model at the end of the cycle
params_dict['power'] = 0.15                                             #power to reduce rate (closer to 1 makes function more linear; closer to 0 makes function more like a step function)
#predict params 
params_dict['predict_using_patches'] = False                                            #whether to predict patchwise or pass entire volume into convolutional net and predict in a single pass
params_dict['patch_overlap'] = 0.75                                                     #percentage of overlap with neighboring patches (i.e. .75 means 75% overlap)
params_dict['percentage_empty_to_skip_patch'] = [0.5, True]                             #patches at the borders of the image will be mostly empty and are not worth predicting on. This parameter skips patches if they are below the threshold selected (i.e. 0.5 means that patches that are 50% empty will be skipped); whether you want to predict a patch if the center of the patch contains data (regardless of what percentage of the patch is actually empty)
params_dict['boundary_removal'] = 3                                                     #number of voxels to remove around the edge of the predicted patch to combat edge effects
params_dict['blend_windows_mode'] = ['constant']                                        #how to blend sliding window patches together (choose between 'constant', 'linear', or 'gaussian'). 'constant' gives equal weight to all predictions; 'linear' gives full weight to middle of patch and linearly decreases weight of samples close to edge; 'gaussian' gives less weight to predictions on edges of windows. If using 'gaussian', provide sigma(s) to determine kernel weights along spatial dimensions (recommended setting between 0.15 and 0.35)
params_dict['average_logits'] = True                                                    #whether raw logits should be averaged together during patch-based inference and sigmoid probability taken at very end
params_dict['logits_clip_value'] = 50                                                   #what value above/below to clip logits to prevent overflow errors in sigmoid function
params_dict['predict_left_right_patch'] = [True, params_dict['left_right_flip'][1]]     #whether to predict on both the left and right flipped patch and then average results, and axis along which to perform flip
params_dict['predict_multi_scale'] = [False, params_dict['scale_mesh'][1], 3]           #whether to predict on scaled inputs (not implemented for patchwise inference); what scales to compute predictions over; what order interpolation to use (anything other than order 0 or order 1 will take a very long time to run)
params_dict['binarize_value'] = [0.5]                                                   #probability at which to threshold sigmoid function outputs at   
params_dict['number_snapshots_ensemble'] = [0]                                          #which snapshots to ensemble together (must be list with no more than "max_number_checkpoints_keep" values)
params_dict['predict_fast_to_find_roi'] = [False, .1, [7,7,2]]                          #whether to first run inference using small patch_overlap and then predict again using the true patch overlap only on the region of interest that has been detected; how many slices above/below ROI to add in case small patch_overlap is incorrect
params_dict['save_uncertainty_map'] = [False, 'uncertainty-mask.nii.gz']                #whether to create uncertainty mask from different predictions; name of uncertainty mask volume
#unet params
params_dict['block_type'] = 'Convolutional_Block'                                       #block type to use in unet (choose between 'Convolutional_Block', 'Dense_Block', 'Residual_Block', 'Residual_Bottleneck_Block')
params_dict['preactivation'] = [True, 32, 7]                                            #whether to use pre-activated blocks (Norm->Act->Conv) instead of post-activated blocks (Conv->Norm->Act); maximum number of filters to use in first filter; kernel size for first convolution
params_dict['levels'] = 4                                                               #number of levels in u-net
params_dict['atrous_conv'] = [0, np.array([2,2])]                                       #number of levels (from the bottom) that use atrous convolutions with no pooling or skip connections (max=levels-1), and size of initial dilation
params_dict['filter_num_per_level'] = [32,64,128,256]                                   #number of filters in each layer of the network (must be list of length equal to levels parameter and each successive value must be double the previous or you might run into compilation errors on some of the more complex unets)
params_dict['blocks_per_level'] = [1,2,2,2]                                             #number of blocks (i.e. conv->norm->act) per level; if an integer, uses that number of blocks on each level throughout the network, else input a list of length 2*levels-1 with number of blocks on each level (useful if want to create asymmetrical encoder-decoder)
params_dict['dense_growth_rate_per_level'] = [16,16,32,32]                              #growth rate if using dense block; if an integer, uses that same growth rate on each level throughout the network, else input a list of length 2*levels-1 with growth rate of each level
params_dict['dense_bottleneck'] = [True, 4, False]                                      #whether to use bottleneck layers in dense block; how much larger than the growth rate the bottleneck size should be; whether to only use bottleneck layers if the current number of filters is more than or equal to the size of the bottleneck layer
params_dict['deep_supervision'] = [True, 3]                                             #whether to use deep supervision, and the number of intermediary levels to use it with
params_dict['num_inputs'] = 3                                                           #number of inputs to network
params_dict['num_outputs'] = 15                                                         #number of output channels to predict
params_dict['activation'] = ['ReLU']                                                    #activation function (choose from 'ReLU', 'LeakyReLU', 'PReLU', 'ELU'); if using LeakyReLU or ELU, specify alpha factor for scaling negative values gradient
params_dict['filter_size_conv'] = 3                                                     #size of convolutional kernel (can specify as tuple if want non-uniform sized kernel i.e. (5,5,3))
params_dict['filter_size_conv_transpose_pool'] = (2,2,2)                                #size of transposed convolutional / upsampling / max pooling kernel (change accordingly if using thick slice patches (i.e. size (2,2,1) for patch size [128,128,32]))
params_dict['num_repeat_pooling'] = [-1,-1,-1]                                          #number of times to repeat pooling (if -1, then pools as many times as there are levels in the network, else pools only specified number of times along axis); For example, a 5 level network with input patch size of [128, 128, 32] and num_repeat_pooling of [-1, -1, 2] would pool as follows: [128,128,32]->[64,64,16]->[32,32,8]->[16,16,8]->[8,8,8] 
params_dict['padding'] = 'same'                                                         #type of padding for convolutional operations (highly suggested to leave as 'same')
params_dict['use_bias_on_convolutions'] = False                                         #whether to use bias term on convolution (not necessary if applying normalization)
params_dict['use_scale_on_normalization'] = False                                       #whether to use scale term on batch normalization (not necessary if applying piecewise linear activations (such as relu))
params_dict['use_grouped_convolutions'] = [False, 8, 8, False]                          #whether to use grouped convolutions, minimum group size, maximum cardinality, whether to use native grouped convolution or naive loop based implementation (loop based implementation seems to train better (something with the gradients/updates in the optimized version))
params_dict['use_selective_kernel'] = [False, 2, 8, 32]                                 #whether to use selective kernel block, number of branches, reduction ratio for bottleneck, minimum size of bottleneck (highly recommended to use selective kernels with grouped convolution to keep parameter count low (even though this will substantially increase training time))
params_dict['use_cbam'] = [False, 8, 32]                                                #whether to use convolutional block attention module, reduction ratio for bottleneck, minimum size of bottleneck
params_dict['use_attentive_normalization'] = [False, 8, 32, [10,10,15,20]]              #whether to use attentive normalization, reduction ratio for bottleneck, minimum size of bottleneck, value of K at each layer of the network
params_dict['loss'] = ['affine_registration_network_similarity_loss','affine_registration_network_parameter_loss']                            #loss function (from loss_functions.py file)
params_dict['metrics'] = []                                                             #metrics (from loss_functions.py file)
params_dict['monitor'] = ['loss', 'min']                                                #monitor for saving model, and direction ('min' or 'max') that indicates better model performance 
params_dict['pooling'] = 'MaxPool'                                                      #pooling operation (choose from 'MaxPool', 'AveragePooling', 'StridedConvolution', 'ProjectionShortcut' (ProjectionShortcut only usable with residual blocks and cannot be used with atrous convolutions or selective kernels since tensorflow does not support convolutions that are both strided and dilated))
params_dict['anti_aliased_pooling'] = [False, False]                                    #whether to use anti-aliased versions of MaxPool and StridedConvolution via the BlurPool operator; whether to use large 4x4x4 kernel or small 2x2x2 box kernel
params_dict['upsampling'] = ['Trilinear', True]                                         #type of upsampling to use in decoder (choose from 'Deconvolution' or 'Trilinear'); whether to use grouped convolution or sparse weight matrix for upsampling (uses more memory/flops but might actually be slightly faster in practice)
params_dict['regularization_values'] = [0.0, 0.0]                                       #regularization coefficients for L1 and L2 kernal regularizers
params_dict['weight_standardization'] = False                                           #whether to apply weight standardization to all convolution kernels that come before a normalization layer (i.e. force kernel to be zero mean unit variance)
params_dict['dropblock_params'] = None                                                  #probability for dropblock at each level of network; block size at that level; size of feature map at that level; whether to use SpatialDropout instead (will be significantly faster for larger feature maps since this implementation is rather slow)
params_dict['dropblock_scheduler'] = [False, 75]                                        #whether to use scheduling for dropblock, and the number of epochs to reach full weighting
params_dict['conv_weights_initializer'] = 'he_normal'                                   #initializer for convolutional layers (choose from 'he_normal', 'he_uniform', 'glorot_normal', 'glorot_uniform')
params_dict['retina_net_bias_initializer'] = [False, 0.01]                              #whether to use a retina-net style initializer for the bias of the final convolutional layer, and the a priori probability of the foreground class (will help limit massive gradient that occurs due to incorrectly classified background examples)
params_dict['zero_init_for_residual'] = False                                           #whether to set the scale term on the normalization layer to zero at initialization to ensure all gradient flow through the identity connection at the start of training
#general loss function params
params_dict['factor_reweight_foreground_classes'] = [1.]*2            #list of size equal to the number of classes (including background class) (i.e. list of size two for binary task, list of size 4 for task with 3 foreground classes and 1 background, etc) with factor to change weighting of each class in cross entropy losses
params_dict['gamma'] = 2.                                             #gamma power term for focal loss
params_dict['dice_over_batch'] = False                                #whether to treat the batch as a pseudo-volume and calcuate dice over the entire matrix, or calculate dice over each patch individually and average
params_dict['dice_with_background_class'] = False                     #whether to use the background class dice in the loss function
params_dict['use_sigmoid_for_multi_class'] = False                    #if doing multi-class segmentation, whether to use softmax to generate probabilities or use sigmoid to generate (overlapping) class probabilities
params_dict['joint_loss_function_params'] = [1.0, 1.0, False]         #if using joint loss function, 1) initial loss function weighting, 2) decay factor to change weighting over time, 3) whether to apply inverse decay (i.e. weighting for loss function one gets reduced over time while weighting for loss function two gets increased over time)
############################################################################################################

#make output directory if it does not already exist
if not os.path.exists(params_dict['model_outputs_dir']):
    os.makedirs(params_dict['model_outputs_dir'])

if params_dict['mixed_precision'] == True:
    if params_dict['enable_xla']:
        tf.config.optimizer.set_jit(True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
else:
    tf.keras.mixed_precision.set_global_policy('float32')

use_mse_affine_param_loss = [False, True]                                           #whether to use MSE on the 15 raw affine params, or the 12 final affine matrix
use_ncc_loss = [['local', True], ['local', True]]                                   #whether to use local or global normalized cross correlation as similarity metric for affine reg and deformable reg
use_seg_labels_during_training = [True, [True, True]]                               #whether to mask out tumors in the loss; whether to use the fixed and/or moving labels for this purpose
use_seg_loss_for_reg = [True, True]                                                 #whether to use dice loss on fixed and moved labels; whether to use ce loss on fixed and moved labels
use_gradient_loss = True                                                            #whether to use deformation loss to constrain spatial gradients in field
use_dice_ce_loss_for_seg = [True, [True, 3.]]                                       #whether to train segmentation network using dice; using cross-entropy (and the weighting for the positive class)
lambda_factors = [1.0, 1.0, 1.0, 0.1, 0.5, 1.0]                                     #weighting for mse, ncc_affine, ncc_deformable, dice loss for reg, deformation smoothing, and dice loss for seg
#
ndims = 3
num_epochs_until_switch = [params_dict['num_epochs_cycle'] + params_dict['use_lr_warmup'][1], params_dict['num_epochs_cycle'] + params_dict['use_lr_warmup'][1] + int(params_dict['num_epochs_cycle'] * params_dict['cycle_multiplier'])]
#global normalized cross correlation metric
eps = tf.constant(1e-7, 'float32')
def normalized_global_cross_correlation_metric(y_true, y_pred, fixed_label, moving_label_pred_forward_transformed):
    #get images to compute cross correlation over
    im1 = y_true
    im2 = y_pred
    #
    mean_im1 = tf.reduce_mean(im1, axis=range(1,len(im1.shape)-1), keepdims=True)
    mean_im2 = tf.reduce_mean(im2, axis=range(1,len(im2.shape)-1), keepdims=True)
    std_im1 = tf.math.reduce_std(im1, axis=range(1,len(im1.shape)-1), keepdims=True)
    std_im2 = tf.math.reduce_std(im2, axis=range(1,len(im2.shape)-1), keepdims=True)
    #
    norm_im1 = (im1 - mean_im1)/(std_im1 + eps)
    norm_im2 = (im2 - mean_im2)/(std_im2 + eps)
    #
    normalized_cross_correlation = norm_im1 * norm_im2
    #only get loss on nonzero voxels and/or normal voxels (i.e. ignore tumor voxels)
    if use_seg_labels_during_training[0] == False:
        nonzero_indices = tf.not_equal(im1, 0) | tf.not_equal(im2, 0)
    elif (use_seg_labels_during_training[1][0] == False) and (use_seg_labels_during_training[1][1] == True):
        nonzero_indices = (tf.not_equal(im1, 0)) | (tf.not_equal(im2, 0) & tf.equal(moving_label_pred_forward_transformed, 0))
    elif (use_seg_labels_during_training[1][0] == True) and (use_seg_labels_during_training[1][1] == False):
        nonzero_indices = (tf.not_equal(im1, 0) & tf.equal(fixed_label, 0)) | (tf.not_equal(im2, 0))
    else:
        nonzero_indices = (tf.not_equal(im1, 0) & tf.equal(fixed_label, 0)) | (tf.not_equal(im2, 0) & tf.equal(moving_label_pred_forward_transformed, 0))
    normalized_cross_correlation = normalized_cross_correlation[nonzero_indices]
    #reduce mean
    mean_normalized_cross_correlation = tf.reduce_mean(normalized_cross_correlation)
    return mean_normalized_cross_correlation

#global normalized cross correlation loss
def normalized_global_cross_correlation_loss(y_true, y_pred, fixed_label, moving_label_pred_forward_transformed):
    return tf.multiply(-1., normalized_global_cross_correlation_metric(y_true, y_pred, fixed_label=fixed_label, moving_label_pred_forward_transformed=moving_label_pred_forward_transformed))

kernel_size_for_ncc = 9
kernel_size_for_ncc_float32 = tf.constant(kernel_size_for_ncc, tf.float32)
smoothing_conv = tf.keras.layers.Conv3D(filters=1, kernel_size=kernel_size_for_ncc, strides=1, dilation_rate=1, padding='same', use_bias=False, activation=None, kernel_regularizer=None, kernel_initializer=tf.keras.initializers.Constant(1.), kernel_constraint=None, groups=1, dtype=tf.float32, trainable=False)
#local normalized cross correlation metric
def normalized_local_cross_correlation_metric(y_true, y_pred, fixed_label, moving_label_pred_forward_transformed):
    signed = False
    #get images to compute cross correlation over
    im1 = y_true
    im2 = y_pred
    #compute squared terms
    im1_im1 = tf.multiply(im1, im1)
    im2_im2 = tf.multiply(im2, im2)
    im1_im2 = tf.multiply(im1, im2)
    #apply smoothing 9x9x9 kernel to terms
    im1_smooth = smoothing_conv(im1)
    im2_smooth = smoothing_conv(im2)
    im1_im1_smooth = smoothing_conv(im1_im1)
    im2_im2_smooth = smoothing_conv(im2_im2)
    im1_im2_smooth = smoothing_conv(im1_im2)
    #get means of terms
    kernel_normalizing_factor = tf.math.pow(kernel_size_for_ncc_float32, ndims)
    mean_im1_smooth = tf.divide(im1_smooth, kernel_normalizing_factor)
    mean_im2_smooth = tf.divide(im2_smooth, kernel_normalizing_factor)
    #compute cross correlation
    cross = im1_im2_smooth - (im1_smooth * mean_im2_smooth) - (im2_smooth * mean_im1_smooth) + (mean_im1_smooth * mean_im2_smooth * kernel_normalizing_factor)
    var_im1_smooth = im1_im1_smooth - (2 * mean_im1_smooth * im1_smooth) + (mean_im1_smooth * mean_im1_smooth * kernel_normalizing_factor)
    var_im2_smooth = im2_im2_smooth - (2 * mean_im2_smooth * im2_smooth) + (mean_im2_smooth * mean_im2_smooth * kernel_normalizing_factor)
    cross = tf.maximum(cross, eps)
    var_im1_smooth = tf.maximum(var_im1_smooth, eps)
    var_im2_smooth = tf.maximum(var_im2_smooth, eps)
    if signed == True:
        normalized_cross_correlation = tf.divide(cross, tf.sqrt(tf.add(tf.multiply(var_im1_smooth, var_im2_smooth), eps)))
    else:
        normalized_cross_correlation = tf.multiply(tf.divide(cross, var_im1_smooth), tf.divide(cross, var_im2_smooth))
    #only get loss on nonzero voxels and/or normal voxels (i.e. ignore tumor voxels)
    if use_seg_labels_during_training[0] == False:
        nonzero_indices = tf.not_equal(im1, 0) | tf.not_equal(im2, 0)
    elif (use_seg_labels_during_training[1][0] == False) and (use_seg_labels_during_training[1][1] == True):
        nonzero_indices = (tf.not_equal(im1, 0)) | (tf.not_equal(im2, 0) & tf.equal(moving_label_pred_forward_transformed, 0))
    elif (use_seg_labels_during_training[1][0] == True) and (use_seg_labels_during_training[1][1] == False):
        nonzero_indices = (tf.not_equal(im1, 0) & tf.equal(fixed_label, 0)) | (tf.not_equal(im2, 0))
    else:
        nonzero_indices = (tf.not_equal(im1, 0) & tf.equal(fixed_label, 0)) | (tf.not_equal(im2, 0) & tf.equal(moving_label_pred_forward_transformed, 0))
    normalized_cross_correlation = normalized_cross_correlation[nonzero_indices]
    #reduce mean
    mean_normalized_cross_correlation = tf.reduce_mean(normalized_cross_correlation)
    return mean_normalized_cross_correlation

#local normalized cross correlation loss
def normalized_local_cross_correlation_loss(y_true, y_pred, fixed_label, moving_label_pred_forward_transformed):
    return tf.multiply(-1., normalized_local_cross_correlation_metric(y_true, y_pred, fixed_label=fixed_label, moving_label_pred_forward_transformed=moving_label_pred_forward_transformed))

#mean squared error loss
def mean_squared_error(y_true, y_pred):
    mean_mse = tf.reduce_mean(tf.math.square(tf.subtract(y_true, y_pred)))
    return mean_mse

#soft sorensen-dice coefficient metric
def dice_coef_metric(y_true, y_pred, sigmoid_pred):
    #activate outputs and one-hot encode
    y_true = tf.concat([tf.subtract(1., y_true), y_true], axis=-1)
    if sigmoid_pred == True:
        y_pred = tf.keras.activations.sigmoid(y_pred)
    y_pred = tf.concat([tf.subtract(1., y_pred), y_pred], axis=-1)
    #calculate dice metric per class
    num_dims = len(y_true.shape) - 1
    axes_to_sum = range(1,num_dims)
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=axes_to_sum)
    union = tf.add(tf.reduce_sum(y_true, axis=axes_to_sum), tf.reduce_sum(y_pred, axis=axes_to_sum))
    numerator = tf.add(tf.multiply(intersection, 2.), 1.)
    denominator = tf.add(union, 1.)
    dice_metric_per_class = tf.divide(numerator, denominator)
    #average across batch
    dice_metric_per_class = tf.reduce_mean(dice_metric_per_class, axis=0)
    #return average dice metric over classes (not including background)
    return tf.reduce_mean(dice_metric_per_class[1:])

#multi-class hard sorensen-dice coefficient metric (only should be used as metric, not as loss)
def hard_dice_coef_metric(y_true, y_pred, sigmoid_pred):
    #activate outputs and one-hot encode
    if sigmoid_pred == True:
        y_pred = tf.keras.activations.sigmoid(y_pred)
    y_pred_prob = tf.concat([tf.subtract(1., y_pred), y_pred], axis=-1)
    #calculate dice metric per class
    num_dims = len(y_true.shape) - 1
    axes_to_sum = range(1,num_dims)
    #argmax to find predicted class and then one-hot the predicted vector
    y_pred_onehot = tf.cast(tf.one_hot(tf.cast(tf.math.argmax(y_pred_prob, axis=-1), tf.int32), 2), tf.float32)
    y_true_onehot = tf.cast(tf.one_hot(tf.cast(tf.squeeze(y_true, axis=-1), tf.int32), 2), tf.float32)
    #calculate dice metric per class
    intersection = tf.reduce_sum(tf.multiply(y_true_onehot, y_pred_onehot), axis=axes_to_sum)
    union = tf.add(tf.reduce_sum(y_true_onehot, axis=axes_to_sum), tf.reduce_sum(y_pred_onehot, axis=axes_to_sum))
    numerator = tf.multiply(intersection, 2.)
    denominator = union
    dice_metric_per_class = tf.divide(numerator, denominator)
    #replace any NaN values with 1.0 (NaN only occurs when both the ground truth predicted label is empty, which should give a true dice score of 1.0)
    dice_metric_per_class = tf.where(tf.math.is_nan(dice_metric_per_class), tf.ones_like(dice_metric_per_class), dice_metric_per_class)
    #return average dice metric over classes (choosing to use or not use the background class)
    dice_metric_per_class = tf.reduce_mean(dice_metric_per_class, axis=0)
    #return average dice metric over classes (not including background)
    return tf.reduce_mean(dice_metric_per_class[1:])

#hard sorensen-dice coefficient loss
def dice_coef_loss(y_true, y_pred, sigmoid_pred=True):
    return tf.subtract(1., dice_coef_metric(y_true, y_pred, sigmoid_pred=sigmoid_pred))

weights = np.ones((np.repeat(2, 2)))
weights = (weights * np.expand_dims(np.array([1., use_dice_ce_loss_for_seg[1][1]]), axis=-1)).astype(np.float32)
#binary cross-entropy loss
def binary_cross_entropy_loss(y_true, y_pred, from_logits=True):
    cross_entropy_matrix = tf.losses.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
    y_pred_sigmoid = tf.keras.activations.sigmoid(y_pred)
    y_pred_prob = tf.concat([tf.subtract(1., y_pred_sigmoid), y_pred_sigmoid], axis=-1)
    y_true_onehot = tf.cast(tf.one_hot(tf.cast(tf.squeeze(y_true, axis=-1), tf.int32), 2), tf.float32)
    #
    final_mask = tf.zeros_like(y_pred_prob[...,0])
    #argmax to find predicted class and then one-hot the predicted vector
    y_pred_onehot = tf.cast(tf.one_hot(tf.cast(tf.math.argmax(y_pred_prob, axis=-1), tf.int32), 2), tf.float32)
    #make per-voxel weight mask given what our network predicted
    for (i,j) in product(range(0, 2), range(0, 2)):
        w = weights[i][j]
        y_t = y_true_onehot[...,i]
        y_p = y_pred_onehot[...,j]
        final_mask = tf.add(final_mask, tf.multiply(w, tf.multiply(y_t, y_p)))
    return tf.reduce_mean(tf.multiply(final_mask, cross_entropy_matrix))

#gradient loss penalizing both first and second derivatives
def deformation_field_gradient_loss(predicted_deformation, fixed, moved_affine, fixed_label, moving_label, penalty='l2', use_weighting_for_tumors=False, weight_for_tumor_class=0.25, weight_for_background_class=0.25, loss_weighting_factors=[0.01, 0.0, 1.0], ndims=3):
    label_map = tf.clip_by_value(fixed_label + moving_label, clip_value_min=0, clip_value_max=1)
    if use_weighting_for_tumors == True:
        weighting = (1 - label_map) + (weight_for_tumor_class * label_map) + tf.where(tf.equal(fixed, 0) & tf.equal(moved_affine, 0), weight_for_background_class*tf.ones_like(fixed), tf.zeros_like(fixed))
    else:
        weighting = tf.ones_like(label_map)
    #
    dif_loss = 0.0
    #absolute penalty on strength of deformations
    if loss_weighting_factors[0] > 0:
        d_edge = tf.where(tf.equal(fixed, 0) & tf.equal(moved_affine, 0), predicted_deformation, tf.zeros_like(predicted_deformation))
        dif0 = d_edge*d_edge
        dif0 = loss_weighting_factors[0] * tf.reduce_mean(dif0)
        dif_loss = dif_loss + dif0
    #
    #penalty on first derivatives
    if loss_weighting_factors[1] > 0:
        d_x = (predicted_deformation[:,:-2,...] - predicted_deformation[:,2:,...]) / 2
        d_y = (predicted_deformation[:,:,:-2,...] - predicted_deformation[:,:,2:,...]) / 2
        if ndims == 3:
            d_z = (predicted_deformation[:,:,:,:-2,...] - predicted_deformation[:,:,:,2:,...]) / 2
        if penalty == 'l1':
            if ndims == 2:
                dif1 = tf.abs(d_x)[:,:,1:-1,...] + tf.abs(d_y)[:,1:-1,...] 
            if ndims == 3:
                dif1 = tf.abs(d_x)[:,:,1:-1,1:-1,...] + tf.abs(d_y)[:,1:-1,:,1:-1,...] + tf.abs(d_z)[:,1:-1,1:-1,...]
        else:
            if ndims == 2:
                dif1 = (d_x*d_x)[:,:,1:-1,...] + (d_y*d_y)[:,1:-1,...]
            if ndims == 3:
                dif1 = (d_x*d_x)[:,:,1:-1,1:-1,...] + (d_y*d_y)[:,1:-1,:,1:-1,...] + (d_z*d_z)[:,1:-1,1:-1,...]
        if ndims == 2:
            dif1 = weighting[:,1:-1,1:-1,:] * dif1
        else:
            dif1 = weighting[:,1:-1,1:-1,1:-1,:] * dif1
        dif1 = weighting[:,1:-1,...] * dif1
        dif1 = loss_weighting_factors[1] * tf.reduce_mean(dif1)
        dif_loss = dif_loss + dif1
    #
    #second derivative bending penalty
    if loss_weighting_factors[2] > 0:
        d_xx = predicted_deformation[:,:-2,...] - 2*predicted_deformation[:,1:-1,...] + predicted_deformation[:,2:,...]
        d_yy = predicted_deformation[:,:,:-2,...] - 2*predicted_deformation[:,:,1:-1,...] + predicted_deformation[:,:,2:,...]
        if ndims == 3:
            d_zz = predicted_deformation[:,:,:,:-2,...] - 2*predicted_deformation[:,:,:,1:-1,...] + predicted_deformation[:,:,:,2:,...]
        d_xy = (predicted_deformation[:,:-2,:-2,...] - predicted_deformation[:,2:,:-2,...] - predicted_deformation[:,:-2,2:,...] + predicted_deformation[:,2:,2:,...]) / 4
        if ndims == 3:
            d_xz = (predicted_deformation[:,:-2,:,:-2,...] - predicted_deformation[:,2:,:,:-2,...] - predicted_deformation[:,:-2,:,2:,...] + predicted_deformation[:,2:,:,2:,...]) / 4
            d_yz = (predicted_deformation[:,:,:-2,:-2,...] - predicted_deformation[:,:,2:,:-2,...] - predicted_deformation[:,:,:-2,2:,...] + predicted_deformation[:,:,2:,2:,...]) / 4
        if ndims == 2:
            dif2 = (d_xx * d_xx)[:,:,1:-1,...] + (d_yy * d_yy)[:,1:-1,...] + 2*(d_xy * d_xy)
        if ndims == 3:
            dif2 = (d_xx * d_xx)[:,:,1:-1,1:-1,...] + (d_yy * d_yy)[:,1:-1,:,1:-1,...] + 2*(d_xy * d_xy)[:,:,:,1:-1,...] + (d_zz * d_zz)[:,1:-1,1:-1,...] + 2*(d_xz * d_xz)[:,:,1:-1,...] + 2*(d_yz * d_yz)[:,1:-1,...]
        if ndims == 2:
            dif2 = weighting[:,1:-1,1:-1,:] * dif2
        else:
            dif2 = weighting[:,1:-1,1:-1,1:-1,:] * dif2
        dif2 = loss_weighting_factors[2] * tf.reduce_mean(dif2)
        dif_loss = dif_loss + dif2
    #
    #return joint loss
    return dif_loss

#registration loss for 32 sized inputs using mse, ncc, and dice/ce
def total_loss_reg_32(fixed_32, fixed_label_32, true_affine_params, true_affine_matrix, affine_params_32, affine_matrix_32, moved_affine_32, moved_label_affine_32, predicted_deformation_field_32, moved_deformable_32, moved_label_deformable_32, mask_for_dice_loss_label_32, mask_for_NCC_deformable_loss_label_32):
    loss = 0.0
    mse_affine_params_loss, local_ncc_loss_affine, global_ncc_loss_affine, local_ncc_loss_deformable, global_ncc_loss_deformable, reg_dice_loss, reg_ce_loss, deformation_loss = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    #loss on affine registration
    if use_mse_affine_param_loss[0] == True:
        mse_affine_params_loss = mean_squared_error(true_affine_params, affine_params_32)
        loss = loss + lambda_factors[0] * mse_affine_params_loss
    elif use_mse_affine_param_loss[1] == True:
        mse_affine_params_loss = mean_squared_error(true_affine_matrix, affine_matrix_32)
        loss = loss + lambda_factors[0] * mse_affine_params_loss
    if use_ncc_loss[0][1] == True:
        if use_ncc_loss[0][0] == 'local':
            local_ncc_loss_affine = normalized_local_cross_correlation_loss(fixed_32, moved_affine_32, fixed_label_32, moved_label_affine_32)
            loss = loss + lambda_factors[1] * local_ncc_loss_affine
        else:
            global_ncc_loss_affine = normalized_global_cross_correlation_loss(fixed_32, moved_affine_32, fixed_label_32, moved_label_affine_32)
            loss = loss + lambda_factors[1] * global_ncc_loss_affine
    #loss on deformable registration
    if use_ncc_loss[1][1] == True:
        if use_ncc_loss[1][0] == 'local':
            local_ncc_loss_deformable = normalized_local_cross_correlation_loss(fixed_32, moved_deformable_32, mask_for_NCC_deformable_loss_label_32, mask_for_NCC_deformable_loss_label_32)
            loss = loss + lambda_factors[2] * local_ncc_loss_deformable
        else:
            global_ncc_loss_deformable = normalized_global_cross_correlation_loss(fixed_32, moved_deformable_32, mask_for_NCC_deformable_loss_label_32, mask_for_NCC_deformable_loss_label_32)
            loss = loss + lambda_factors[2] * global_ncc_loss_deformable
    if use_seg_loss_for_reg[0] == True:
        reg_dice_loss = dice_coef_loss(mask_for_dice_loss_label_32, moved_label_deformable_32, sigmoid_pred=False)
        loss = loss + lambda_factors[3] * reg_dice_loss
    if use_seg_loss_for_reg[1] == True:
        reg_ce_loss = binary_cross_entropy_loss(mask_for_dice_loss_label_32, moved_label_deformable_32, from_logits=False)
        loss = loss + lambda_factors[3] * reg_ce_loss
    if use_gradient_loss == True:
        deformation_loss = deformation_field_gradient_loss(predicted_deformation_field_32, fixed_32, moved_affine_32, fixed_label_32, moved_label_affine_32)
        loss = loss + lambda_factors[4] * deformation_loss
    return [loss, mse_affine_params_loss, local_ncc_loss_affine, global_ncc_loss_affine, local_ncc_loss_deformable, global_ncc_loss_deformable, reg_dice_loss, reg_ce_loss, deformation_loss]

#registration loss for 64 sized inputs using mse, ncc, and dice/ce
def total_loss_reg_64(fixed_64, fixed_label_64, true_affine_params, true_affine_matrix, affine_params_32, affine_params_64, affine_matrix_32, affine_matrix_64, moved_affine_32, moved_affine_64, moved_label_affine_32, moved_label_affine_64, predicted_deformation_field_32, predicted_deformation_field_64, moved_deformable_32, moved_deformable_64, moved_label_deformable_32, moved_label_deformable_64, mask_for_dice_loss_label_64, mask_for_NCC_deformable_loss_label_64):
    loss = 0.0
    mse_affine_params_loss, local_ncc_loss_affine, global_ncc_loss_affine, local_ncc_loss_deformable, global_ncc_loss_deformable, reg_dice_loss, reg_ce_loss, deformation_loss = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    mse_affine_params_loss_32, local_ncc_loss_affine_32, global_ncc_loss_affine_32, local_ncc_loss_deformable_32, global_ncc_loss_deformable_32, reg_dice_loss_32, reg_ce_loss_32, deformation_loss_32 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    mse_affine_params_loss_64, local_ncc_loss_affine_64, global_ncc_loss_affine_64, local_ncc_loss_deformable_64, global_ncc_loss_deformable_64, reg_dice_loss_64, reg_ce_loss_64, deformation_loss_64 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    #
    fixed_32 = fixed_64[:,::2,::2,::2,:]
    fixed_label_32 = fixed_label_64[:,::2,::2,::2,:]
    mask_for_dice_loss_label_32 = mask_for_dice_loss_label_64[:,::2,::2,::2,:]
    mask_for_NCC_deformable_loss_label_32 = mask_for_NCC_deformable_loss_label_64[:,::2,::2,::2,:]
    #loss on affine registration
    if use_mse_affine_param_loss[0] == True:
        mse_affine_params_loss_32 = mean_squared_error(true_affine_params, affine_params_32)
        mse_affine_params_loss_64 = mean_squared_error(true_affine_matrix, affine_matrix_64)
        mse_affine_params_loss = (0.5 * mse_affine_params_loss_32) + (1.0 * mse_affine_params_loss_64)
        loss = loss + lambda_factors[0] * mse_affine_params_loss
    elif use_mse_affine_param_loss[1] == True:
        mse_affine_params_loss_32 = mean_squared_error(true_affine_matrix, affine_matrix_32)
        mse_affine_params_loss_64 = mean_squared_error(true_affine_matrix, affine_matrix_64)
        mse_affine_params_loss = (0.5 * mse_affine_params_loss_32) + (1.0 * mse_affine_params_loss_64)
        loss = loss + lambda_factors[0] * mse_affine_params_loss
    if use_ncc_loss[0][1] == True:
        if use_ncc_loss[0][0] == 'local':
            local_ncc_loss_affine_32 = normalized_local_cross_correlation_loss(fixed_32, moved_affine_32, fixed_label_32, moved_label_affine_32)
            local_ncc_loss_affine_64 = normalized_local_cross_correlation_loss(fixed_64, moved_affine_64, fixed_label_64, moved_label_affine_64)
            local_ncc_loss_affine = (0.5 * local_ncc_loss_affine_32) + (1.0 * local_ncc_loss_affine_64)
            loss = loss + lambda_factors[1] * local_ncc_loss_affine
        else:
            global_ncc_loss_affine_32 = normalized_global_cross_correlation_loss(fixed_32, moved_affine_32, fixed_label_32, moved_label_affine_32)
            global_ncc_loss_affine_64 = normalized_global_cross_correlation_loss(fixed_64, moved_affine_64, fixed_label_64, moved_label_affine_64)
            global_ncc_loss_affine = (0.5 * global_ncc_loss_affine_32) + (1.0 * global_ncc_loss_affine_64)
            loss = loss + lambda_factors[1] * global_ncc_loss_affine
    #loss on deformable registration
    if use_ncc_loss[1][1] == True:
        if use_ncc_loss[1][0] == 'local':
            local_ncc_loss_deformable_32 = normalized_local_cross_correlation_loss(fixed_32, moved_deformable_32, mask_for_NCC_deformable_loss_label_32, mask_for_NCC_deformable_loss_label_32)
            local_ncc_loss_deformable_64 = normalized_local_cross_correlation_loss(fixed_64, moved_deformable_64, mask_for_NCC_deformable_loss_label_64, mask_for_NCC_deformable_loss_label_64)
            local_ncc_loss_deformable = (0.5 * local_ncc_loss_deformable_32) + (1.0 * local_ncc_loss_deformable_64)
            loss = loss + lambda_factors[2] * local_ncc_loss_deformable
        else:
            global_ncc_loss_deformable_32 = normalized_global_cross_correlation_loss(fixed_32, moved_deformable_32, mask_for_NCC_deformable_loss_label_32, mask_for_NCC_deformable_loss_label_32)
            global_ncc_loss_deformable_64 = normalized_global_cross_correlation_loss(fixed_64, moved_deformable_64, mask_for_NCC_deformable_loss_label_64, mask_for_NCC_deformable_loss_label_64)
            global_ncc_loss_deformable = (0.5 * global_ncc_loss_deformable_32) + (1.0 * global_ncc_loss_deformable_64)
            loss = loss + lambda_factors[2] * global_ncc_loss_deformable
    if use_seg_loss_for_reg[0] == True:
        reg_dice_loss_32 = dice_coef_loss(mask_for_dice_loss_label_32, moved_label_deformable_32, sigmoid_pred=False)
        reg_dice_loss_64 = dice_coef_loss(mask_for_dice_loss_label_64, moved_label_deformable_64, sigmoid_pred=False)
        reg_dice_loss = (0.5 * reg_dice_loss_32) + (1.0 * reg_dice_loss_64)
        loss = loss + lambda_factors[3] * reg_dice_loss
    if use_seg_loss_for_reg[1] == True:
        reg_ce_loss_32 = binary_cross_entropy_loss(mask_for_dice_loss_label_32, moved_label_deformable_32, from_logits=False)
        reg_ce_loss_64 = binary_cross_entropy_loss(mask_for_dice_loss_label_64, moved_label_deformable_64, from_logits=False)
        reg_ce_loss = (0.5 * reg_ce_loss_32) + (1.0 * reg_ce_loss_64)
        loss = loss + lambda_factors[3] * reg_ce_loss
    if use_gradient_loss == True:
        deformation_loss_32 = deformation_field_gradient_loss(predicted_deformation_field_32, fixed_32, moved_affine_32, fixed_label_32, moved_label_affine_32)
        deformation_loss_64 = deformation_field_gradient_loss(predicted_deformation_field_64, fixed_64, moved_affine_64, fixed_label_64, moved_label_affine_64)
        deformation_loss = (0.5 * deformation_loss_32) + (1.0 * deformation_loss_64)
        loss = loss + lambda_factors[4] * deformation_loss
    return [loss, mse_affine_params_loss_64, local_ncc_loss_affine_64, global_ncc_loss_affine_64, local_ncc_loss_deformable_64, global_ncc_loss_deformable_64, reg_dice_loss_64, reg_ce_loss_64, deformation_loss_64]

#registration loss for 128 sized inputs using mse, ncc, and dice/ce
def total_loss_reg_128(fixed_128, fixed_label_128, true_affine_params, true_affine_matrix, affine_params_32, affine_params_64, affine_params_128, affine_matrix_32, affine_matrix_64, affine_matrix_128, moved_affine_32, moved_affine_64, moved_affine_128, moved_label_affine_32, moved_label_affine_64, moved_label_affine_128, predicted_deformation_field_32, predicted_deformation_field_64, predicted_deformation_field_128, moved_deformable_32, moved_deformable_64, moved_deformable_128, moved_label_deformable_32, moved_label_deformable_64, moved_label_deformable_128, mask_for_dice_loss_label_128, mask_for_NCC_deformable_loss_label_128):
    loss = 0.0
    mse_affine_params_loss, local_ncc_loss_affine, global_ncc_loss_affine, local_ncc_loss_deformable, global_ncc_loss_deformable, reg_dice_loss, reg_ce_loss, deformation_loss = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    mse_affine_params_loss_32, local_ncc_loss_affine_32, global_ncc_loss_affine_32, local_ncc_loss_deformable_32, global_ncc_loss_deformable_32, reg_dice_loss_32, reg_ce_loss_32, deformation_loss_32 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    mse_affine_params_loss_64, local_ncc_loss_affine_64, global_ncc_loss_affine_64, local_ncc_loss_deformable_64, global_ncc_loss_deformable_64, reg_dice_loss_64, reg_ce_loss_64, deformation_loss_64 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    mse_affine_params_loss_128, local_ncc_loss_affine_128, global_ncc_loss_affine_128, local_ncc_loss_deformable_128, global_ncc_loss_deformable_128, reg_dice_loss_128, reg_ce_loss_128, deformation_loss_128 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    #
    fixed_32 = fixed_128[:,::4,::4,::4,:]
    fixed_label_32 = fixed_label_128[:,::4,::4,::4,:]
    fixed_64 = fixed_128[:,::2,::2,::2,:]
    fixed_label_64 = fixed_label_128[:,::2,::2,::2,:]
    mask_for_dice_loss_label_32 = mask_for_dice_loss_label_128[:,::4,::4,::4,:]
    mask_for_NCC_deformable_loss_label_32 = mask_for_NCC_deformable_loss_label_128[:,::4,::4,::4,:]
    mask_for_dice_loss_label_64 = mask_for_dice_loss_label_128[:,::2,::2,::2,:]
    mask_for_NCC_deformable_loss_label_64 = mask_for_NCC_deformable_loss_label_128[:,::2,::2,::2,:]
    #loss on affine registration
    if use_mse_affine_param_loss[0] == True:
        mse_affine_params_loss_32 = mean_squared_error(true_affine_params, affine_params_32)
        mse_affine_params_loss_64 = mean_squared_error(true_affine_matrix, affine_matrix_64)
        mse_affine_params_loss_128 = mean_squared_error(true_affine_matrix, affine_matrix_128)
        mse_affine_params_loss = (0.25 * mse_affine_params_loss_32) + (0.5 * mse_affine_params_loss_64) + (1.0 * mse_affine_params_loss_128)
        loss = loss + lambda_factors[0] * mse_affine_params_loss
    elif use_mse_affine_param_loss[1] == True:
        mse_affine_params_loss_32 = mean_squared_error(true_affine_matrix, affine_matrix_32)
        mse_affine_params_loss_64 = mean_squared_error(true_affine_matrix, affine_matrix_64)
        mse_affine_params_loss_128 = mean_squared_error(true_affine_matrix, affine_matrix_128)
        mse_affine_params_loss = (0.25 * mse_affine_params_loss_32) + (0.5 * mse_affine_params_loss_64) + (1.0 * mse_affine_params_loss_128)
        loss = loss + lambda_factors[0] * mse_affine_params_loss
    if use_ncc_loss[0][1] == True:
        if use_ncc_loss[0][0] == 'local':
            local_ncc_loss_affine_32 = normalized_local_cross_correlation_loss(fixed_32, moved_affine_32, fixed_label_32, moved_label_affine_32)
            local_ncc_loss_affine_64 = normalized_local_cross_correlation_loss(fixed_64, moved_affine_64, fixed_label_64, moved_label_affine_64)
            local_ncc_loss_affine_128 = normalized_local_cross_correlation_loss(fixed_128, moved_affine_128, fixed_label_128, moved_label_affine_128)
            local_ncc_loss_affine = (0.25 * local_ncc_loss_affine_32) + (0.5 * local_ncc_loss_affine_64) + (1.0 * local_ncc_loss_affine_128)
            loss = loss + lambda_factors[1] * local_ncc_loss_affine
        else:
            global_ncc_loss_affine_32 = normalized_global_cross_correlation_loss(fixed_32, moved_affine_32, fixed_label_32, moved_label_affine_32)
            global_ncc_loss_affine_64 = normalized_global_cross_correlation_loss(fixed_64, moved_affine_64, fixed_label_64, moved_label_affine_64)
            global_ncc_loss_affine_128 = normalized_global_cross_correlation_loss(fixed_128, moved_affine_128, fixed_label_128, moved_label_affine_128)
            global_ncc_loss_affine = (0.25 * global_ncc_loss_affine_32) + (0.5 * global_ncc_loss_affine_64) + (1.0 * global_ncc_loss_affine_128)
            loss = loss + lambda_factors[1] * global_ncc_loss_affine
    #loss on deformable registration
    if use_ncc_loss[1][1] == True:
        if use_ncc_loss[1][0] == 'local':
            local_ncc_loss_deformable_32 = normalized_local_cross_correlation_loss(fixed_32, moved_deformable_32, mask_for_NCC_deformable_loss_label_32, mask_for_NCC_deformable_loss_label_32)
            local_ncc_loss_deformable_64 = normalized_local_cross_correlation_loss(fixed_64, moved_deformable_64, mask_for_NCC_deformable_loss_label_64, mask_for_NCC_deformable_loss_label_64)
            local_ncc_loss_deformable_128 = normalized_local_cross_correlation_loss(fixed_128, moved_deformable_128, mask_for_NCC_deformable_loss_label_128, mask_for_NCC_deformable_loss_label_128)
            local_ncc_loss_deformable = (0.25 * local_ncc_loss_deformable_32) + (0.5 * local_ncc_loss_deformable_64) + (1.0 * local_ncc_loss_deformable_128)
            loss = loss + lambda_factors[2] * local_ncc_loss_deformable
        else:
            global_ncc_loss_deformable_32 = normalized_global_cross_correlation_loss(fixed_32, moved_deformable_32, mask_for_NCC_deformable_loss_label_32, mask_for_NCC_deformable_loss_label_32)
            global_ncc_loss_deformable_64 = normalized_global_cross_correlation_loss(fixed_64, moved_deformable_64, mask_for_NCC_deformable_loss_label_64, mask_for_NCC_deformable_loss_label_64)
            global_ncc_loss_deformable_128 = normalized_global_cross_correlation_loss(fixed_128, moved_deformable_128, mask_for_NCC_deformable_loss_label_128, mask_for_NCC_deformable_loss_label_128)
            global_ncc_loss_deformable = (0.25 * global_ncc_loss_deformable_32) + (0.5 * global_ncc_loss_deformable_64) + (1.0 * global_ncc_loss_deformable_128)
            loss = loss + lambda_factors[2] * global_ncc_loss_deformable
    if use_seg_loss_for_reg[0] == True:
        reg_dice_loss_32 = dice_coef_loss(mask_for_dice_loss_label_32, moved_label_deformable_32, sigmoid_pred=False)
        reg_dice_loss_64 = dice_coef_loss(mask_for_dice_loss_label_64, moved_label_deformable_64, sigmoid_pred=False)
        reg_dice_loss_128 = dice_coef_loss(mask_for_dice_loss_label_128, moved_label_deformable_128, sigmoid_pred=False)
        reg_dice_loss = (0.25 * reg_dice_loss_32) + (0.5 * reg_dice_loss_64) + (1.0 * reg_dice_loss_128)
        loss = loss + lambda_factors[3] * reg_dice_loss
    if use_seg_loss_for_reg[1] == True:
        reg_ce_loss_32 = binary_cross_entropy_loss(mask_for_dice_loss_label_32, moved_label_deformable_32, from_logits=False)
        reg_ce_loss_64 = binary_cross_entropy_loss(mask_for_dice_loss_label_64, moved_label_deformable_64, from_logits=False)
        reg_ce_loss_128 = binary_cross_entropy_loss(mask_for_dice_loss_label_128, moved_label_deformable_128, from_logits=False)
        reg_ce_loss = (0.25 * reg_ce_loss_32) + (0.5 * reg_ce_loss_64) + (1.0 * reg_ce_loss_128)
        loss = loss + lambda_factors[3] * reg_ce_loss
    if use_gradient_loss == True:
        deformation_loss_32 = deformation_field_gradient_loss(predicted_deformation_field_32, fixed_32, moved_affine_32, fixed_label_32, moved_label_affine_32)
        deformation_loss_64 = deformation_field_gradient_loss(predicted_deformation_field_64, fixed_64, moved_affine_64, fixed_label_64, moved_label_affine_64)
        deformation_loss_128 = deformation_field_gradient_loss(predicted_deformation_field_128, fixed_128, moved_affine_128, fixed_label_128, moved_label_affine_128)
        deformation_loss = (0.25 * deformation_loss_32) + (0.5 * deformation_loss_64) + (1.0 * deformation_loss_128)
        loss = loss + lambda_factors[4] * deformation_loss
    return [loss, mse_affine_params_loss_128, local_ncc_loss_affine_128, global_ncc_loss_affine_128, local_ncc_loss_deformable_128, global_ncc_loss_deformable_128, reg_dice_loss_128, reg_ce_loss_128, deformation_loss_128]

#segmentation loss using dice and cross entropy
def total_loss_seg(fixed_label, pred_seg_final):
    loss = 0.0
    seg_dice_loss, seg_dice_metric, seg_ce_loss = np.nan, np.nan, np.nan
    if use_dice_ce_loss_for_seg[0] == True:
        seg_dice_loss = dice_coef_loss(fixed_label, pred_seg_final, sigmoid_pred=True)
        seg_dice_metric = hard_dice_coef_metric(fixed_label, pred_seg_final, sigmoid_pred=True)
        loss = loss + lambda_factors[5] * seg_dice_loss
    if use_dice_ce_loss_for_seg[1][0] == True:
        seg_ce_loss = binary_cross_entropy_loss(fixed_label, pred_seg_final, from_logits=True)
        loss = loss + lambda_factors[5] * seg_ce_loss
    return [loss, seg_dice_metric, seg_ce_loss]

#callback to control cyclic learning rate schedule
class CustomLearningRateSchedules(tf.keras.callbacks.Callback):
    def __init__(self, params_dict, learning_rate, weight_decay, iterations_per_epoch, optimizer):
        super(CustomLearningRateSchedules, self).__init__()
        #unpack relevant dictionary elements
        lr_schedule_type, use_lr_warmup, step_epoch_nums, step_factors, cycle, num_epochs_cycle, factor_decrease_learning_rate, cycle_multiplier, factor_decrease_max_rate_per_cycle, factor_decrease_min_rate_per_cycle, power = unpack_keys(params_dict)
        self.iterations_per_epoch = iterations_per_epoch
        self.lr_schedule_type = lr_schedule_type
        self.initial_learning_rate = learning_rate
        self.initial_weight_decay = weight_decay
        self.current_iteration = 0
        self.current_epoch = 0
        #params for learning rate warmup
        self.use_lr_warmup = use_lr_warmup
        self.min_lr_warmup = self.initial_learning_rate / self.use_lr_warmup[2]
        self.min_wd_warmup = self.initial_weight_decay / self.use_lr_warmup[2]
        self.current_warmup_step = 0.0
        #params for step decay
        self.step_epoch_nums = np.array(step_epoch_nums)
        self.step_factors = step_factors
        #params for cosine annealing learning rate decay (and exponential)
        self.cycle = cycle
        self.num_epochs_cycle = num_epochs_cycle
        self.factor_decrease_learning_rate = factor_decrease_learning_rate
        self.cycle_multiplier = cycle_multiplier
        self.factor_decrease_max_rate_per_cycle = factor_decrease_max_rate_per_cycle
        self.factor_decrease_min_rate_per_cycle = factor_decrease_min_rate_per_cycle
        self.power = power
        self.min_lr = self.initial_learning_rate / self.factor_decrease_learning_rate
        self.min_wd= self.initial_weight_decay / self.factor_decrease_learning_rate
        self.current_step = 0.0
        #model optimizer
        self.optimizer = optimizer
    #
    def on_train_batch_begin(self, batch, logs=None):
        if self.use_lr_warmup[0] == True and self.current_epoch < self.use_lr_warmup[1]:
            self.update_warmup()
        elif self.lr_schedule_type == 'StepDecay':
            self.update_step_decay()
        elif self.lr_schedule_type == 'CosineAnneal':
            self.update_cosine_anneal()
        elif self.lr_schedule_type == 'Exponential':
            self.update_exponential()
        #update global iteration counter
        self.current_iteration = self.current_iteration + 1
        #update global epoch counter
        if self.current_iteration % self.iterations_per_epoch == 0:
            self.current_epoch = self.current_epoch + 1
    #
    def update_warmup(self):
        current_position = self.current_warmup_step / (self.iterations_per_epoch * self.use_lr_warmup[1])
        new_lr_warmup = self.min_lr_warmup + (self.initial_learning_rate - self.min_lr_warmup) * current_position
        if hasattr(self.optimizer, 'weight_decay') and tf.is_tensor(self.optimizer.weight_decay):
            new_wd_warmup = self.min_wd_warmup + (self.initial_weight_decay - self.min_wd_warmup) * current_position
        else:
            new_wd_warmup = None
        self.set_learning_rate_and_weight_decay(new_lr_warmup, new_wd_warmup)
        self.current_warmup_step = self.current_warmup_step + 1
    #
    def update_step_decay(self):
        current_step_decay = np.where(self.current_epoch < self.step_epoch_nums)[0]
        if np.any(current_step_decay):
            factor_reduce_index = current_step_decay[0]
        else:
            factor_reduce_index = -1
        new_lr_step_decay = self.initial_learning_rate / self.step_factors[factor_reduce_index]
        if hasattr(self.optimizer, 'weight_decay') and tf.is_tensor(self.optimizer.weight_decay):
            new_wd_step_decay = self.initial_weight_decay / self.step_factors[factor_reduce_index]
        else:
            new_wd_step_decay = None
        self.set_learning_rate_and_weight_decay(new_lr_step_decay, new_wd_step_decay)
    #
    def update_cosine_anneal(self):
        current_position = self.update_general()
        new_lr_cosine = self.min_lr + 0.5 * (self.initial_learning_rate - self.min_lr) * (1 + np.cos(current_position * np.pi))
        if hasattr(self.optimizer, 'weight_decay') and tf.is_tensor(self.optimizer.weight_decay):
            new_wd_cosine = self.min_wd + 0.5 * (self.initial_weight_decay - self.min_wd) * (1 + np.cos(current_position * np.pi))
        else:
            new_wd_cosine = None
        self.set_learning_rate_and_weight_decay(new_lr_cosine, new_wd_cosine)
    #
    def update_exponential(self):
        current_position = self.update_general()
        new_lr_exponential = self.min_lr + (self.initial_learning_rate - self.min_lr) * ((1.0 - current_position) ** self.power)
        if hasattr(self.optimizer, 'weight_decay') and tf.is_tensor(self.optimizer.weight_decay):
            new_wd_exponential = self.min_wd + (self.initial_weight_decay - self.min_wd) * ((1.0 - current_position) ** self.power)
        else:
            new_wd_exponential = None
        self.set_learning_rate_and_weight_decay(new_lr_exponential, new_wd_exponential)
    #
    def update_general(self):
        current_position = self.current_step / (self.iterations_per_epoch * self.num_epochs_cycle)
        if self.cycle == True:
            if current_position == 1.0:
                #end of cycle
                self.current_step = 0
                self.num_epochs_cycle = self.num_epochs_cycle * self.cycle_multiplier
                self.initial_learning_rate = self.initial_learning_rate / self.factor_decrease_max_rate_per_cycle
                self.min_lr = self.min_lr / self.factor_decrease_min_rate_per_cycle
                self.initial_weight_decay = self.initial_weight_decay / self.factor_decrease_max_rate_per_cycle
                self.min_wd = self.min_wd / self.factor_decrease_min_rate_per_cycle
        else:
            current_position = min(current_position, 1)
        self.current_step = self.current_step + 1
        return current_position
    #
    def set_learning_rate_and_weight_decay(self, new_lr, new_wd=None):
        tf.keras.backend.set_value(self.optimizer.learning_rate, new_lr)
        if new_wd != None:
            tf.keras.backend.set_value(self.optimizer.weight_decay, new_wd)
        tf.summary.scalar('learning_rate', data=new_lr, step=self.current_iteration)

#callback to save model at best validation metric and save images out 
class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, manager, params_dict, validation_data_x, validation_data_y, save_model_every_n_epochs=[False, 50]):
        super(SaveModelCallback, self).__init__()
        self.manager = manager
        self.loss_value = np.Inf
        self.output_file = params_dict['output_file']
        self.logits_clip_value = params_dict['logits_clip_value']
        self.epoch_save = 0
        self.start_epoch = 0.0
        self.num_decimals_to_round = 3
        self.save_model_every_n_epochs = save_model_every_n_epochs
        #
        self.fixed = tf.cast(tf.expand_dims(validation_data_x[...,0], axis=-1), tf.float32)
        self.moving = tf.cast(tf.expand_dims(validation_data_x[...,1], axis=-1), tf.float32)
        self.fixed_label = tf.cast(tf.expand_dims(validation_data_y[0][...,0], axis=-1), tf.float32)
        self.moving_label = tf.cast(tf.expand_dims(validation_data_y[0][...,1], axis=-1), tf.float32)
        self.true_affine_params = tf.cast(validation_data_y[1], tf.float32)
        self.true_affine_matrix = make_affine_matrix_from_params(self.true_affine_params, ndims=3)
        self.num_images_to_plot = min(10, self.fixed.shape[0])
        self.index = 0
        self.image_names = ['brain_fixed.nii.gz', 'brain_fixed-label.nii.gz', 'brain_moving.nii.gz', 'brain_moving-label.nii.gz', 'brain_moved_affine.nii.gz', 'brain_moved_affine-label.nii.gz', 'brain_moved_deformable.nii.gz', 'brain_moved_deformable-label.nii.gz', 'brain_deformation_field.nii.gz', 'brain_fixed_pred_1-label.nii.gz', 'brain_fixed_pred_2-label.nii.gz']
        self.final_png_name = 'brains.png'
        self.final_flow_png_name = 'flow.png'
    #
    def on_epoch_begin(self, epoch, logs=None):
        self.start_epoch = time.time()
    #
    def on_epoch_end(self, epoch, logs=None):
        #save epoch statistics to output file
        self.save_epoch_stats_to_output_file(epoch, logs)
        #if validation loss has improved, save model
        #current_loss = 1 - logs['val_Seg_Dice_2']
        current_loss = logs['val_Reg_Loss']
        if current_loss < self.loss_value:
            self.loss_value = current_loss
            self.epoch_save = epoch
            self.save_my_model()
            self.plot_images()
        elif self.save_model_every_n_epochs[0] == True:
            if epoch % self.save_model_every_n_epochs[1] == 0:
                self.epoch_save = epoch
                self.save_my_model()
                self.plot_images()
    #
    def save_epoch_stats_to_output_file(self, epoch, logs):
        timeForEpoch = int(time.time() - self.start_epoch)
        epoch_summary_str = 'Epoch ' + str(epoch) + ': ' + str(timeForEpoch) + 's'
        train_summary_str = 'Train Reg Loss = ' + str(np.round(logs['Reg_Loss'], self.num_decimals_to_round))
        val_summary_str = 'Val Reg Loss = ' + str(np.round(logs['val_Reg_Loss'], self.num_decimals_to_round))
        train_summary_str = '\tTrain Seg Loss = ' + str(np.round(logs['Seg_Loss_2'], self.num_decimals_to_round))
        val_summary_str = '\tVal Seg Loss = ' + str(np.round(logs['val_Seg_Loss_2'], self.num_decimals_to_round))
        if not np.isnan(logs['MSE']):
            train_summary_str = train_summary_str + '\tMSE = ' + str(np.round(logs['MSE'], self.num_decimals_to_round))
            val_summary_str = val_summary_str + '\tMSE = ' + str(np.round(logs['val_MSE'], self.num_decimals_to_round))
        if not np.isnan(logs['Local_NCC_Affine']):
            train_summary_str = train_summary_str + '\tLocal_NCC_Affine = ' + str(np.round(logs['Local_NCC_Affine'], self.num_decimals_to_round))
            val_summary_str = val_summary_str + '\tLocal_NCC_Affine = ' + str(np.round(logs['val_Local_NCC_Affine'], self.num_decimals_to_round))
        if not np.isnan(logs['Global_NCC_Affine']):
            train_summary_str = train_summary_str + '\tGlobal_NCC_Affine = ' + str(np.round(logs['Global_NCC_Affine'], self.num_decimals_to_round))
            val_summary_str = val_summary_str + '\tGlobal_NCC_Affine = ' + str(np.round(logs['val_Global_NCC_Affine'], self.num_decimals_to_round))
        if not np.isnan(logs['Local_NCC_Deformable']):
            train_summary_str = train_summary_str + '\tLocal_NCC_Deformable = ' + str(np.round(logs['Local_NCC_Deformable'], self.num_decimals_to_round))
            val_summary_str = val_summary_str + '\tLocal_NCC_Deformable = ' + str(np.round(logs['val_Local_NCC_Deformable'], self.num_decimals_to_round))
        if not np.isnan(logs['Global_NCC_Deformable']):
            train_summary_str = train_summary_str + '\tGlobal_NCC_Deformable = ' + str(np.round(logs['Global_NCC_Deformable'], self.num_decimals_to_round))
            val_summary_str = val_summary_str + '\tGlobal_NCC_Deformable = ' + str(np.round(logs['val_Global_NCC_Deformable'], self.num_decimals_to_round))
        if not np.isnan(logs['Reg_Dice']):
            #convert dice loss to dice metric by subtracting from 1
            train_summary_str = train_summary_str + '\tReg_Dice = ' + str(np.round(1. - logs['Reg_Dice'], self.num_decimals_to_round))
            val_summary_str = val_summary_str + '\tReg_Dice = ' + str(np.round(1. - logs['val_Reg_Dice'], self.num_decimals_to_round))
        if not np.isnan(logs['Reg_CE']):
            train_summary_str = train_summary_str + '\tReg_CE = ' + str(np.round(logs['Reg_CE'], self.num_decimals_to_round))
            val_summary_str = val_summary_str + '\tReg_CE = ' + str(np.round(logs['val_Reg_CE'], self.num_decimals_to_round))
        if not np.isnan(logs['Deformation']):
            train_summary_str = train_summary_str + '\tDeformation = ' + str(np.round(logs['Deformation'], self.num_decimals_to_round))
            val_summary_str = val_summary_str + '\tDeformation = ' + str(np.round(logs['val_Deformation'], self.num_decimals_to_round))
        if not np.isnan(logs['Seg_Dice_1']):
            train_summary_str = train_summary_str + '\tSeg_Dice_1 = ' + str(np.round(logs['Seg_Dice_1'], self.num_decimals_to_round))
            val_summary_str = val_summary_str + '\tSeg_Dice_1 = ' + str(np.round(logs['val_Seg_Dice_1'], self.num_decimals_to_round))
        if not np.isnan(logs['Seg_CE_1']):
            train_summary_str = train_summary_str + '\tSeg_CE_1 = ' + str(np.round(logs['Seg_CE_1'], self.num_decimals_to_round))
            val_summary_str = val_summary_str + '\tSeg_CE_1 = ' + str(np.round(logs['val_Seg_CE_1'], self.num_decimals_to_round))
        if not np.isnan(logs['Seg_Dice_2']):
            train_summary_str = train_summary_str + '\tSeg_Dice_2 = ' + str(np.round(logs['Seg_Dice_2'], self.num_decimals_to_round))
            val_summary_str = val_summary_str + '\tSeg_Dice_2 = ' + str(np.round(logs['val_Seg_Dice_2'], self.num_decimals_to_round))
        if not np.isnan(logs['Seg_CE_2']):
            train_summary_str = train_summary_str + '\tSeg_CE_2 = ' + str(np.round(logs['Seg_CE_2'], self.num_decimals_to_round))
            val_summary_str = val_summary_str + '\tSeg_CE_2 = ' + str(np.round(logs['val_Seg_CE_2'], self.num_decimals_to_round))
        with open(self.output_file, 'a') as f:
            f.write(epoch_summary_str + '\n')
            f.write(train_summary_str + '\n')
            f.write(val_summary_str + '\n')
    #
    def save_my_model(self):
        #save model weights using tensorflow checkpoint format
        self.manager.save(checkpoint_number=self.epoch_save)
        with open(self.output_file, 'a') as f:
            f.write('Model saved with validation loss of ' + str(self.loss_value) + '!\n')
        tf.print('Model saved with validation loss of ' + str(self.loss_value) + '!\n')
    #
    def plot_images(self):
        if train_32 == True:
            affine_params_32, affine_matrix_32, moved_affine_32, moved_label_affine_32, predicted_deformation_field_32, moved_deformable_32, moved_label_deformable_32 = self.model.size_32_registration_network(tf.concat([self.fixed, self.moving, self.moving_label], axis=-1), training=False)
            pred_seg_32_final_1 = self.model.size_32_segmentation_network_1(self.fixed, training=False)
            pred_seg_32_final_2 = self.model.size_32_segmentation_network_2(tf.concat([self.fixed, pred_seg_32_final_1, moved_deformable_32, moved_label_deformable_32], axis=-1), training=False)
            moved_affine = moved_affine_32
            moved_label_affine = np.round(moved_label_affine_32)
            moved_deformable = moved_deformable_32
            moved_label_deformable = np.round(moved_label_deformable_32)
            predicted_deformation_field = predicted_deformation_field_32
            pred_seg_final_1 = pred_seg_32_final_1
            pred_seg_final_2 = pred_seg_32_final_2
        elif train_64 == True:
            affine_params_32, affine_params_64, affine_matrix_32, affine_matrix_64, moved_affine_32, moved_affine_64, moved_label_affine_32, moved_label_affine_64, predicted_deformation_field_32, predicted_deformation_field_64, moved_deformable_32, moved_deformable_64, moved_label_deformable_32, moved_label_deformable_64 = self.model.size_64_registration_network(tf.concat([self.fixed, self.moving, self.moving_label], axis=-1), training=False)
            pred_seg_64_final_1 = self.model.size_64_segmentation_network_1(self.fixed, training=False)
            pred_seg_64_final_2 = self.model.size_64_segmentation_network_2(tf.concat([self.fixed, pred_seg_64_final_1, moved_deformable_64, moved_label_deformable_64], axis=-1), training=False)
            moved_affine = moved_affine_64
            moved_label_affine = np.round(moved_label_affine_64)
            moved_deformable = moved_deformable_64
            moved_label_deformable = np.round(moved_label_deformable_64)
            predicted_deformation_field = predicted_deformation_field_64
            pred_seg_final_1 = pred_seg_64_final_1
            pred_seg_final_2 = pred_seg_64_final_2
        elif train_128 == True:
            affine_params_32, affine_params_64, affine_params_128, affine_matrix_32, affine_matrix_64, affine_matrix_128, moved_affine_32, moved_affine_64, moved_affine_128, moved_label_affine_32, moved_label_affine_64, moved_label_affine_128, predicted_deformation_field_32, predicted_deformation_field_64, predicted_deformation_field_128, moved_deformable_32, moved_deformable_64, moved_deformable_128, moved_label_deformable_32, moved_label_deformable_64, moved_label_deformable_128 = self.model.size_128_registration_network(tf.concat([self.fixed, self.moving, self.moving_label], axis=-1), training=False)
            pred_seg_128_final_1 = self.model.size_128_segmentation_network_1(self.fixed, training=False)
            pred_seg_128_final_2 = self.model.size_128_segmentation_network_2(tf.concat([self.fixed, pred_seg_128_final_1, moved_deformable_128, moved_label_deformable_128], axis=-1), training=False)
            moved_affine = moved_affine_128
            moved_label_affine = np.round(moved_label_affine_128)
            moved_deformable = moved_deformable_128
            moved_label_deformable = np.round(moved_label_deformable_128)
            predicted_deformation_field = predicted_deformation_field_128
            pred_seg_final_1 = pred_seg_128_final_1
            pred_seg_final_2 = pred_seg_128_final_2
        #
        pred_seg_final_1 = np.round(1 / (1 + np.exp(np.clip(-pred_seg_final_1, -self.logits_clip_value, self.logits_clip_value))))
        pred_seg_final_2 = np.round(1 / (1 + np.exp(np.clip(-pred_seg_final_2, -self.logits_clip_value, self.logits_clip_value))))
        #save all images out
        nib.save(nib.Nifti1Image(np.squeeze(self.fixed[self.index,...,0]), affine=np.eye(4)), params_dict['scratch_dir'] + self.image_names[0])
        nib.save(nib.Nifti1Image(np.squeeze(self.fixed_label[self.index,...,0]), affine=np.eye(4)), params_dict['scratch_dir'] + self.image_names[1])
        nib.save(nib.Nifti1Image(np.squeeze(self.moving[self.index,...,0]), affine=np.eye(4)), params_dict['scratch_dir'] + self.image_names[2])
        nib.save(nib.Nifti1Image(np.squeeze(self.moving_label[self.index,...,0]), affine=np.eye(4)), params_dict['scratch_dir'] + self.image_names[3])
        nib.save(nib.Nifti1Image(np.squeeze(moved_affine[self.index,...,0]).astype(float), affine=np.eye(4)), params_dict['scratch_dir'] + self.image_names[4])
        nib.save(nib.Nifti1Image(np.squeeze(moved_label_affine[self.index,...,0]).astype(float), affine=np.eye(4)), params_dict['scratch_dir'] + self.image_names[5])
        nib.save(nib.Nifti1Image(np.squeeze(moved_deformable[self.index,...,0]).astype(float), affine=np.eye(4)), params_dict['scratch_dir'] + self.image_names[6])
        nib.save(nib.Nifti1Image(np.squeeze(moved_label_deformable[self.index,...,0]).astype(float), affine=np.eye(4)), params_dict['scratch_dir'] + self.image_names[7])
        nib.save(nib.Nifti1Image(np.squeeze(predicted_deformation_field[self.index,...]).astype(float), affine=np.eye(4)), params_dict['scratch_dir'] + self.image_names[8])
        nib.save(nib.Nifti1Image(np.squeeze(pred_seg_final_1[self.index,...,0]).astype(float), affine=np.eye(4)), params_dict['scratch_dir'] + self.image_names[9])
        nib.save(nib.Nifti1Image(np.squeeze(pred_seg_final_2[self.index,...,0]).astype(float), affine=np.eye(4)), params_dict['scratch_dir'] + self.image_names[10])
        #copy images to logs folder
        [shutil.copyfile(params_dict['scratch_dir'] + self.image_names[i], '/home/jay.patel/' + self.image_names[i]) for i in range(0, len(self.image_names))]
        #
        fixed = self.fixed[:self.num_images_to_plot,...].numpy()
        moved_deformable = moved_deformable[:self.num_images_to_plot,...].numpy()
        moved_affine = moved_affine[:self.num_images_to_plot,...].numpy()
        moving = self.moving[:self.num_images_to_plot,...].numpy()
        titles_list = ['Fixed', 'Deformable', 'Affine', 'Moving']
        images_list = [fixed, moved_deformable, moved_affine, moving]
        all_images_array = np.squeeze(np.stack(images_list, axis=0), axis=-1)
        # Convert the tensors to 8-bit images
        for i in range(0, all_images_array.shape[0]): #four images
            for j in range(0, all_images_array.shape[1]): #batch dimension
                for k in range(0, all_images_array.shape[-1]): #z dimension
                    all_images_array[i,j,...,k] = 255.0 * ((all_images_array[i,j,...,k] - np.min(all_images_array[i,j,...,k])) / (np.max(all_images_array[i,j,...,k]) - np.min(all_images_array[i,j,...,k]) + 1e-7))
        all_images_array = all_images_array.astype(np.uint8)
        #
        # Plot images
        fig = plt.figure(figsize=(len(titles_list) * 2.0, self.num_images_to_plot * 2.0))
        slice_to_plot = int(all_images_array.shape[-1] / 2)
        num_images_per_row = len(images_list)
        for i in range(self.num_images_to_plot):
            for j in range(num_images_per_row):
                ax = fig.add_subplot(self.num_images_to_plot, num_images_per_row, i * num_images_per_row + j + 1)
                if i == 0:
                    ax.set_title(titles_list[j], fontsize=20)
                ax.set_axis_off()
                ax.imshow(all_images_array[j,i,...,slice_to_plot], cmap='gray')
        #
        plt.tight_layout()
        plt.savefig('/home/jay.patel/' + self.final_png_name)
        plt.close('all')
        #
        # Plot flow field
        predicted_deformation_field = predicted_deformation_field.numpy()
        ne.plot.flow([predicted_deformation_field[self.index,:,:,slice_to_plot,:2]], width=5)[0].savefig('/home/jay.patel/' + self.final_flow_png_name)
        plt.close('all')

#transforms for model pdf image
def get_transforms(temp_params_dict):
    transforms = [
        # prune graph to make it easier to read
        ht.Prune("Cast"),
        ht.PruneBranch("Add > Rsqrt"),
        ht.PruneBranch("Mul > Sub"),
        # Fold remaining parts of BN into one block
        ht.Fold("Mul > Add", "Norm", "Norm"),
        # Fold Upsample into one block
        ht.Fold("((Shape > Mul) | (ExpandDims > Tile)) > Reshape > ((Shape > Mul) | (ExpandDims > Tile)) > Reshape > ((Shape > Mul) | (ExpandDims > Tile)) > Reshape > Pad", "Upsample0"),
        ht.Fold("Split > Concat > Split > Concat > Split > Concat > Pad", "Upsample0"),
        ht.Fold("Upsample0 > Conv", "Upsample1", "Upsample"),
        ht.Fold("Conv > Upsample1", "Upsample2", "Conv Upsample"),
        ht.Fold("Upsample2 > Norm > Relu", "Upsample3", "Conv Upsample Norm Act"),
        ht.Fold("Upsample1 > Norm > Relu", "Upsample4", "Upsample Norm Act"),
        #Fold Deconvolution into one block
        ht.Fold("StridedSlice > Mul", "Deconv0"),
        ht.Fold("(Deconv0 | StridedSlice | Deconv0 | Deconv0) > Pack", "Deconv1"),
        ht.Fold("Deconv0 > Pack", "Deconv_1"),
        ht.Fold("Deconv0 > Deconv_1", "Deconv_2"),
        ht.Fold("Deconv0 > Deconv_2", "Deconv_3"),
        ht.Fold("StridedSlice > Deconv_3", "Deconv_4"),
        ht.Fold("Shape > Deconv1", "Deconv2"),
        ht.Fold("Deconv2 > ConvBackpropInput", "Deconv3", "Deconvolution"),
        ht.Fold("Deconv_4 > ConvBackpropInput", "Deconv3", "Deconvolution")]
    # Fold Conv, BN, and Relu together if they come together
    if temp_params_dict['preactivation'][0] == False:
        temp_transform = [
            ht.Fold("Conv > Norm > Relu", "ConvNormRelu", "Conv Norm Act"),
            ht.Fold("Deconv3 > Norm > Relu", "DeConvNormRelu", "DeConv Norm Act")]
        transforms.extend(temp_transform)
    else:
        temp_transform = [
            ht.Fold("Norm > Relu > Conv", "NormReluConv",  "Norm Act Conv"),
            ht.Fold("Norm > Relu > Deconv3", "NormReluDeConv",  "Norm Act DeConv"),
            ht.Fold("Norm > Relu", "NormRelu",  "Norm Act"),
            ht.Fold("NormRelu > MaxPool > Conv", "NormReluMaxPoolConv", "Norm Act MaxPool Conv"),
            ht.Fold("NormRelu > Upsample2", "NormReluConvUpsample", "Norm Act Conv Upsample")]
        transforms.extend(temp_transform)
    #Fold basic residual block
    if temp_params_dict['block_type'] == 'Residual_Block':
        temp_transform = [
            ht.Fold("((ConvNormRelu > Conv > Norm) | (Conv > Norm)) > Add > Relu", "BasicResidualBlockWithShortcut",  "Basic Residual Block With Shortcut"),
            ht.Fold("(ConvNormRelu > Conv > Norm) > Add > Relu", "BasicResidualBlock",  "Basic Residual Block")]
        transforms.extend(temp_transform)
    #Fold bottleneck residual block
    elif temp_params_dict['block_type'] == 'Residual_Bottleneck_Block':
        temp_transform = [
            ht.Fold("((ConvNormRelu > ConvNormRelu > Conv > Norm) | (Conv > Norm)) > Add > Relu", "ResidualBottleneckBlockWithShortcut",  "Residual Bottleneck Block With Shortcut"),
            ht.Fold("(ConvNormRelu > ConvNormRelu > Conv > Norm) > Add > Relu", "ResidualBottleneckBlock",  "Residual Bottleneck Block")]
        transforms.extend(temp_transform)
    #Fold dense block
    elif temp_params_dict['block_type'] == 'Dense_Block':
        if temp_params_dict['preactivation'][0] == False:
            temp_transform = [
                ht.Fold("ConvNormRelu > ConvNormRelu > Concat", "DenseBottleneckBlock",  "Dense Bottleneck Block"),
                ht.Fold("ConvNormRelu > Concat", "DenseBlock",  "Basic Dense Block")]
            transforms.extend(temp_transform)
        else:
            temp_transform = [
                ht.Fold("NormReluConv > NormReluConv > Concat", "PreActDenseBottleneckBlock",  "Preact Dense Bottleneck Block"),
                ht.Fold("NormReluConv > Concat", "PreActDenseBlock",  "Preact Basic Dense Block")]
            transforms.extend(temp_transform)
    #Dropblock transforms
    if np.any(temp_params_dict['dropblock_params']):
        temp_transform = [
            ht.PruneBranch("Mul > Mul > Less"),
            ht.PruneBranch("Prod > ExpandDims > ExpandDims > Sub > Add > Add > GreaterEqual"),
            ht.PruneBranch("Neg > MaxPool > Neg > Mean"),
            ht.PruneBranch("Neg > MaxPool > Neg"),
            ht.Fold("Mean > DivNoNan > Mul", "DropBlock0", "DropBlock"),
            ht.Fold("Min > DropBlock0", "DropBlock1", "DropBlock")]
    # Fold repeated nodes
    transforms.append(ht.FoldDuplicates())
    return transforms

#make model pdf image
def save_model_image(params_dict):
    temp_params_dict = deepcopy(params_dict)
    #replace normalization with BN to simplify model pruning
    temp_params_dict['normalization'] = ['BatchNormalization', False] 
    #replace activation with ReLU to simplify model pruning
    temp_params_dict['activation'] = ['ReLU']
    #replace mixed_precision variables to simplify model pruning
    temp_params_dict['enable_xla'] = False
    temp_params_dict['mixed_precision'] = False
    temp_params_dict['custom_fit'][0] = False
    #ensure input placeholder does not use None values to get tensor shapes printed
    temp_params_dict['adaptive_full_image_patching'] = [False, False]
    with tf.compat.v1.Session() as sess:
        with tf.Graph().as_default() as tf_graph:
            #create model
            batch_size = 1
            inputs = tf.keras.Input(shape=(32,32,32,3), batch_size=batch_size, name='fixed_moving_32', dtype=tf.float16 if params_dict['mixed_precision'] == True else tf.float32)
            outputs = total_size_32_model(batch_size, params_dict['mixed_precision'], False)(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            # Build HiddenLayer graph
            hl_graph = hl.build_graph(tf_graph, transforms=get_transforms(temp_params_dict))
    hl_graph.save('/home/jay.patel/model_image')

#take vector of affine params and turn into affine matrix
def make_affine_matrix_from_params(affine_params, ndims=3):
    if ndims == 2:
        affine_transform_size = 2
        translate_factors = affine_params[:, 0:2]
        rotate_factors = affine_params[:, 2:3]
        scale_factors = affine_params[:, 3:5]
        shear_factors = affine_params[:, 5:7]
    else:
        affine_transform_size = 3
        translate_factors = affine_params[:, 0:3]
        rotate_factors = affine_params[:, 3:6]
        scale_factors = affine_params[:, 6:9]
        shear_factors = affine_params[:, 9:15]
    #generate rotation transform
    trig_values_cosine = tf.split(tf.cos(rotate_factors), 1 if ndims==2 else 3, axis=-1)
    trig_values_sine = tf.split(tf.sin(rotate_factors), 1 if ndims==2 else 3, axis=-1)
    if affine_transform_size == 2:
        rotation_matrix = tf.stack((
            tf.concat([trig_values_cosine[0], -trig_values_sine[0]], axis=-1), 
            tf.concat([trig_values_sine[0], trig_values_cosine[0]], axis=-1)
            ), axis=1)
    else:
        zero_float_precision = tf.zeros_like(trig_values_cosine[0])
        one_float_precision = tf.ones_like(trig_values_cosine[0])
        rotation_x = tf.stack((
            tf.concat([one_float_precision, zero_float_precision, zero_float_precision], axis=-1), 
            tf.concat([zero_float_precision, trig_values_cosine[0], -trig_values_sine[0]], axis=-1), 
            tf.concat([zero_float_precision, trig_values_sine[0], trig_values_cosine[0]], axis=-1)
            ), axis=1)
        rotation_y = tf.stack((
            tf.concat([trig_values_cosine[1], zero_float_precision, trig_values_sine[1]], axis=-1), 
            tf.concat([zero_float_precision, one_float_precision, zero_float_precision], axis=-1), 
            tf.concat([-trig_values_sine[1], zero_float_precision, trig_values_cosine[1]], axis=-1)
            ), axis=1)
        rotation_z = tf.stack((
            tf.concat([trig_values_cosine[2], -trig_values_sine[2], zero_float_precision], axis=-1), 
            tf.concat([trig_values_sine[2], trig_values_cosine[2], zero_float_precision], axis=-1), 
            tf.concat([zero_float_precision, zero_float_precision, one_float_precision], axis=-1)
            ), axis=1)
        rotation_matrix = tf.matmul(tf.matmul(rotation_x, rotation_y), rotation_z)
    #generate scale transform
    scale_matrix = tf.linalg.diag(scale_factors)
    #generate shear transform
    shear_values = tf.split(shear_factors, 2 if ndims==2 else 6, axis=-1)
    zero_float_precision = tf.zeros_like(shear_values[0])
    one_float_precision = tf.ones_like(shear_values[0])
    if affine_transform_size == 2:
        shear_x = tf.stack((
            tf.concat([one_float_precision, shear_values[0]], axis=-1), 
            tf.concat([zero_float_precision, one_float_precision], axis=-1)
            ), axis=1)
        shear_y = tf.stack((
            tf.concat([one_float_precision, zero_float_precision], axis=-1), 
            tf.concat([shear_values[1], one_float_precision], axis=-1)
            ), axis=1)
        shear_matrix = tf.matmul(shear_x, shear_y)
    else:
        shear_x = tf.stack((
            tf.concat([one_float_precision, shear_values[0], shear_values[1]], axis=-1), 
            tf.concat([zero_float_precision, one_float_precision, zero_float_precision], axis=-1), 
            tf.concat([zero_float_precision, zero_float_precision, one_float_precision], axis=-1)
            ), axis=1)
        shear_y = tf.stack((
            tf.concat([one_float_precision, zero_float_precision, zero_float_precision], axis=-1), 
            tf.concat([shear_values[2], one_float_precision, shear_values[3]], axis=-1), 
            tf.concat([zero_float_precision, zero_float_precision, one_float_precision], axis=-1)
            ), axis=1)
        shear_z = tf.stack((
            tf.concat([one_float_precision, zero_float_precision, zero_float_precision], axis=-1), 
            tf.concat([zero_float_precision, one_float_precision, zero_float_precision], axis=-1), 
            tf.concat([shear_values[4], shear_values[5], one_float_precision], axis=-1)
            ), axis=1)
        shear_matrix = tf.matmul(tf.matmul(shear_x, shear_y), shear_z)
    #compose transforms together
    affine_transform_matrix = tf.matmul(tf.matmul(rotation_matrix, scale_matrix), shear_matrix)
    #concatenate translate factors
    affine_transform_matrix = tf.concat([affine_transform_matrix, tf.expand_dims(translate_factors, axis=-1)], axis=-1)
    return affine_transform_matrix

#linear/nearest neighbor interpolation of image via tensorflow operations (affine transform is performed before the deformable transform)
def transform_image(affine_transform, deformable_transform, moving, input_shape, batch_size, ndims, order=1, mode='constant', mixed_precision=False):
    #affine_transform is size (batch_size, ndims, ndims+1)
    #deformable_transform is size (batch_size, *input_shape, ndims)
    #moving is moving volume of size (batch_size, *input_shape, n), where n is usually 1 (for volumes) or 3 (for deformation fields) 
    #input_shape is the size (height, width, depth) of this volume as a list
    #batch_size is the number of images in this batch
    #ndims is the dimension (must be 2 or 3)
    #order is the order of the interpolation (either 1 for linear or 0 for nearest neighbor)
    #mode is how to handle edge cases: 'constant' will apply zeros outside image boundary; 'nearest' will extend the boundary in that direction with nearest pixel value
    #mixed precision is for using 16 bit precision
    #
    do_affine_transform = False
    do_deformable_transform = False
    if affine_transform is not None:
        do_affine_transform = True
    if deformable_transform is not None:
        do_deformable_transform = True
    #
    if mixed_precision == True:
        zero_float_precision = tf.constant(0, tf.float16)
    else:
        zero_float_precision = tf.constant(0, tf.float32)
    if mode != 'constant':
        zero_int_precision = tf.constant(0, tf.int32)
    # Make meshgrid using 'ij' indexing and shift to center
    x = tf.linspace(zero_float_precision, input_shape[0] - 1., input_shape[0])
    y = tf.linspace(zero_float_precision, input_shape[1] - 1., input_shape[1])
    if ndims == 2:
        X, Y = tf.meshgrid(x, y, indexing='ij')
        X, Y = [mesh - (shape / 2) for (mesh, shape) in zip((X,Y), input_shape)]
    else:
        z = tf.linspace(zero_float_precision, input_shape[2] - 1., input_shape[2])
        X, Y, Z = tf.meshgrid(x, y, z, indexing='ij')
        X, Y, Z = [mesh - (shape / 2) for (mesh, shape) in zip((X,Y,Z), input_shape)]
    # Make flattened mesh matrix
    x_flat = tf.reshape(X, shape=[-1])
    y_flat = tf.reshape(Y, shape=[-1])
    #
    ones = tf.ones_like(x_flat)
    if ndims == 2:
        grid_flat = tf.stack([x_flat, y_flat, ones])
    else:
        z_flat = tf.reshape(Z, shape=[-1])
        grid_flat = tf.stack([x_flat, y_flat, z_flat, ones])
    grid_flat = tf.expand_dims(grid_flat, axis=0)
    grid_flat = tf.tile(grid_flat, tf.stack([batch_size, 1, 1]))
    # Apply affine transform to mesh matrix (if requested)
    if do_affine_transform == True:
        grid_new = tf.matmul(affine_transform, grid_flat)
    else:
        grid_new = grid_flat[:,:-1,:]
    grid_new = tf.transpose(grid_new, perm=[0,2,1])
    # Reshape back to original image size
    if ndims == 2:
        grid_new = tf.reshape(grid_new, [batch_size, input_shape[0], input_shape[1], 2])
    else:
        grid_new = tf.reshape(grid_new, [batch_size, input_shape[0], input_shape[1], input_shape[2], 3])
    # Apply deformable transform to mesh matrix (if requested)
    if do_deformable_transform == True:
        grid_new = grid_new + deformable_transform
    # Unshift grid
    grid_new = tf.stack([grid_new[..., dim_index] + (input_shape[dim_index] / 2) for dim_index in range(0, ndims)], axis=-1)
    # Clip values outside boundary
    grid_new = tf.stack([tf.clip_by_value(grid_new[..., dim_index], 0, input_shape[dim_index]) for dim_index in range(0, ndims)], axis=-1)
    #
    x_rescale = grid_new[..., 0]
    y_rescale = grid_new[..., 1]
    if ndims == 3:
        z_rescale = grid_new[..., 2]
    #
    # The value at (x, y) is a weighted average of the values at the four nearest integer locations for 2D images (eight neighbors in 3D)
    x0_f = tf.floor(x_rescale)
    x1_f = x0_f + 1.
    y0_f = tf.floor(y_rescale)
    y1_f = y0_f + 1.
    if ndims == 3:
        z0_f = tf.floor(z_rescale)
        z1_f = z0_f + 1.
    # GPU will automatically impute 0 for values outside grid range, so no need for clipping
    if mode == 'constant':
        x0 = tf.cast(x0_f, tf.int32)
        x1 = tf.cast(x1_f, tf.int32)
        y0 = tf.cast(y0_f, tf.int32)
        y1 = tf.cast(y1_f, tf.int32)
        if ndims == 3:
            z0 = tf.cast(z0_f, tf.int32)
            z1 = tf.cast(z1_f, tf.int32)
    # If want nearest neighbor interpolation at edges, can apply clipping to grid size
    else:
        x_max = tf.cast(input_shape[0] - 1, tf.int32)
        y_max = tf.cast(input_shape[1] - 1, tf.int32)
        if ndims == 3:
            z_max = tf.cast(input_shape[2] - 1, tf.int32)
        x0 = tf.clip_by_value(x0, zero_int_precision, x_max)
        x1 = tf.clip_by_value(x1, zero_int_precision, x_max)
        y0 = tf.clip_by_value(y0, zero_int_precision, y_max)
        y1 = tf.clip_by_value(y1, zero_int_precision, y_max)
        if ndims == 3:
            z0 = tf.clip_by_value(z0, zero_int_precision, z_max)
            z1 = tf.clip_by_value(z1, zero_int_precision, z_max)
    # Collect indices of the corners
    if ndims == 2:
        b = tf.ones_like(x0) * tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
        idx_a = tf.stack([b, x0, y0], axis=-1)
        idx_b = tf.stack([b, x1, y0], axis=-1)
        idx_c = tf.stack([b, x0, y1], axis=-1)
        idx_d = tf.stack([b, x1, y1], axis=-1)
    else:
        b = tf.ones_like(x0) * tf.reshape(tf.range(batch_size), [batch_size, 1, 1, 1])
        idx_a = tf.stack([b, x0, y0, z0], axis=-1)
        idx_b = tf.stack([b, x1, y0, z0], axis=-1)
        idx_c = tf.stack([b, x0, y1, z0], axis=-1)
        idx_d = tf.stack([b, x1, y1, z0], axis=-1)
        idx_e = tf.stack([b, x0, y0, z1], axis=-1)
        idx_f = tf.stack([b, x1, y0, z1], axis=-1)
        idx_g = tf.stack([b, x0, y1, z1], axis=-1)
        idx_h = tf.stack([b, x1, y1, z1], axis=-1)
    # Collect values at the corners
    moving_a = tf.gather_nd(moving, idx_a)
    moving_b = tf.gather_nd(moving, idx_b)
    moving_c = tf.gather_nd(moving, idx_c)
    moving_d = tf.gather_nd(moving, idx_d)
    if ndims == 3:
        moving_e = tf.gather_nd(moving, idx_e)
        moving_f = tf.gather_nd(moving, idx_f)
        moving_g = tf.gather_nd(moving, idx_g)
        moving_h = tf.gather_nd(moving, idx_h)
    # Final interpolation (either bi/tri-linear or nearest neighbor)
    if order == 1:
        # Calculate the weights
        if ndims == 2:
            wa = tf.expand_dims((x1_f - x_rescale) * (y1_f - y_rescale), axis=-1)
            wb = tf.expand_dims((x_rescale - x0_f) * (y1_f - y_rescale), axis=-1)
            wc = tf.expand_dims((x1_f - x_rescale) * (y_rescale - y0_f), axis=-1)
            wd = tf.expand_dims((x_rescale - x0_f) * (y_rescale - y0_f), axis=-1)
        else:
            wa = tf.expand_dims((x1_f - x_rescale) * (y1_f - y_rescale) * (z1_f - z_rescale), axis=-1)
            wb = tf.expand_dims((x_rescale - x0_f) * (y1_f - y_rescale) * (z1_f - z_rescale), axis=-1)
            wc = tf.expand_dims((x1_f - x_rescale) * (y_rescale - y0_f) * (z1_f - z_rescale), axis=-1)
            wd = tf.expand_dims((x_rescale - x0_f) * (y_rescale - y0_f) * (z1_f - z_rescale), axis=-1)
            we = tf.expand_dims((x1_f - x_rescale) * (y1_f - y_rescale) * (z_rescale - z0_f), axis=-1)
            wf = tf.expand_dims((x_rescale - x0_f) * (y1_f - y_rescale) * (z_rescale - z0_f), axis=-1)
            wg = tf.expand_dims((x1_f - x_rescale) * (y_rescale - y0_f) * (z_rescale - z0_f), axis=-1)
            wh = tf.expand_dims((x_rescale - x0_f) * (y_rescale - y0_f) * (z_rescale - z0_f), axis=-1)
        # Calculate the weighted sum
        if ndims == 2:
            moved = tf.add_n([
                wa * moving_a,
                wb * moving_b,
                wc * moving_c,
                wd * moving_d
                ])
        else:
            moved = tf.add_n([
                wa * moving_a,
                wb * moving_b,
                wc * moving_c,
                wd * moving_d,
                we * moving_e,
                wf * moving_f,
                wg * moving_g,
                wh * moving_h
                ])
    else:
        # Calculate the weights
        # weights are "opposite" of what they were for linear interpolation
        if ndims == 2:
            wd = tf.expand_dims(((x1_f - x_rescale) ** 2) + ((y1_f - y_rescale) ** 2), axis=-1)
            wc = tf.expand_dims(((x_rescale - x0_f) ** 2) + ((y1_f - y_rescale) ** 2), axis=-1)
            wb = tf.expand_dims(((x1_f - x_rescale) ** 2) + ((y_rescale - y0_f) ** 2), axis=-1)
            wa = tf.expand_dims(((x_rescale - x0_f) ** 2) + ((y_rescale - y0_f) ** 2), axis=-1)
            #find nearest neighbor
            nearest_neighbor = tf.one_hot(tf.math.argmin(tf.concat([wa,wb,wc,wd], axis=-1), axis=-1), depth=4)
        else:
            wh = tf.expand_dims(((x1_f - x_rescale) ** 2) + ((y1_f - y_rescale) ** 2) + ((z1_f - z_rescale) ** 2), axis=-1)
            wg = tf.expand_dims(((x_rescale - x0_f) ** 2) + ((y1_f - y_rescale) ** 2) + ((z1_f - z_rescale) ** 2), axis=-1)
            wf = tf.expand_dims(((x1_f - x_rescale) ** 2) + ((y_rescale - y0_f) ** 2) + ((z1_f - z_rescale) ** 2), axis=-1)
            we = tf.expand_dims(((x_rescale - x0_f) ** 2) + ((y_rescale - y0_f) ** 2) + ((z1_f - z_rescale) ** 2), axis=-1)
            wd = tf.expand_dims(((x1_f - x_rescale) ** 2) + ((y1_f - y_rescale) ** 2) + ((z_rescale - z0_f) ** 2), axis=-1)
            wc = tf.expand_dims(((x_rescale - x0_f) ** 2) + ((y1_f - y_rescale) ** 2) + ((z_rescale - z0_f) ** 2), axis=-1)
            wb = tf.expand_dims(((x1_f - x_rescale) ** 2) + ((y_rescale - y0_f) ** 2) + ((z_rescale - z0_f) ** 2), axis=-1)
            wa = tf.expand_dims(((x_rescale - x0_f) ** 2) + ((y_rescale - y0_f) ** 2) + ((z_rescale - z0_f) ** 2), axis=-1)
            nearest_neighbor = tf.one_hot(tf.math.argmin(tf.concat([wa,wb,wc,wd,we,wf,wg,wh], axis=-1), axis=-1), depth=8)
        # Use nearest neighbor to get moved image
        if ndims == 2:
            moved = tf.add_n([
                tf.expand_dims(nearest_neighbor[...,0], axis=-1) * moving_a,
                tf.expand_dims(nearest_neighbor[...,1], axis=-1) * moving_b,
                tf.expand_dims(nearest_neighbor[...,2], axis=-1) * moving_c,
                tf.expand_dims(nearest_neighbor[...,3], axis=-1) * moving_d
                ])
        else:
            moved = tf.add_n([
                tf.expand_dims(nearest_neighbor[...,0], axis=-1) * moving_a,
                tf.expand_dims(nearest_neighbor[...,1], axis=-1) * moving_b,
                tf.expand_dims(nearest_neighbor[...,2], axis=-1) * moving_c,
                tf.expand_dims(nearest_neighbor[...,3], axis=-1) * moving_d,
                tf.expand_dims(nearest_neighbor[...,4], axis=-1) * moving_e,
                tf.expand_dims(nearest_neighbor[...,5], axis=-1) * moving_f,
                tf.expand_dims(nearest_neighbor[...,6], axis=-1) * moving_g,
                tf.expand_dims(nearest_neighbor[...,7], axis=-1) * moving_h
                ])
    #
    return moved

#linear/nearest neighbor interpolation of image via numpy operations (affine transform is performed before the deformable transform)
def transform_image_numpy(affine_transform, deformable_transform, moving, input_shape, batch_size, ndims, order=3, mode='constant'):
    #affine_transform is size (batch_size, ndims, ndims+1)
    #deformable_transform is size (batch_size, *input_shape, ndims)
    #moving is moving volume of size (batch_size, *input_shape, n), where n is usually 1 (for volumes) or 3 (for deformation fields) 
    #input_shape is the size (height, width, depth) of this volume as a list
    #batch_size is the number of images in this batch
    #ndims is the dimension (must be 2 or 3)
    #order is the order of the interpolation (3 for cubic, 1 for linear, 0 for nearest neighbor)
    #mode is how to handle edge cases: 'constant' will apply zeros outside image boundary; 'nearest' will extend the boundary in that direction with nearest pixel value
    #
    do_affine_transform = False
    do_deformable_transform = False
    if affine_transform is not None:
        do_affine_transform = True
    if deformable_transform is not None:
        do_deformable_transform = True
    #
    # Make meshgrid using 'ij' indexing and shift to center
    x = np.linspace(0.0, input_shape[0] - 1, input_shape[0])
    y = np.linspace(0.0, input_shape[1] - 1, input_shape[1])
    if ndims == 2:
        X, Y = np.meshgrid(x, y, indexing='ij')
        X, Y = [mesh - (shape / 2) for (mesh, shape) in zip((X,Y), input_shape)]
    else:
        z = np.linspace(0.0, input_shape[2] - 1, input_shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        X, Y, Z = [mesh - (shape / 2) for (mesh, shape) in zip((X,Y,Z), input_shape)]
    # Make flattened mesh matrix
    x_flat = np.reshape(X, [-1])
    y_flat = np.reshape(Y, [-1])
    #
    ones = np.ones(x_flat.shape)
    if ndims == 2:
        grid_flat = np.stack([x_flat, y_flat, ones])
    else:
        z_flat = np.reshape(Z, [-1])
        grid_flat = np.stack([x_flat, y_flat, z_flat, ones])
    grid_flat = np.expand_dims(grid_flat, axis=0)
    grid_flat = np.tile(grid_flat, np.stack([batch_size, 1, 1]))
    # Apply affine transform to mesh matrix (if requested)
    if do_affine_transform == True:
        grid_new = np.matmul(affine_transform, grid_flat)
    else:
        grid_new = grid_flat[:,:-1,:]
    grid_new = np.transpose(grid_new, [0,2,1])
    # Reshape back to original image size
    if ndims == 2:
        grid_new = np.reshape(grid_new, [batch_size, input_shape[0], input_shape[1], 2])
    else:
        grid_new = np.reshape(grid_new, [batch_size, input_shape[0], input_shape[1], input_shape[2], 3])
    # Apply deformable transform to mesh matrix (if requested)
    if do_deformable_transform == True:
        grid_new = grid_new + deformable_transform
    # Unshift grid
    grid_new = np.stack([grid_new[..., dim_index] + (input_shape[dim_index] / 2) for dim_index in range(0, ndims)], axis=-1)
    # transpose meshgrid to match scipy function
    if ndims == 2:
        grid_new = np.transpose(grid_new, [0,-1,1,2])
    else:
        grid_new = np.transpose(grid_new, [0,-1,1,2,3])
    # resample image at requested order
    moved = np.zeros((moving.shape))
    for i in range(0, batch_size):
        moved[i,...,0] = map_coordinates(moving[i,...,0], grid_new[i,...], order=order, mode=mode, output=np.float32)
    return moved

#compose multiple affine matrices together
def compose_affine_transforms(transform1, transform2, batch_size, mixed_precision):
    temp_transform1 = tf.concat([transform1, tf.tile(tf.expand_dims(tf.constant([[0.0]*3 + [1.0]], tf.float16 if mixed_precision else tf.float32), axis=0), [batch_size,1,1])], axis=1)
    temp_transform2 = tf.concat([transform2, tf.tile(tf.expand_dims(tf.constant([[0.0]*3 + [1.0]], tf.float16 if mixed_precision else tf.float32), axis=0), [batch_size,1,1])], axis=1)
    final_transform = tf.linalg.matmul(temp_transform1, temp_transform2)[:, :3, :]
    return final_transform

#helper function to make normalization layer
def instantiate_normalization_layer(num_filters, prior_filter_num, epsilon):
    if prior_filter_num is None:
        prior_filter_num = num_filters
    if params_dict['normalization'][0] == 'BatchNormalization':
        if params_dict['normalization'][1] == True:
            norm_operation = tf.keras.layers.BatchNormalization(scale=False, center=True, epsilon=epsilon)
        else:
            norm_operation = tf.keras.layers.BatchNormalization(scale=False, center=True, trainable=False, epsilon=epsilon)
    elif params_dict['normalization'][0] == 'GroupNormalization':
        norm_operation = tfa_layers.GroupNormalization(groups=max(prior_filter_num // params_dict['normalization'][1], params_dict['normalization'][2]), scale=False, center=True, epsilon=epsilon)
    return norm_operation

#helper function to apply normalization layers
def apply_normalization(x, normalization_operation, training):
    if params_dict['normalization'][0] == 'BatchNormalization':
        if params_dict['normalization'][1] == False:
            return normalization_operation(x, training=False)
        else:
            return normalization_operation(x, training=training)
    else:
        return normalization_operation(x)

#gaussian blurring filter tensorflow
def gaussian_blur(kernel_size_for_gaussian_conv=7, sigma_for_gaussian_conv=1.):
    ax = tf.range(-kernel_size_for_gaussian_conv // 2 + 1.0, kernel_size_for_gaussian_conv // 2 + 1.0)
    xx, yy, zz = tf.meshgrid(ax, ax, ax)
    gaussian_kernel = tf.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2.0 * sigma_for_gaussian_conv ** 2))
    gaussian_kernel = gaussian_kernel / tf.reduce_sum(gaussian_kernel)
    gaussian_kernel = tf.expand_dims(gaussian_kernel, axis=-1)
    return gaussian_kernel

#gaussian blurring filter numpy
def gaussian_blur_numpy(kernel_size_for_gaussian_conv=7, sigma_for_gaussian_conv=1.):
    ax = np.arange(-kernel_size_for_gaussian_conv // 2 + 1.0, kernel_size_for_gaussian_conv // 2 + 1.0)
    xx, yy, zz = np.meshgrid(ax, ax, ax)
    gaussian_kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2.0 * sigma_for_gaussian_conv ** 2))
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    gaussian_kernel = np.expand_dims(gaussian_kernel, axis=-1)
    return gaussian_kernel

class gaussian_blur_conv(tf.keras.Model):
    def __init__(self, kernel_size_for_gaussian_conv=7, sigma_for_gaussian_conv=1., name='gaussian_blur_conv', **kwargs):
        super(gaussian_blur_conv, self).__init__(name=name, **kwargs)
        #
        gaussian_kernel = gaussian_blur_numpy(kernel_size_for_gaussian_conv=kernel_size_for_gaussian_conv, sigma_for_gaussian_conv=sigma_for_gaussian_conv)
        self.gaussian_conv = tf.keras.layers.Conv3D(filters=1, kernel_size=kernel_size_for_gaussian_conv, strides=1, dilation_rate=1, padding='same', use_bias=False, activation=None, kernel_regularizer=None, kernel_initializer=tf.keras.initializers.Constant(gaussian_kernel), kernel_constraint=None, groups=1, trainable=False)
    #
    def call(self, x, training=True):
        x = self.gaussian_conv(x)
        return x

class upsample_operations(tf.keras.Model):
    def __init__(self, num_filters=32, name='upsample_operations', **kwargs):
        super(upsample_operations, self).__init__(name=name, **kwargs)
        #
        self.upsample_1 = tf.keras.layers.UpSampling3D(size=2)
        self.padding_values = [(0,0)] + [(pad_val // 2, pad_val - (pad_val // 2)) for pad_val in [1,1,1]] + [(0,0)]
        self.upsample_2 = tf.keras.layers.Conv3D(filters=num_filters, kernel_size=2, groups=num_filters, padding='valid', use_bias=False, kernel_initializer=tf.constant_initializer(0.125), trainable=False)
    #
    def call(self, x, training=True):
        x_upsample = self.upsample_1(x)
        x_upsample = tf.pad(x_upsample, self.padding_values)
        x_upsample = self.upsample_2(x_upsample)
        return x_upsample

class norm_act_conv_pool_upsample(tf.keras.Model):
    def __init__(self, num_filters=32, prior_filter_num=None, kernel_size=3, use_norm_act=True, use_pooling=True, use_upsample=True, conv1=False, use_bias=False, kernel_initializer='he_normal', bias_initializer='zeros', name='norm_act_conv_pool_upsample', epsilon=0.00001, **kwargs):
        super(norm_act_conv_pool_upsample, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.use_norm_act = use_norm_act
        self.use_pooling = use_pooling
        self.use_upsample = use_upsample
        self.conv_kernel_size_1 = conv1
        #
        if self.use_norm_act == True:
            self.norm1 = instantiate_normalization_layer(num_filters, prior_filter_num, epsilon)
            self.act1 = tf.keras.layers.Activation('relu')
        self.conv1 = tf.keras.layers.Conv3D(num_filters, kernel_size=kernel_size, strides=1, padding='same', activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        if self.use_pooling == True:
            self.pooling = tf.keras.layers.MaxPool3D(pool_size=2)
        if self.use_upsample == True:
            self.upsample_operations = upsample_operations(num_filters)
        if self.conv_kernel_size_1 == True:
            self.norm2 = instantiate_normalization_layer(num_filters, None, epsilon)
            self.act2 = tf.keras.layers.Activation('relu')
            self.conv2 = tf.keras.layers.Conv3D(num_filters // 2, kernel_size=1, strides=1, padding='same', activation=None, use_bias=False, kernel_initializer=kernel_initializer)
    #
    def call(self, x, training=True):
        if self.use_norm_act == True:
            x = apply_normalization(x, self.norm1, training)
            x = self.act1(x)
        x = self.conv1(x)
        if self.use_pooling == True:
            x_pool = self.pooling(x)
            return x, x_pool
        elif self.use_upsample == True:
            x_upsample = self.upsample_operations(x)
            if self.conv_kernel_size_1 == True:
                x_upsample = apply_normalization(x_upsample, self.norm2, training)
                x_upsample = self.act2(x_upsample)
                x_upsample = self.conv2(x_upsample)
            return x, x_upsample 
        else:
            return x

class encoder_1(tf.keras.Model):
    def __init__(self, num_filters=32, kernel_size=3, name='encoder_1', **kwargs):
        super(encoder_1, self).__init__(name=name, **kwargs)
        #
        if not isinstance(num_filters, list):
            num_filters = [num_filters]*4
        self.level1 = norm_act_conv_pool_upsample(num_filters=num_filters[0], kernel_size=kernel_size, use_pooling=True, use_upsample=False)
        self.level2 = norm_act_conv_pool_upsample(num_filters=num_filters[1], kernel_size=kernel_size, use_pooling=True, use_upsample=False)
        self.level3 = norm_act_conv_pool_upsample(num_filters=num_filters[2], kernel_size=kernel_size, use_pooling=True, use_upsample=False)
        self.level4 = norm_act_conv_pool_upsample(num_filters=num_filters[3], kernel_size=kernel_size, use_pooling=False, use_upsample=False)
    #
    def call(self, x, training=True):
        skip1, x_pool = self.level1(x)
        skip2, x_pool = self.level2(x_pool)
        skip3, x_pool = self.level3(x_pool)
        skip4 = self.level4(x_pool)
        return skip1, skip2, skip3, skip4

class encoder_2(tf.keras.Model):
    def __init__(self, num_filters=32, prior_filter_num=32, num_dense_units=128, ndims=3, epsilon=0.00001, name='encoder_2', **kwargs):
        super(encoder_2, self).__init__(name=name, **kwargs)
        self.ndims=ndims
        #
        if not isinstance(num_filters, list):
            num_filters = [num_filters]*2
        if not isinstance(prior_filter_num, list):
            prior_filter_num = [prior_filter_num]*2
        #
        self.level1 = norm_act_conv_pool_upsample(num_filters=num_filters[0], prior_filter_num=prior_filter_num[0], kernel_size=1, use_pooling=False, use_upsample=False, conv1=False)
        self.level2 = norm_act_conv_pool_upsample(num_filters=num_filters[1], prior_filter_num=prior_filter_num[1], kernel_size=1, use_pooling=False, use_upsample=False, conv1=False)
        self.level3 = norm_act_conv_pool_upsample(num_filters=num_filters[2], prior_filter_num=prior_filter_num[2], kernel_size=1, use_pooling=False, use_upsample=False, conv1=False)
        self.level4 = norm_act_conv_pool_upsample(num_filters=num_filters[3], prior_filter_num=prior_filter_num[3], kernel_size=1, use_pooling=False, use_upsample=False, conv1=False)
        self.level5 = norm_act_conv_pool_upsample(num_filters=num_filters[4], prior_filter_num=prior_filter_num[4], kernel_size=1, use_pooling=False, use_upsample=False, conv1=False)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(num_dense_units, activation='relu', kernel_initializer='he_normal')
        self.dropout1 = tf.keras.layers.Dropout(0.15)
        self.dense2 = tf.keras.layers.Dense(15, kernel_initializer='zeros', bias_initializer=tf.constant_initializer([0,0,0,0,0,0,1,1,1,0,0,0,0,0,0]))
    #
    def call(self, x, training=True):
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)
        x = self.level5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        affine_params = self.dense2(x)
        affine_matrix = make_affine_matrix_from_params(affine_params, ndims=self.ndims)
        return affine_params, affine_matrix

class decoder_1(tf.keras.Model):
    def __init__(self, num_filters=32, prior_filter_num=32, kernel_size=3, name='decoder_1', **kwargs):
        super(decoder_1, self).__init__(name=name, **kwargs)
        #
        if not isinstance(num_filters, list):
            num_filters = [num_filters]*4
        if not isinstance(prior_filter_num, list):
            prior_filter_num = [prior_filter_num]*4
        #only use 1x1x1 convolution in bottleneck dimension to save on trainable parameters
        self.level4 = norm_act_conv_pool_upsample(num_filters=num_filters[0], prior_filter_num=prior_filter_num[0], kernel_size=1, use_pooling=False, use_upsample=True, conv1=False)
        self.level3 = norm_act_conv_pool_upsample(num_filters=num_filters[1], prior_filter_num=prior_filter_num[1], kernel_size=kernel_size, use_pooling=False, use_upsample=True, conv1=True)
        self.level2 = norm_act_conv_pool_upsample(num_filters=num_filters[2], prior_filter_num=prior_filter_num[2], kernel_size=kernel_size, use_pooling=False, use_upsample=True, conv1=True)
        self.level1 = norm_act_conv_pool_upsample(num_filters=num_filters[3], prior_filter_num=prior_filter_num[3], kernel_size=kernel_size, use_pooling=False, use_upsample=False, conv1=False)
    #
    def call(self, skip4, skip3, skip2, skip1, training=True):
        _, x_upsample = self.level4(skip4)
        x_upsample = tf.concat([x_upsample, skip3], axis=-1)
        #
        _, x_upsample = self.level3(x_upsample)
        x_upsample = tf.concat([x_upsample, skip2], axis=-1)
        #
        _, x_upsample = self.level2(x_upsample)
        x_upsample = tf.concat([x_upsample, skip1], axis=-1)
        #
        x = self.level1(x_upsample)
        return x

class decoder_2(tf.keras.Model):
    def __init__(self, num_filters=32, prior_filter_num=None, conv1=False, kernel_size=3, kernel_initializer='he_normal', name='decoder_2', **kwargs):
        super(decoder_2, self).__init__(name=name, **kwargs)
        self.conv_kernel_size_1 = conv1
        #
        if self.conv_kernel_size_1 == True:
            self.conv = norm_act_conv_pool_upsample(num_filters=num_filters, prior_filter_num=prior_filter_num, kernel_size=1, use_pooling=False, use_upsample=False)
        self.level = norm_act_conv_pool_upsample(num_filters=num_filters, prior_filter_num=prior_filter_num, kernel_size=kernel_size, use_pooling=False, use_upsample=False)
    #
    def call(self, x1, x2, training=True):
        if self.conv_kernel_size_1 == True:
            x1 = self.conv(x1)
        x = tf.concat([x1, x2], axis=-1)
        x = self.level(x)
        return x

#Encoder shared between affine/deformable/segmentation networks
class shared_encoder_1(tf.keras.Model):
    def __init__(self, name='shared_encoder_1', **kwargs):
        super(shared_encoder_1, self).__init__(name=name, **kwargs)
        #
        #shared encoder
        self.encoder_top = norm_act_conv_pool_upsample(num_filters=64, kernel_size=3, use_pooling=False, use_upsample=False)
    #
    def call(self, x):
        x = self.encoder_top(x)
        #
        return x

class shared_encoder_2(tf.keras.Model):
    def __init__(self, name='shared_encoder_2', **kwargs):
        super(shared_encoder_2, self).__init__(name=name, **kwargs)
        #
        #shared encoder
        self.encoder1 = encoder_1(num_filters=[128,256,512,1024])
    #
    def call(self, x):
        skip1, skip2, skip3, skip4 = self.encoder1(x)
        #
        return skip1, skip2, skip3, skip4

#Size 32 models
class size_32_registration_network(tf.keras.Model):
    def __init__(self, shared_encoder_1, shared_encoder_2, batch_size, mixed_precision, name='size_32_registration_network', **kwargs):
        super(size_32_registration_network, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.shared_encoder_1 = shared_encoder_1
        self.shared_encoder_2 = shared_encoder_2
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        #
        #affine registration pathway
        self.top1_affine_reg = norm_act_conv_pool_upsample(num_filters=32, kernel_size=7, use_norm_act=False, use_pooling=False, use_upsample=False)
        self.encoder2 = encoder_2(num_filters=[512,256,128,64,32], prior_filter_num=[1024,512,256,128,64])
        #deformable registration pathway
        self.top1_deformable_reg = norm_act_conv_pool_upsample(num_filters=32, kernel_size=7, use_norm_act=False, use_pooling=False, use_upsample=False)
        self.decoder1_reg = decoder_1(num_filters=[512,512,256,128], prior_filter_num=[1024,1024,512,256])
        self.decoder2_1_reg = decoder_2(num_filters=64, prior_filter_num=128, conv1=True)
        self.decoder2_2_reg = decoder_2(num_filters=32, prior_filter_num=64, conv1=True)
        self.reg_conv1 = norm_act_conv_pool_upsample(num_filters=3, prior_filter_num=128, kernel_size=1, use_norm_act=True, use_pooling=False, use_upsample=False, use_bias=True, kernel_initializer='zeros', bias_initializer='zeros')
        self.reg_conv2 = norm_act_conv_pool_upsample(num_filters=3, prior_filter_num=64, kernel_size=1, use_norm_act=True, use_pooling=False, use_upsample=False, use_bias=True, kernel_initializer='zeros', bias_initializer='zeros')
        self.reg_conv3 = norm_act_conv_pool_upsample(num_filters=3, prior_filter_num=32, kernel_size=1, use_norm_act=True, use_pooling=False, use_upsample=False, use_bias=True, kernel_initializer='zeros', bias_initializer='zeros')
        self.add = tf.keras.layers.Add()
        self.cast_to_float_32_layer = tf.keras.layers.Activation('linear', dtype='float32')
        #
    def call(self, x):
        fixed_32 = tf.expand_dims(x[...,0], axis=-1)
        moving_32 = tf.expand_dims(x[...,1], axis=-1)
        moving_label_32 = tf.expand_dims(x[...,2], axis=-1)
        ######################################################################################
        #affine registration arm
        x = tf.concat([fixed_32, moving_32], axis=-1)
        x = self.top1_affine_reg(x)
        x = self.shared_encoder_1(x)
        _, _, _, x = self.shared_encoder_2(x)
        affine_params_32, affine_matrix_32 = self.encoder2(x)
        #move size 32 image/label using found affine transform
        moved_affine_32 = transform_image(affine_matrix_32, None, moving_32, [32,32,32], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        moved_label_affine_32 = transform_image(affine_matrix_32, None, moving_label_32, [32,32,32], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        ######################################################################################
        #deformable registration arm
        x = tf.concat([fixed_32, moved_affine_32], axis=-1)
        skip_a = self.top1_deformable_reg(x)
        skip_b = self.shared_encoder_1(skip_a)
        skip1, skip2, skip3, skip4 = self.shared_encoder_2(skip_b)
        out1 = self.decoder1_reg(skip4, skip3, skip2, skip1)
        out2 = self.decoder2_1_reg(out1, skip_b)
        out3 = self.decoder2_2_reg(out2, skip_a)
        #deep supervison
        out1_final = self.reg_conv1(out1)
        out2_final = self.reg_conv2(out2)
        out3_final = self.reg_conv3(out3)
        predicted_deformation_field_32 = self.add([out1_final, out2_final, out3_final])
        #move size 32 image/label using found affine/deformable transform
        moved_deformable_32 = transform_image(affine_matrix_32, predicted_deformation_field_32, moving_32, [32,32,32], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        moved_label_deformable_32 = transform_image(affine_matrix_32, predicted_deformation_field_32, moving_label_32, [32,32,32], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        ######################################################################################
        if self.mixed_precision == True:
            affine_params_32 = self.cast_to_float_32_layer(affine_params_32)
            affine_matrix_32 = self.cast_to_float_32_layer(affine_matrix_32)
            moved_affine_32 = self.cast_to_float_32_layer(moved_affine_32)
            moved_label_affine_32 = self.cast_to_float_32_layer(moved_label_affine_32)
            predicted_deformation_field_32 = self.cast_to_float_32_layer(predicted_deformation_field_32)
            moved_deformable_32 = self.cast_to_float_32_layer(moved_deformable_32)
            moved_label_deformable_32 = self.cast_to_float_32_layer(moved_label_deformable_32)
        #
        return affine_params_32, affine_matrix_32, moved_affine_32, moved_label_affine_32, predicted_deformation_field_32, moved_deformable_32, moved_label_deformable_32

class size_32_segmentation_network_1(tf.keras.Model):
    def __init__(self, shared_encoder_1, shared_encoder_2, mixed_precision, name='size_32_segmentation_network_1', **kwargs):
        super(size_32_segmentation_network_1, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.shared_encoder_1 = shared_encoder_1
        self.shared_encoder_2 = shared_encoder_2
        self.mixed_precision = mixed_precision
        #preliminary segmentation pathway
        self.top1_seg = norm_act_conv_pool_upsample(num_filters=32, kernel_size=3, use_norm_act=False, use_pooling=False, use_upsample=False)
        self.decoder1_seg = decoder_1(num_filters=[512,512,256,128], prior_filter_num=[1024,1024,512,256])
        self.decoder2_1_seg = decoder_2(num_filters=64, prior_filter_num=128, conv1=True)
        self.decoder2_2_seg = decoder_2(num_filters=32, prior_filter_num=64, conv1=True)
        bias_initializer_value = -np.log((1 - np.array(0.01)) / np.array(0.01))
        self.seg_conv1 = norm_act_conv_pool_upsample(num_filters=1, prior_filter_num=128, kernel_size=1, use_norm_act=True, use_pooling=False, use_upsample=False, use_bias=True, bias_initializer=tf.constant_initializer(bias_initializer_value))
        self.seg_conv2 = norm_act_conv_pool_upsample(num_filters=1, prior_filter_num=64, kernel_size=1, use_norm_act=True, use_pooling=False, use_upsample=False, use_bias=True, bias_initializer=tf.constant_initializer(bias_initializer_value))
        self.seg_conv3 = norm_act_conv_pool_upsample(num_filters=1, prior_filter_num=32, kernel_size=1, use_norm_act=True, use_pooling=False, use_upsample=False, use_bias=True, bias_initializer=tf.constant_initializer(bias_initializer_value))
        self.add = tf.keras.layers.Add()
        self.cast_to_float_32_layer = tf.keras.layers.Activation('linear', dtype='float32')
        #
    def call(self, x):
        fixed_32 = tf.expand_dims(x[...,0], axis=-1)
        ######################################################################################
        #preliminary segmentation arm
        skip_a = self.top1_seg(fixed_32)
        skip_b = self.shared_encoder_1(skip_a)
        skip1, skip2, skip3, skip4 = self.shared_encoder_2(skip_b)
        out1 = self.decoder1_seg(skip4, skip3, skip2, skip1)
        out2 = self.decoder2_1_seg(out1, skip_b)
        out3 = self.decoder2_2_seg(out2, skip_a)
        #deep supervison
        out1_final = self.seg_conv1(out1)
        out2_final = self.seg_conv2(out2)
        out3_final = self.seg_conv3(out3)
        pred_seg_32_final_1 = self.add([out1_final, out2_final, out3_final])
        ######################################################################################
        if self.mixed_precision == True:
            pred_seg_32_final_1 = self.cast_to_float_32_layer(pred_seg_32_final_1)
        #
        return pred_seg_32_final_1

class size_32_segmentation_network_2(tf.keras.Model):
    def __init__(self, mixed_precision, kernel_size_for_gaussian_conv=5, sigma_for_gaussian_conv=1, name='size_32_segmentation_network_2', **kwargs):
        super(size_32_segmentation_network_2, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.mixed_precision = mixed_precision
        self.kernel_size_for_gaussian_conv = kernel_size_for_gaussian_conv
        self.sigma_for_gaussian_conv = sigma_for_gaussian_conv
        #refinement segmentation pathway
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.gaussian_conv = gaussian_blur_conv(kernel_size_for_gaussian_conv=self.kernel_size_for_gaussian_conv, sigma_for_gaussian_conv=self.sigma_for_gaussian_conv)
        self.end_seg = norm_act_conv_pool_upsample(num_filters=32, kernel_size=3, use_norm_act=False, use_pooling=False, use_upsample=False)
        self.end_seg1 = norm_act_conv_pool_upsample(num_filters=32, kernel_size=3, use_norm_act=True, use_pooling=False, use_upsample=False)
        self.end_seg2 = norm_act_conv_pool_upsample(num_filters=32, kernel_size=3, use_norm_act=True, use_pooling=False, use_upsample=False)
        bias_initializer_value = -np.log((1 - np.array(0.01)) / np.array(0.01))
        self.seg_conv4 = norm_act_conv_pool_upsample(num_filters=1, prior_filter_num=32, kernel_size=1, use_norm_act=True, use_pooling=False, use_upsample=False, use_bias=True, bias_initializer=tf.constant_initializer(bias_initializer_value))
        self.cast_to_float_32_layer = tf.keras.layers.Activation('linear', dtype='float32')
        #
    def call(self, x):
        fixed_32 = tf.expand_dims(x[...,0], axis=-1)
        pred_seg_32_final_1 = tf.expand_dims(x[...,1], axis=-1)
        moved_deformable_32 = tf.expand_dims(x[...,2], axis=-1)
        moved_label_deformable_32 = tf.expand_dims(x[...,3], axis=-1)
        ######################################################################################
        #refinement segmentation arm
        pred_seg_32_sigmoid_final_1 = self.sigmoid(pred_seg_32_final_1)
        moved_label_deformable_32_gauss = self.gaussian_conv(moved_label_deformable_32)
        x = tf.concat([fixed_32, pred_seg_32_sigmoid_final_1, moved_deformable_32, moved_label_deformable_32_gauss], axis=-1)
        x = self.end_seg(x)
        skip = x
        x = self.end_seg1(x)
        x = self.end_seg2(x)
        x = skip + x
        pred_seg_32_final_2 = self.seg_conv4(x)
        ######################################################################################
        if self.mixed_precision == True:
            pred_seg_32_final_2 = self.cast_to_float_32_layer(pred_seg_32_final_2)
        #
        return pred_seg_32_final_2

#Size 64 models
class size_64_registration_network(tf.keras.Model):
    def __init__(self, total_model_32, batch_size, mixed_precision, name='size_64_registration_network', **kwargs):
        super(size_64_registration_network, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.total_model_32 = total_model_32
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        #
        self.pooling = tf.keras.layers.MaxPool3D(pool_size=2)
        self.upsample_operations3 = upsample_operations(3)
        self.upsample_operations128 = upsample_operations(128)
    #
    def call(self, x):
        fixed_64 = tf.expand_dims(x[...,0], axis=-1)
        moving_64 = tf.expand_dims(x[...,1], axis=-1)
        moving_label_64 = tf.expand_dims(x[...,2], axis=-1)
        fixed_32 = fixed_64[:,::2,::2,::2,:]
        moving_32 = moving_64[:,::2,::2,::2,:]
        moving_label_32 = moving_label_64[:,::2,::2,::2,:]
        ######################################################################################
        #affine registration arm of size 32
        x = tf.concat([fixed_32, moving_32], axis=-1)
        x = self.total_model_32.size_32_registration_network.layers[1].layers[2](x) #top1_affine_reg
        x = self.total_model_32.size_32_registration_network.layers[1].layers[0](x) #shared_encoder_1
        _, _, _, x = self.total_model_32.size_32_registration_network.layers[1].layers[1](x) #shared_encoder_2
        affine_params_32, affine_matrix_32 = self.total_model_32.size_32_registration_network.layers[1].layers[3](x) #encoder2
        #move size 32 image/label using found affine transform
        moved_affine_32 = transform_image(affine_matrix_32, None, moving_32, [32,32,32], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        moved_label_affine_32 = transform_image(affine_matrix_32, None, moving_label_32, [32,32,32], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        ######################################################################################
        #affine registration arm of size 64
        #multiply translation by two to ensure translation augmentation is correct at size 64
        affine_params_32 = tf.concat([affine_params_32[:, :3] * 2.0, affine_params_32[:, 3:]], axis=-1)
        affine_matrix_32 = tf.concat([affine_matrix_32[:, :, :-1], tf.expand_dims(affine_matrix_32[:, :, -1] * 2.0, axis=-1)], axis=-1)
        #move size 64 image as initialization
        moved_64 = transform_image(affine_matrix_32, None, moving_64, [64,64,64], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        #
        x = tf.concat([fixed_64, moved_64], axis=-1)
        x = self.total_model_32.size_32_registration_network.layers[1].layers[2](x) #top1_affine_reg
        x = self.total_model_32.size_32_registration_network.layers[1].layers[0](x) #shared_encoder_1
        #add pooling layer to account for size 64
        x = self.pooling(x)
        _, _, _, x = self.total_model_32.size_32_registration_network.layers[1].layers[1](x) #shared_encoder_2
        affine_params_64, affine_matrix_64 = self.total_model_32.size_32_registration_network.layers[1].layers[3](x) #encoder2
        #combine affines by composing them together
        affine_matrix_64 = compose_affine_transforms(affine_matrix_32, affine_matrix_64, self.batch_size, self.mixed_precision)
        #move size 64 image/label using found affine transform
        moved_affine_64 = transform_image(affine_matrix_64, None, moving_64, [64,64,64], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        moved_label_affine_64 = transform_image(affine_matrix_64, None, moving_label_64, [64,64,64], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        ######################################################################################
        #deformable registration arm of size 32
        x = tf.concat([fixed_32, moved_affine_64[:,::2,::2,::2,:]], axis=-1)
        skip_a = self.total_model_32.size_32_registration_network.layers[1].layers[4](x) #top1_deformable_reg
        skip_b = self.total_model_32.size_32_registration_network.layers[1].layers[0](skip_a) #shared_encoder_1
        skip1, skip2, skip3, skip4 = self.total_model_32.size_32_registration_network.layers[1].layers[1](skip_b) #shared_encoder_2
        out1 = self.total_model_32.size_32_registration_network.layers[1].layers[5](skip4, skip3, skip2, skip1) #decoder1_reg
        out2 = self.total_model_32.size_32_registration_network.layers[1].layers[6](out1, skip_b) #decoder2_1_reg
        out3 = self.total_model_32.size_32_registration_network.layers[1].layers[7](out2, skip_a) #decoder2_2_reg
        #deep supervison
        out1_final = self.total_model_32.size_32_registration_network.layers[1].layers[8](out1) #reg_conv1
        out2_final = self.total_model_32.size_32_registration_network.layers[1].layers[9](out2) #reg_conv2
        out3_final = self.total_model_32.size_32_registration_network.layers[1].layers[10](out3) #reg_conv3
        predicted_deformation_field_32 = self.total_model_32.size_32_registration_network.layers[1].layers[11]([out1_final, out2_final, out3_final]) #add
        #divide translation by two to ensure translation augmentation is correct at size 32
        affine_matrix_64_div_2 = tf.concat([affine_matrix_64[:, :, :-1], tf.expand_dims(affine_matrix_64[:, :, -1] / 2.0, axis=-1)], axis=-1)
        #move size 32 image/label using found affine/deformable transform
        moved_deformable_32 = transform_image(affine_matrix_64_div_2, predicted_deformation_field_32, moving_32, [32,32,32], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        moved_label_deformable_32 = transform_image(affine_matrix_64_div_2, predicted_deformation_field_32, moving_label_32, [32,32,32], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        ######################################################################################
        #deformable registration arm of size 64
        #upsample deformation field
        upsampled_predicted_deformation_field_32 = self.upsample_operations3(predicted_deformation_field_32)
        #multiply deformation field by two to ensure deformation is correct at size 64
        upsampled_predicted_deformation_field_32 = upsampled_predicted_deformation_field_32 * 2.0
        #move size 64 image as initialization
        moved_64 = transform_image(affine_matrix_64, upsampled_predicted_deformation_field_32, moving_64, [64,64,64], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        #
        x = tf.concat([fixed_64, moved_64], axis=-1)
        skip_a = self.total_model_32.size_32_registration_network.layers[1].layers[4](x) #top1_deformable_reg
        skip_b = self.total_model_32.size_32_registration_network.layers[1].layers[0](skip_a) #shared_encoder_1
        #add pooling layer to account for size 64
        x = self.pooling(skip_b)
        skip1, skip2, skip3, skip4 = self.total_model_32.size_32_registration_network.layers[1].layers[1](x) #shared_encoder_2
        out1 = self.total_model_32.size_32_registration_network.layers[1].layers[5](skip4, skip3, skip2, skip1) #decoder1_reg
        #add upsampling layer to account for size 64
        out1_upsample = self.upsample_operations128(out1)
        out2 = self.total_model_32.size_32_registration_network.layers[1].layers[6](out1_upsample, skip_b) #decoder2_1_reg
        out3 = self.total_model_32.size_32_registration_network.layers[1].layers[7](out2, skip_a) #decoder2_2_reg
        #deep supervison
        out1 = self.total_model_32.size_32_registration_network.layers[1].layers[8](out1) #reg_conv1
        #add upsampling layer to account for size 64
        out1_final = self.upsample_operations3(out1)
        out2_final = self.total_model_32.size_32_registration_network.layers[1].layers[9](out2) #reg_conv2
        out3_final = self.total_model_32.size_32_registration_network.layers[1].layers[10](out3) #reg_conv3
        predicted_deformation_field_64 = self.total_model_32.size_32_registration_network.layers[1].layers[11]([out1_final, out2_final, out3_final]) #add
        #combine deformations by adding fields together
        resampled_deformation_residual_64 = transform_image(None, upsampled_predicted_deformation_field_32, predicted_deformation_field_64, [64,64,64], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        predicted_deformation_field_64 = self.total_model_32.size_32_registration_network.layers[1].layers[11]([upsampled_predicted_deformation_field_32, resampled_deformation_residual_64]) #add
        #move size 64 image/label found affine/deformable transform
        moved_deformable_64 = transform_image(affine_matrix_64, predicted_deformation_field_64, moving_64, [64,64,64], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        moved_label_deformable_64 = transform_image(affine_matrix_64, predicted_deformation_field_64, moving_label_64, [64,64,64], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        ######################################################################################
        if self.mixed_precision == True:
            affine_params_32 = self.total_model_32.size_32_registration_network.layers[1].layers[12](affine_params_32) #cast_to_float_32_layer
            affine_params_64 = self.total_model_32.size_32_registration_network.layers[1].layers[12](affine_params_64) #cast_to_float_32_layer
            affine_matrix_32 = self.total_model_32.size_32_registration_network.layers[1].layers[12](affine_matrix_32) #cast_to_float_32_layer
            affine_matrix_64 = self.total_model_32.size_32_registration_network.layers[1].layers[12](affine_matrix_64) #cast_to_float_32_layer
            moved_affine_32 = self.total_model_32.size_32_registration_network.layers[1].layers[12](moved_affine_32) #cast_to_float_32_layer
            moved_affine_64 = self.total_model_32.size_32_registration_network.layers[1].layers[12](moved_affine_64) #cast_to_float_32_layer
            moved_label_affine_32 = self.total_model_32.size_32_registration_network.layers[1].layers[12](moved_label_affine_32) #cast_to_float_32_layer
            moved_label_affine_64 = self.total_model_32.size_32_registration_network.layers[1].layers[12](moved_label_affine_64) #cast_to_float_32_layer
            predicted_deformation_field_32 = self.total_model_32.size_32_registration_network.layers[1].layers[12](predicted_deformation_field_32) #cast_to_float_32_layer
            predicted_deformation_field_64 = self.total_model_32.size_32_registration_network.layers[1].layers[12](predicted_deformation_field_64) #cast_to_float_32_layer
            moved_deformable_32 = self.total_model_32.size_32_registration_network.layers[1].layers[12](moved_deformable_32) #cast_to_float_32_layer
            moved_deformable_64 = self.total_model_32.size_32_registration_network.layers[1].layers[12](moved_deformable_64) #cast_to_float_32_layer
            moved_label_deformable_32 = self.total_model_32.size_32_registration_network.layers[1].layers[12](moved_label_deformable_32) #cast_to_float_32_layer
            moved_label_deformable_64 = self.total_model_32.size_32_registration_network.layers[1].layers[12](moved_label_deformable_64) #cast_to_float_32_layer
        #
        return affine_params_32, affine_params_64, affine_matrix_32, affine_matrix_64, moved_affine_32, moved_affine_64, moved_label_affine_32, moved_label_affine_64, predicted_deformation_field_32, predicted_deformation_field_64, moved_deformable_32, moved_deformable_64, moved_label_deformable_32, moved_label_deformable_64

class size_64_segmentation_network_1(tf.keras.Model):
    def __init__(self, total_model_32, mixed_precision, name='size_64_segmentation_network_1', **kwargs):
        super(size_64_segmentation_network_1, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.total_model_32 = total_model_32
        self.mixed_precision = mixed_precision
        #
        self.pooling = tf.keras.layers.MaxPool3D(pool_size=2)
        self.upsample_operations128 = upsample_operations(128)
        self.upsample_operations1 = upsample_operations(1)
        #
    def call(self, x):
        fixed_64 = tf.expand_dims(x[...,0], axis=-1)
        ######################################################################################
        #preliminary segmentation arm
        skip_a = self.total_model_32.size_32_segmentation_network_1.layers[1].layers[2](fixed_64) #top1_seg
        skip_b = self.total_model_32.size_32_segmentation_network_1.layers[1].layers[0](skip_a) #shared_encoder_1
        #add pooling layer to account for size 64
        x = self.pooling(skip_b)
        skip1, skip2, skip3, skip4 = self.total_model_32.size_32_segmentation_network_1.layers[1].layers[1](x) #shared_encoder_2
        out1 = self.total_model_32.size_32_segmentation_network_1.layers[1].layers[3](skip4, skip3, skip2, skip1) #decoder1_seg
        #add upsampling layer to account for size 64
        out1_upsample = self.upsample_operations128(out1)
        out2 = self.total_model_32.size_32_segmentation_network_1.layers[1].layers[4](out1_upsample, skip_b) #decoder2_1_seg
        out3 = self.total_model_32.size_32_segmentation_network_1.layers[1].layers[5](out2, skip_a) #decoder2_2_seg
        #deep supervison
        out1 = self.total_model_32.size_32_segmentation_network_1.layers[1].layers[6](out1) #seg_conv1
        #add upsampling layer to account for size 64
        out1_final = self.upsample_operations1(out1)
        out2_final = self.total_model_32.size_32_segmentation_network_1.layers[1].layers[7](out2) #seg_conv2
        out3_final = self.total_model_32.size_32_segmentation_network_1.layers[1].layers[8](out3) #seg_conv3
        pred_seg_64_final_1 = self.total_model_32.size_32_segmentation_network_1.layers[1].layers[9]([out1_final, out2_final, out3_final]) #add
        ######################################################################################
        if self.mixed_precision == True:
            pred_seg_64_final_1 = self.total_model_32.size_32_segmentation_network_1.layers[1].layers[10](pred_seg_64_final_1) #cast_to_float_32_layer
        #
        return pred_seg_64_final_1

class size_64_segmentation_network_2(tf.keras.Model):
    def __init__(self, total_model_32, mixed_precision, name='size_64_segmentation_network_2', **kwargs):
        super(size_64_segmentation_network_2, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.total_model_32 = total_model_32
        self.mixed_precision = mixed_precision
        #
    def call(self, x):
        fixed_64 = tf.expand_dims(x[...,0], axis=-1)
        pred_seg_64_final_1 = tf.expand_dims(x[...,1], axis=-1)
        moved_deformable_64 = tf.expand_dims(x[...,2], axis=-1)
        moved_label_deformable_64 = tf.expand_dims(x[...,3], axis=-1)
        ######################################################################################
        #refinement segmentation arm
        pred_seg_64_sigmoid_final_1 = self.total_model_32.size_32_segmentation_network_2.layers[1].layers[0](pred_seg_64_final_1) #sigmoid
        moved_label_deformable_64_gauss = self.total_model_32.size_32_segmentation_network_2.layers[1].layers[1](moved_label_deformable_64) #gaussian_conv
        x = tf.concat([fixed_64, pred_seg_64_sigmoid_final_1, moved_deformable_64, moved_label_deformable_64_gauss], axis=-1)
        x = self.total_model_32.size_32_segmentation_network_2.layers[1].layers[2](x) #end_seg
        skip = x
        x = self.total_model_32.size_32_segmentation_network_2.layers[1].layers[3](x) #end_seg1
        x = self.total_model_32.size_32_segmentation_network_2.layers[1].layers[4](x) #end_seg2
        x = skip + x
        pred_seg_64_final_2 = self.total_model_32.size_32_segmentation_network_2.layers[1].layers[5](x) #seg_conv4
        ######################################################################################
        if self.mixed_precision == True:
            pred_seg_64_final_2 = self.total_model_32.size_32_segmentation_network_2.layers[1].layers[6](pred_seg_64_final_2) #cast_to_float_32_layer
        #
        return pred_seg_64_final_2

#Size 128 models
class size_128_registration_network(tf.keras.Model):
    def __init__(self, total_model_64, batch_size, mixed_precision, name='size_128_registration_network', **kwargs):
        super(size_128_registration_network, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.total_model_64 = total_model_64
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        #
        self.upsample_operations64 = upsample_operations(64)
    #
    def call(self, x):
        fixed_128 = tf.expand_dims(x[...,0], axis=-1)
        moving_128 = tf.expand_dims(x[...,1], axis=-1)
        moving_label_128 = tf.expand_dims(x[...,2], axis=-1)
        fixed_32 = fixed_128[:,::4,::4,::4,:]
        moving_32 = moving_128[:,::4,::4,::4,:]
        moving_label_32 = moving_label_128[:,::4,::4,::4,:]
        fixed_64 = fixed_128[:,::2,::2,::2,:]
        moving_64 = moving_128[:,::2,::2,::2,:]
        moving_label_64 = moving_label_128[:,::2,::2,::2,:]
        ######################################################################################
        #affine registration arm of size 32
        x = tf.concat([fixed_32, moving_32], axis=-1)
        x = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[2](x) #top1_affine_reg
        x = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[0](x) #shared_encoder_1
        _, _, _, x = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[1](x) #shared_encoder_2
        affine_params_32, affine_matrix_32 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[3](x) #encoder2
        #move size 32 image/label using found affine transform
        moved_affine_32 = transform_image(affine_matrix_32, None, moving_32, [32,32,32], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        moved_label_affine_32 = transform_image(affine_matrix_32, None, moving_label_32, [32,32,32], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        ######################################################################################
        #affine registration arm of size 64
        #multiply translation by two to ensure translation augmentation is correct at size 64
        affine_params_32 = tf.concat([affine_params_32[:, :3] * 2.0, affine_params_32[:, 3:]], axis=-1)
        affine_matrix_32 = tf.concat([affine_matrix_32[:, :, :-1], tf.expand_dims(affine_matrix_32[:, :, -1] * 2.0, axis=-1)], axis=-1)
        #move size 64 image as initialization
        moved_64 = transform_image(affine_matrix_32, None, moving_64, [64,64,64], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        #
        x = tf.concat([fixed_64, moved_64], axis=-1)
        x = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[2](x) #top1_affine_reg
        x = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[0](x) #shared_encoder_1
        #add pooling layer to account for size 64
        x = self.total_model_64.size_64_registration_network.layers[1].layers[1](x) #pooling
        _, _, _, x = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[1](x) #shared_encoder_2
        affine_params_64, affine_matrix_64 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[3](x) #encoder2
        #combine affines by composing them together
        affine_matrix_64 = compose_affine_transforms(affine_matrix_32, affine_matrix_64, self.batch_size, self.mixed_precision)
        #move size 64 image/label using found affine transform
        moved_affine_64 = transform_image(affine_matrix_64, None, moving_64, [64,64,64], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        moved_label_affine_64 = transform_image(affine_matrix_64, None, moving_label_64, [64,64,64], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        ######################################################################################
        #affine registration arm of size 128
        #multiply translation by two to ensure translation augmentation is correct at size 128
        affine_params_32 = tf.concat([affine_params_32[:, :3] * 2.0, affine_params_32[:, 3:]], axis=-1)
        affine_matrix_32 = tf.concat([affine_matrix_32[:, :, :-1], tf.expand_dims(affine_matrix_32[:, :, -1] * 2.0, axis=-1)], axis=-1)
        affine_params_64 = tf.concat([affine_params_64[:, :3] * 2.0, affine_params_64[:, 3:]], axis=-1)
        affine_matrix_64 = tf.concat([affine_matrix_64[:, :, :-1], tf.expand_dims(affine_matrix_64[:, :, -1] * 2.0, axis=-1)], axis=-1)
        #move size 128 image as initialization
        moved_128 = transform_image(affine_matrix_64, None, moving_128, [128,128,128], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        #
        x = tf.concat([fixed_128, moved_128], axis=-1)
        x = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[2](x) #top1_affine_reg
        #add pooling layer to account for size 128
        x = self.total_model_64.size_64_registration_network.layers[1].layers[1](x) #pooling
        x = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[0](x) #shared_encoder_1
        #add pooling layer to account for size 128
        x = self.total_model_64.size_64_registration_network.layers[1].layers[1](x) #pooling
        _, _, _, x = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[1](x) #shared_encoder_2
        affine_params_128, affine_matrix_128 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[3](x) #encoder2
        #combine affines by composing them together
        affine_matrix_128 = compose_affine_transforms(affine_matrix_64, affine_matrix_128, self.batch_size, self.mixed_precision)
        #move size 128 image/label using composed affine transform
        moved_affine_128 = transform_image(affine_matrix_128, None, moving_128, [128,128,128], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        moved_label_affine_128 = transform_image(affine_matrix_128, None, moving_label_128, [128,128,128], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        ######################################################################################
        #deformable registration arm of size 32
        x = tf.concat([fixed_32, moved_affine_128[:,::4,::4,::4,:]], axis=-1)
        skip_a = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[4](x) #top1_deformable_reg
        skip_b = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[0](skip_a) #shared_encoder_1
        skip1, skip2, skip3, skip4 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[1](skip_b) #shared_encoder_2
        out1 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[5](skip4, skip3, skip2, skip1) #decoder1_reg
        out2 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[6](out1, skip_b) #decoder2_1_reg
        out3 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[7](out2, skip_a) #decoder2_2_reg
        #deep supervison
        out1_final = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[8](out1) #reg_conv1
        out2_final = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[9](out2) #reg_conv2
        out3_final = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[10](out3) #reg_conv3
        predicted_deformation_field_32 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[11]([out1_final, out2_final, out3_final]) #add
        #divide translation by four to ensure translation augmentation is correct at size 32
        affine_matrix_128_div_4 = tf.concat([affine_matrix_128[:, :, :-1], tf.expand_dims(affine_matrix_128[:, :, -1] / 4.0, axis=-1)], axis=-1)
        #move size 32 image/label using found affine/deformable transform
        moved_deformable_32 = transform_image(affine_matrix_128_div_4, predicted_deformation_field_32, moving_32, [32,32,32], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        moved_label_deformable_32 = transform_image(affine_matrix_128_div_4, predicted_deformation_field_32, moving_label_32, [32,32,32], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        ######################################################################################
        #deformable registration arm of size 64
        #upsample deformation field
        upsampled_predicted_deformation_field_32 = self.total_model_64.size_64_registration_network.layers[1].layers[2](predicted_deformation_field_32)
        #multiply deformation field by two to ensure deformation is correct at size 64
        upsampled_predicted_deformation_field_32 = upsampled_predicted_deformation_field_32 * 2.0
        #divide translation by two to ensure translation augmentation is correct at size 64
        affine_matrix_128_div_2 = tf.concat([affine_matrix_128[:, :, :-1], tf.expand_dims(affine_matrix_128[:, :, -1] / 2.0, axis=-1)], axis=-1)
        #move size 64 image as initialization
        moved_64 = transform_image(affine_matrix_128_div_2, upsampled_predicted_deformation_field_32, moving_64, [64,64,64], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        #
        x = tf.concat([fixed_64, moved_64], axis=-1)
        skip_a = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[4](x) #top1_deformable_reg
        skip_b = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[0](skip_a) #shared_encoder_1
        #add pooling layer to account for size 64
        x = self.total_model_64.size_64_registration_network.layers[1].layers[1](skip_b)
        skip1, skip2, skip3, skip4 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[1](x) #shared_encoder_2
        out1 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[5](skip4, skip3, skip2, skip1) #decoder1_reg
        #add upsampling layer to account for size 64
        out1_upsample = self.total_model_64.size_64_registration_network.layers[1].layers[3](out1)
        out2 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[6](out1_upsample, skip_b) #decoder2_1_reg
        out3 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[7](out2, skip_a) #decoder2_2_reg
        #deep supervison
        out1 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[8](out1) #reg_conv1
        #add upsampling layer to account for size 64
        out1_final = self.total_model_64.size_64_registration_network.layers[1].layers[2](out1)
        out2_final = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[9](out2) #reg_conv2
        out3_final = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[10](out3) #reg_conv3
        predicted_deformation_field_64 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[11]([out1_final, out2_final, out3_final]) #add
        #combine deformations by adding fields together
        resampled_deformation_residual_64 = transform_image(None, upsampled_predicted_deformation_field_32, predicted_deformation_field_64, [64,64,64], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        predicted_deformation_field_64 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[11]([upsampled_predicted_deformation_field_32, resampled_deformation_residual_64]) #add
        #move size 64 image/label found affine/deformable transform
        moved_deformable_64 = transform_image(affine_matrix_128_div_2, predicted_deformation_field_64, moving_64, [64,64,64], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        moved_label_deformable_64 = transform_image(affine_matrix_128_div_2, predicted_deformation_field_64, moving_label_64, [64,64,64], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        ######################################################################################
        #deformable registration arm of size 128
        #upsample deformation field
        upsampled_predicted_deformation_field_64 = self.total_model_64.size_64_registration_network.layers[1].layers[2](predicted_deformation_field_64)
        #multiply deformation field by two to ensure deformation is correct at size 128
        upsampled_predicted_deformation_field_64 = upsampled_predicted_deformation_field_64 * 2.0
        #move size 128 image as initialization
        moved_128 = transform_image(affine_matrix_128, upsampled_predicted_deformation_field_64, moving_128, [128,128,128], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        #
        x = tf.concat([fixed_128, moved_128], axis=-1)
        skip_a = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[4](x) #top1_deformable_reg
        #add pooling layer to account for size 128
        x = self.total_model_64.size_64_registration_network.layers[1].layers[1](skip_a) #pooling
        skip_b = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[0](x) #shared_encoder_1
        #add pooling layer to account for size 128
        x = self.total_model_64.size_64_registration_network.layers[1].layers[1](skip_b) #pooling
        skip1, skip2, skip3, skip4 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[1](x) #shared_encoder_2
        out1 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[5](skip4, skip3, skip2, skip1) #decoder1_reg
        #add upsampling layer to account for size 128
        out1_upsample = self.total_model_64.size_64_registration_network.layers[1].layers[3](out1)
        out2 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[6](out1_upsample, skip_b) #decoder2_1_reg
        #add upsampling layer to account for size 128
        out2_upsample = self.upsample_operations64(out2)
        out3 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[7](out2_upsample, skip_a) #decoder2_2_reg
        #deep supervision
        out1 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[8](out1) #reg_conv1
        #add upsampling layer to account for size 128
        out1_final = self.total_model_64.size_64_registration_network.layers[1].layers[2](out1)
        out2 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[9](out2) #reg_conv2
        out2 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[11]([out1_final, out2]) #add
        #add upsampling layer to account for size 128
        out2_final = self.total_model_64.size_64_registration_network.layers[1].layers[2](out2)
        out3_final = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[10](out3) #reg_conv3
        predicted_deformation_field_128 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[11]([out2_final, out3_final]) #add
        #combine deformations by adding fields together
        resampled_deformation_residual_128 = transform_image(None, upsampled_predicted_deformation_field_64, predicted_deformation_field_128, [128,128,128], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        predicted_deformation_field_128 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[11]([upsampled_predicted_deformation_field_64, resampled_deformation_residual_128]) #add
        #move size 128 image/label using found deformable transform
        moved_deformable_128 = transform_image(affine_matrix_128, predicted_deformation_field_128, moving_128, [128,128,128], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        moved_label_deformable_128 = transform_image(affine_matrix_128, predicted_deformation_field_128, moving_label_128, [128,128,128], self.batch_size, 3, order=1, mode='constant', mixed_precision = self.mixed_precision)
        ######################################################################################
        if self.mixed_precision == True:
            affine_params_32 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](affine_params_32) #cast_to_float_32_layer
            affine_params_64 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](affine_params_64) #cast_to_float_32_layer
            affine_params_128 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](affine_params_128) #cast_to_float_32_layer
            affine_matrix_32 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](affine_matrix_32) #cast_to_float_32_layer
            affine_matrix_64 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](affine_matrix_64) #cast_to_float_32_layer
            affine_matrix_128 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](affine_matrix_128) #cast_to_float_32_layer
            moved_affine_32 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](moved_affine_32) #cast_to_float_32_layer
            moved_affine_64 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](moved_affine_64) #cast_to_float_32_layer
            moved_affine_128 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](moved_affine_128) #cast_to_float_32_layer
            moved_label_affine_32 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](moved_label_affine_32) #cast_to_float_32_layer
            moved_label_affine_64 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](moved_label_affine_64) #cast_to_float_32_layer
            moved_label_affine_128 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](moved_label_affine_128) #cast_to_float_32_layer
            predicted_deformation_field_32 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](predicted_deformation_field_32) #cast_to_float_32_layer
            predicted_deformation_field_64 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](predicted_deformation_field_64) #cast_to_float_32_layer
            predicted_deformation_field_128 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](predicted_deformation_field_128) #cast_to_float_32_layer
            moved_deformable_32 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](moved_deformable_32) #cast_to_float_32_layer
            moved_deformable_64 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](moved_deformable_64) #cast_to_float_32_layer
            moved_deformable_128 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](moved_deformable_128) #cast_to_float_32_layer
            moved_label_deformable_32 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](moved_label_deformable_32) #cast_to_float_32_layer
            moved_label_deformable_64 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](moved_label_deformable_64) #cast_to_float_32_layer
            moved_label_deformable_128 = self.total_model_64.size_64_registration_network.layers[1].layers[0].layers[0].layers[1].layers[12](moved_label_deformable_128) #cast_to_float_32_layer
        #
        return affine_params_32, affine_params_64, affine_params_128, affine_matrix_32, affine_matrix_64, affine_matrix_128, moved_affine_32, moved_affine_64, moved_affine_128, moved_label_affine_32, moved_label_affine_64, moved_label_affine_128, predicted_deformation_field_32, predicted_deformation_field_64, predicted_deformation_field_128, moved_deformable_32, moved_deformable_64, moved_deformable_128, moved_label_deformable_32, moved_label_deformable_64, moved_label_deformable_128

class size_128_segmentation_network_1(tf.keras.Model):
    def __init__(self, total_model_64, mixed_precision, name='size_128_segmentation_network_1', **kwargs):
        super(size_128_segmentation_network_1, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.total_model_64 = total_model_64
        self.mixed_precision = mixed_precision
        #
        self.upsample_operations64 = upsample_operations(64)
        #
    def call(self, x):
        fixed_128 = tf.expand_dims(x[...,0], axis=-1)
        ######################################################################################
        #preliminary segmentation arm
        skip_a = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[0].layers[1].layers[1].layers[2](fixed_128) #top1_seg
        #add pooling layer to account for size 128
        x = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[1](skip_a) #pooling
        skip_b = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[0].layers[1].layers[1].layers[0](x) #shared_encoder_1
        #add pooling layer to account for size 128
        x = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[1](skip_b) #pooling
        skip1, skip2, skip3, skip4 = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[0].layers[1].layers[1].layers[1](x) #shared_encoder_2
        out1 = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[0].layers[1].layers[1].layers[3](skip4, skip3, skip2, skip1) #decoder1_seg
        #add upsampling layer to account for size 128
        out1_upsample = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[2](out1)
        out2 = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[0].layers[1].layers[1].layers[4](out1_upsample, skip_b) #decoder2_1_seg
        #add upsampling layer to account for size 128
        out2_upsample = self.upsample_operations64(out2)
        out3 = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[0].layers[1].layers[1].layers[5](out2_upsample, skip_a) #decoder2_2_seg
        #deep supervision
        out1 = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[0].layers[1].layers[1].layers[6](out1) #seg_conv1
        #add upsampling layer to account for size 128
        out1_final = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[3](out1)
        out2 = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[0].layers[1].layers[1].layers[7](out2) #seg_conv2
        out2 = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[0].layers[1].layers[1].layers[9]([out1_final, out2]) #add
        #add upsampling layer to account for size 128
        out2_final = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[3](out2)
        out3_final = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[0].layers[1].layers[1].layers[8](out3) #seg_conv3
        pred_seg_128_final_1 = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[0].layers[1].layers[1].layers[9]([out2_final, out3_final]) #add
        ######################################################################################
        if self.mixed_precision == True:
            pred_seg_128_final_1 = self.total_model_64.size_64_segmentation_network_1.layers[1].layers[0].layers[1].layers[1].layers[10](pred_seg_128_final_1) #cast_to_float_32_layer
        #
        return pred_seg_128_final_1

class size_128_segmentation_network_2(tf.keras.Model):
    def __init__(self, total_model_64, mixed_precision, name='size_128_segmentation_network_2', **kwargs):
        super(size_128_segmentation_network_2, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.total_model_64 = total_model_64
        self.mixed_precision = mixed_precision
        #
    def call(self, x):
        fixed_128 = tf.expand_dims(x[...,0], axis=-1)
        pred_seg_128_final_1 = tf.expand_dims(x[...,1], axis=-1)
        moved_deformable_128 = tf.expand_dims(x[...,2], axis=-1)
        moved_label_deformable_128 = tf.expand_dims(x[...,3], axis=-1)
        ######################################################################################
        #refinement segmentation arm
        pred_seg_128_sigmoid_final_1 = self.total_model_64.size_64_segmentation_network_2.layers[1].layers[0].layers[2].layers[1].layers[0](pred_seg_128_final_1) #sigmoid
        moved_label_deformable_128_gauss = self.total_model_64.size_64_segmentation_network_2.layers[1].layers[0].layers[2].layers[1].layers[1](moved_label_deformable_128) #gaussian_conv
        x = tf.concat([fixed_128, pred_seg_128_sigmoid_final_1, moved_deformable_128, moved_label_deformable_128_gauss], axis=-1)
        x = self.total_model_64.size_64_segmentation_network_2.layers[1].layers[0].layers[2].layers[1].layers[2](x) #end_seg
        skip = x
        x = self.total_model_64.size_64_segmentation_network_2.layers[1].layers[0].layers[2].layers[1].layers[3](x) #end_seg1
        x = self.total_model_64.size_64_segmentation_network_2.layers[1].layers[0].layers[2].layers[1].layers[4](x) #end_seg2
        x = skip + x
        pred_seg_128_final_2 = self.total_model_64.size_64_segmentation_network_2.layers[1].layers[0].layers[2].layers[1].layers[5](x) #seg_conv4
        ######################################################################################
        if self.mixed_precision == True:
            pred_seg_128_final_2 = self.total_model_64.size_64_segmentation_network_2.layers[1].layers[0].layers[2].layers[1].layers[6](pred_seg_128_final_2) #cast_to_float_32_layer
        #
        return pred_seg_128_final_2

#initialize loss trackers
train_reg_loss_tracker = tf.keras.metrics.Mean(name='train_reg_loss')
train_mse_loss_tracker = tf.keras.metrics.Mean(name='train_mse_loss')
train_local_ncc_affine_loss_tracker = tf.keras.metrics.Mean(name='train_local_ncc_affine_loss')
train_global_ncc_affine_loss_tracker = tf.keras.metrics.Mean(name='train_global_ncc_affine_loss')
train_local_ncc_deformable_loss_tracker = tf.keras.metrics.Mean(name='train_local_ncc_deformable_loss')
train_global_ncc_deformable_loss_tracker = tf.keras.metrics.Mean(name='train_global_ncc_deformable_loss')
train_reg_dice_loss_tracker = tf.keras.metrics.Mean(name='train_reg_dice_loss')
train_reg_ce_loss_tracker = tf.keras.metrics.Mean(name='train_reg_ce_loss')
train_deformation_loss_tracker = tf.keras.metrics.Mean(name='train_deformation_loss')
train_seg_loss_1_tracker = tf.keras.metrics.Mean(name='train_seg_loss_1')
train_seg_dice_metric_1_tracker = tf.keras.metrics.Mean(name='train_seg_dice_metric_1')
train_seg_ce_loss_1_tracker = tf.keras.metrics.Mean(name='train_seg_ce_loss_1')
train_seg_loss_2_tracker = tf.keras.metrics.Mean(name='train_seg_loss_2')
train_seg_dice_metric_2_tracker = tf.keras.metrics.Mean(name='train_seg_dice_metric_2')
train_seg_ce_loss_2_tracker = tf.keras.metrics.Mean(name='train_seg_ce_loss_2')
#
test_reg_loss_tracker = tf.keras.metrics.Mean(name='test_reg_loss')
test_mse_loss_tracker = tf.keras.metrics.Mean(name='test_mse_loss')
test_local_ncc_affine_loss_tracker = tf.keras.metrics.Mean(name='test_local_ncc_affine_loss')
test_global_ncc_affine_loss_tracker = tf.keras.metrics.Mean(name='test_global_ncc_affine_loss')
test_local_ncc_deformable_loss_tracker = tf.keras.metrics.Mean(name='test_local_ncc_deformable_loss')
test_global_ncc_deformable_loss_tracker = tf.keras.metrics.Mean(name='test_global_ncc_deformable_loss')
test_reg_dice_loss_tracker = tf.keras.metrics.Mean(name='test_reg_dice_loss')
test_reg_ce_loss_tracker = tf.keras.metrics.Mean(name='test_reg_ce_loss')
test_deformation_loss_tracker = tf.keras.metrics.Mean(name='test_deformation_loss')
test_seg_loss_1_tracker = tf.keras.metrics.Mean(name='test_seg_loss_1')
test_seg_dice_metric_1_tracker = tf.keras.metrics.Mean(name='test_seg_dice_metric_1')
test_seg_ce_loss_1_tracker = tf.keras.metrics.Mean(name='test_seg_ce_loss_1')
test_seg_loss_2_tracker = tf.keras.metrics.Mean(name='test_seg_loss_2')
test_seg_dice_metric_2_tracker = tf.keras.metrics.Mean(name='test_seg_dice_metric_2')
test_seg_ce_loss_2_tracker = tf.keras.metrics.Mean(name='test_seg_ce_loss_2')
#
train_loss_trackers_list = [train_reg_loss_tracker, train_mse_loss_tracker, train_local_ncc_affine_loss_tracker, train_global_ncc_affine_loss_tracker, train_local_ncc_deformable_loss_tracker, train_global_ncc_deformable_loss_tracker, train_reg_dice_loss_tracker, train_reg_ce_loss_tracker, train_deformation_loss_tracker, train_seg_loss_1_tracker, train_seg_dice_metric_1_tracker, train_seg_ce_loss_1_tracker, train_seg_loss_2_tracker, train_seg_dice_metric_2_tracker, train_seg_ce_loss_2_tracker]
test_loss_trackers_list = [test_reg_loss_tracker, test_mse_loss_tracker, test_local_ncc_affine_loss_tracker, test_global_ncc_affine_loss_tracker, test_local_ncc_deformable_loss_tracker, test_global_ncc_deformable_loss_tracker, test_reg_dice_loss_tracker, test_reg_ce_loss_tracker, test_deformation_loss_tracker, test_seg_loss_1_tracker, test_seg_dice_metric_1_tracker, test_seg_ce_loss_1_tracker, test_seg_loss_2_tracker, test_seg_dice_metric_2_tracker, test_seg_ce_loss_2_tracker]

class Reg_Seg_32(tf.keras.Model):
    def __init__(self, size_32_registration_network, size_32_segmentation_network_1, size_32_segmentation_network_2, mixed_precision, ndims, name='Reg_Seg_32', **kwargs):
        super(Reg_Seg_32, self).__init__(name=name, **kwargs)
        self.size_32_registration_network = size_32_registration_network
        self.size_32_segmentation_network_1 = size_32_segmentation_network_1
        self.size_32_segmentation_network_2 = size_32_segmentation_network_2
        self.mixed_precision = mixed_precision
        self.ndims = ndims
        #
    def compile(self, registration_optimizer, segmentation_1_optimizer, segmentation_2_optimizer):
        super(Reg_Seg_32, self).compile()
        self.registration_optimizer = registration_optimizer
        self.segmentation_1_optimizer = segmentation_1_optimizer
        self.segmentation_2_optimizer = segmentation_2_optimizer
        #
    def get_data(self, data):
        x, y = data
        fixed =  tf.cast(tf.expand_dims(x[...,0], axis=-1), tf.float32)
        moving = tf.cast(tf.expand_dims(x[...,1], axis=-1), tf.float32)
        fixed_label = tf.cast(tf.expand_dims(y[0][...,0], axis=-1), tf.float32)
        moving_label = tf.cast(tf.expand_dims(y[0][...,1], axis=-1), tf.float32)
        mask_for_dice_loss_label = tf.cast(tf.expand_dims(y[0][...,2], axis=-1), tf.float32)
        mask_for_NCC_deformable_loss_label = tf.cast(tf.expand_dims(y[0][...,3], axis=-1), tf.float32)
        true_affine_params = tf.cast(y[1], tf.float32)
        true_affine_matrix = make_affine_matrix_from_params(true_affine_params, ndims=self.ndims)
        return fixed, moving, fixed_label, moving_label, mask_for_dice_loss_label, mask_for_NCC_deformable_loss_label, true_affine_params, true_affine_matrix
    #
    def train_step(self, data):
        fixed, moving, fixed_label, moving_label, mask_for_dice_loss_label, mask_for_NCC_deformable_loss_label, true_affine_params, true_affine_matrix = self.get_data(data)
        # registration model
        with tf.GradientTape() as tape:
            affine_params_32, affine_matrix_32, moved_affine_32, moved_label_affine_32, predicted_deformation_field_32, moved_deformable_32, moved_label_deformable_32 = self.size_32_registration_network(tf.concat([fixed, moving, moving_label], axis=-1), training=True)
            # Compute the loss
            registration_loss = total_loss_reg_32(fixed, fixed_label, true_affine_params, true_affine_matrix, affine_params_32, affine_matrix_32, moved_affine_32, moved_label_affine_32, predicted_deformation_field_32, moved_deformable_32, moved_label_deformable_32, mask_for_dice_loss_label, mask_for_NCC_deformable_loss_label)
            if self.mixed_precision == True:
                total_registration_loss = self.registration_optimizer.get_scaled_loss(registration_loss[0])
            else:
                total_registration_loss = registration_loss[0]
        # Compute gradients.
        grads_reg = tape.gradient(total_registration_loss, self.size_32_registration_network.trainable_variables)
        if self.mixed_precision == True:
            grads_reg = self.registration_optimizer.get_unscaled_gradients(grads_reg)
        # Update the trainable parameters.
        self.registration_optimizer.apply_gradients(zip(grads_reg, self.size_32_registration_network.trainable_variables))
        #
        # segmentation model 1
        with tf.GradientTape() as tape:
            pred_seg_32_final_1 = self.size_32_segmentation_network_1(fixed, training=True)
            # Compute the loss
            segmentation_loss_1 = total_loss_seg(fixed_label, pred_seg_32_final_1)
            if self.mixed_precision == True:
                total_segmentation_loss_1 = self.segmentation_1_optimizer.get_scaled_loss(segmentation_loss_1[0])
            else:
                total_segmentation_loss_1 = segmentation_loss_1[0]
        # Compute gradients.
        grads_seg_1 = tape.gradient(total_segmentation_loss_1, self.size_32_segmentation_network_1.trainable_variables)
        if self.mixed_precision == True:
            grads_seg_1 = self.segmentation_1_optimizer.get_unscaled_gradients(grads_seg_1)
        # Update the trainable parameters.
        self.segmentation_1_optimizer.apply_gradients(zip(grads_seg_1, self.size_32_segmentation_network_1.trainable_variables))
        #
        # segmentation model 2
        with tf.GradientTape() as tape:
            pred_seg_32_final_2 = self.size_32_segmentation_network_2(tf.concat([fixed, pred_seg_32_final_1, moved_deformable_32, moved_label_deformable_32], axis=-1), training=True)
            # Compute the loss
            segmentation_loss_2 = total_loss_seg(fixed_label, pred_seg_32_final_2)
            if self.mixed_precision == True:
                total_segmentation_loss_2 = self.segmentation_2_optimizer.get_scaled_loss(segmentation_loss_2[0])
            else:
                total_segmentation_loss_2 = segmentation_loss_2[0]
        # Compute gradients.
        grads_seg_2 = tape.gradient(total_segmentation_loss_2, self.size_32_segmentation_network_2.trainable_variables)
        if self.mixed_precision == True:
            grads_seg_2 = self.segmentation_2_optimizer.get_unscaled_gradients(grads_seg_2)
        # Update the trainable parameters.
        self.segmentation_2_optimizer.apply_gradients(zip(grads_seg_2, self.size_32_segmentation_network_2.trainable_variables))
        #
        return self.update_and_return_tracker_values(train_loss_trackers_list, registration_loss, segmentation_loss_1, segmentation_loss_2)
    #
    def test_step(self, data):
        fixed, moving, fixed_label, moving_label, mask_for_dice_loss_label, mask_for_NCC_deformable_loss_label, true_affine_params, true_affine_matrix = self.get_data(data)
        # registration model
        affine_params_32, affine_matrix_32, moved_affine_32, moved_label_affine_32, predicted_deformation_field_32, moved_deformable_32, moved_label_deformable_32 = self.size_32_registration_network(tf.concat([fixed, moving, moving_label], axis=-1), training=False)
        registration_loss = total_loss_reg_32(fixed, fixed_label, true_affine_params, true_affine_matrix, affine_params_32, affine_matrix_32, moved_affine_32, moved_label_affine_32, predicted_deformation_field_32, moved_deformable_32, moved_label_deformable_32, mask_for_dice_loss_label, mask_for_NCC_deformable_loss_label)
        # segmentation model 1
        pred_seg_32_final_1 = self.size_32_segmentation_network_1(fixed, training=False)
        segmentation_loss_1 = total_loss_seg(fixed_label, pred_seg_32_final_1)
        # segmentation model 2
        pred_seg_32_final_2 = self.size_32_segmentation_network_2(tf.concat([fixed, pred_seg_32_final_1, moved_deformable_32, moved_label_deformable_32], axis=-1), training=False)
        segmentation_loss_2 = total_loss_seg(fixed_label, pred_seg_32_final_2)
        #
        return self.update_and_return_tracker_values(test_loss_trackers_list, registration_loss, segmentation_loss_1, segmentation_loss_2)
    #
    def update_and_return_tracker_values(self, tracker_list, registration_loss, segmentation_loss_1, segmentation_loss_2):
        #update tracker states
        [mean_tracker.update_state(loss_value) for mean_tracker,loss_value in zip(tracker_list, registration_loss + segmentation_loss_1 + segmentation_loss_2)]
        return {
                "Reg_Loss": tracker_list[0].result(),
                "MSE": tracker_list[1].result(),
                "Local_NCC_Affine": tracker_list[2].result(),
                "Global_NCC_Affine": tracker_list[3].result(),
                "Local_NCC_Deformable": tracker_list[4].result(),
                "Global_NCC_Deformable": tracker_list[5].result(),
                "Reg_Dice": tracker_list[6].result(),
                "Reg_CE": tracker_list[7].result(),
                "Deformation": tracker_list[8].result(),
                "Seg_Loss_1": tracker_list[9].result(),
                "Seg_Dice_1": tracker_list[10].result(),
                "Seg_CE_1": tracker_list[11].result(),
                "Seg_Loss_2": tracker_list[12].result(),
                "Seg_Dice_2": tracker_list[13].result(),
                "Seg_CE_2": tracker_list[14].result()
                }
    #
    @property
    def metrics(self):
        # `reset_states()` at beginning of each epoch
        return train_loss_trackers_list + test_loss_trackers_list

class Reg_Seg_64(tf.keras.Model):
    def __init__(self, size_64_registration_network, size_64_segmentation_network_1, size_64_segmentation_network_2, mixed_precision, ndims, name='Reg_Seg_64', **kwargs):
        super(Reg_Seg_64, self).__init__(name=name, **kwargs)
        self.size_64_registration_network = size_64_registration_network
        self.size_64_segmentation_network_1 = size_64_segmentation_network_1
        self.size_64_segmentation_network_2 = size_64_segmentation_network_2
        self.mixed_precision = mixed_precision
        self.ndims = ndims
        #
    def compile(self, registration_optimizer, segmentation_1_optimizer, segmentation_2_optimizer):
        super(Reg_Seg_64, self).compile()
        self.registration_optimizer = registration_optimizer
        self.segmentation_1_optimizer = segmentation_1_optimizer
        self.segmentation_2_optimizer = segmentation_2_optimizer
        #
    def get_data(self, data):
        x, y = data
        fixed =  tf.cast(tf.expand_dims(x[...,0], axis=-1), tf.float32)
        moving = tf.cast(tf.expand_dims(x[...,1], axis=-1), tf.float32)
        fixed_label = tf.cast(tf.expand_dims(y[0][...,0], axis=-1), tf.float32)
        moving_label = tf.cast(tf.expand_dims(y[0][...,1], axis=-1), tf.float32)
        mask_for_dice_loss_label = tf.cast(tf.expand_dims(y[0][...,2], axis=-1), tf.float32)
        mask_for_NCC_deformable_loss_label = tf.cast(tf.expand_dims(y[0][...,3], axis=-1), tf.float32)
        true_affine_params = tf.cast(y[1], tf.float32)
        true_affine_matrix = make_affine_matrix_from_params(true_affine_params, ndims=self.ndims)
        return fixed, moving, fixed_label, moving_label, mask_for_dice_loss_label, mask_for_NCC_deformable_loss_label, true_affine_params, true_affine_matrix
    #
    def train_step(self, data):
        fixed, moving, fixed_label, moving_label, mask_for_dice_loss_label, mask_for_NCC_deformable_loss_label, true_affine_params, true_affine_matrix = self.get_data(data)
        # registration model
        with tf.GradientTape() as tape:
            affine_params_32, affine_params_64, affine_matrix_32, affine_matrix_64, moved_affine_32, moved_affine_64, moved_label_affine_32, moved_label_affine_64, predicted_deformation_field_32, predicted_deformation_field_64, moved_deformable_32, moved_deformable_64, moved_label_deformable_32, moved_label_deformable_64 = self.size_64_registration_network(tf.concat([fixed, moving, moving_label], axis=-1), training=True)
            # Compute the loss
            registration_loss = total_loss_reg_64(fixed, fixed_label, true_affine_params, true_affine_matrix, affine_params_32, affine_params_64, affine_matrix_32, affine_matrix_64, moved_affine_32, moved_affine_64, moved_label_affine_32, moved_label_affine_64, predicted_deformation_field_32, predicted_deformation_field_64, moved_deformable_32, moved_deformable_64, moved_label_deformable_32, moved_label_deformable_64, mask_for_dice_loss_label, mask_for_NCC_deformable_loss_label)
            if self.mixed_precision == True:
                total_registration_loss = self.registration_optimizer.get_scaled_loss(registration_loss[0])
            else:
                total_registration_loss = registration_loss[0]
        # Compute gradients.
        grads_reg = tape.gradient(total_registration_loss, self.size_64_registration_network.trainable_variables)
        if self.mixed_precision == True:
            grads_reg = self.registration_optimizer.get_unscaled_gradients(grads_reg)
        # Update the trainable parameters.
        self.registration_optimizer.apply_gradients(zip(grads_reg, self.size_64_registration_network.trainable_variables))
        #
        # segmentation model 1
        with tf.GradientTape() as tape:
            pred_seg_64_final_1 = self.size_64_segmentation_network_1(fixed, training=True)
            # Compute the loss
            segmentation_loss_1 = total_loss_seg(fixed_label, pred_seg_64_final_1)
            if self.mixed_precision == True:
                total_segmentation_loss_1 = self.segmentation_1_optimizer.get_scaled_loss(segmentation_loss_1[0])
            else:
                total_segmentation_loss_1 = segmentation_loss_1[0]
        # Compute gradients.
        grads_seg_1 = tape.gradient(total_segmentation_loss_1, self.size_64_segmentation_network_1.trainable_variables)
        if self.mixed_precision == True:
            grads_seg_1 = self.segmentation_1_optimizer.get_unscaled_gradients(grads_seg_1)
        # Update the trainable parameters.
        self.segmentation_1_optimizer.apply_gradients(zip(grads_seg_1, self.size_64_segmentation_network_1.trainable_variables))
        #
        # segmentation model 2
        with tf.GradientTape() as tape:
            pred_seg_64_final_2 = self.size_64_segmentation_network_2(tf.concat([fixed, pred_seg_64_final_1, moved_deformable_64, moved_label_deformable_64], axis=-1), training=True)
            # Compute the loss
            segmentation_loss_2 = total_loss_seg(fixed_label, pred_seg_64_final_2)
            if self.mixed_precision == True:
                total_segmentation_loss_2 = self.segmentation_2_optimizer.get_scaled_loss(segmentation_loss_2[0])
            else:
                total_segmentation_loss_2 = segmentation_loss_2[0]
        # Compute gradients.
        grads_seg_2 = tape.gradient(total_segmentation_loss_2, self.size_64_segmentation_network_2.trainable_variables)
        if self.mixed_precision == True:
            grads_seg_2 = self.segmentation_2_optimizer.get_unscaled_gradients(grads_seg_2)
        # Update the trainable parameters.
        self.segmentation_2_optimizer.apply_gradients(zip(grads_seg_2, self.size_64_segmentation_network_2.trainable_variables))
        #
        return self.update_and_return_tracker_values(train_loss_trackers_list, registration_loss, segmentation_loss_1, segmentation_loss_2)
    #
    def test_step(self, data):
        fixed, moving, fixed_label, moving_label, mask_for_dice_loss_label, mask_for_NCC_deformable_loss_label, true_affine_params, true_affine_matrix = self.get_data(data)
        # registration model
        affine_params_32, affine_params_64, affine_matrix_32, affine_matrix_64, moved_affine_32, moved_affine_64, moved_label_affine_32, moved_label_affine_64, predicted_deformation_field_32, predicted_deformation_field_64, moved_deformable_32, moved_deformable_64, moved_label_deformable_32, moved_label_deformable_64 = self.size_64_registration_network(tf.concat([fixed, moving, moving_label], axis=-1), training=False)
        registration_loss = total_loss_reg_64(fixed, fixed_label, true_affine_params, true_affine_matrix, affine_params_32, affine_params_64, affine_matrix_32, affine_matrix_64, moved_affine_32, moved_affine_64, moved_label_affine_32, moved_label_affine_64, predicted_deformation_field_32, predicted_deformation_field_64, moved_deformable_32, moved_deformable_64, moved_label_deformable_32, moved_label_deformable_64, mask_for_dice_loss_label, mask_for_NCC_deformable_loss_label)
        # segmentation model 1
        pred_seg_64_final_1 = self.size_64_segmentation_network_1(fixed, training=False)
        segmentation_loss_1 = total_loss_seg(fixed_label, pred_seg_64_final_1)
        # segmentation model 2
        pred_seg_64_final_2 = self.size_64_segmentation_network_2(tf.concat([fixed, pred_seg_64_final_1, moved_deformable_64, moved_label_deformable_64], axis=-1), training=False)
        segmentation_loss_2 = total_loss_seg(fixed_label, pred_seg_64_final_2)
        #
        return self.update_and_return_tracker_values(test_loss_trackers_list, registration_loss, segmentation_loss_1, segmentation_loss_2)
    #
    def update_and_return_tracker_values(self, tracker_list, registration_loss, segmentation_loss_1, segmentation_loss_2):
        #update tracker states
        [mean_tracker.update_state(loss_value) for mean_tracker,loss_value in zip(tracker_list, registration_loss + segmentation_loss_1 + segmentation_loss_2)]
        return {
                "Reg_Loss": tracker_list[0].result(),
                "MSE": tracker_list[1].result(),
                "Local_NCC_Affine": tracker_list[2].result(),
                "Global_NCC_Affine": tracker_list[3].result(),
                "Local_NCC_Deformable": tracker_list[4].result(),
                "Global_NCC_Deformable": tracker_list[5].result(),
                "Reg_Dice": tracker_list[6].result(),
                "Reg_CE": tracker_list[7].result(),
                "Deformation": tracker_list[8].result(),
                "Seg_Loss_1": tracker_list[9].result(),
                "Seg_Dice_1": tracker_list[10].result(),
                "Seg_CE_1": tracker_list[11].result(),
                "Seg_Loss_2": tracker_list[12].result(),
                "Seg_Dice_2": tracker_list[13].result(),
                "Seg_CE_2": tracker_list[14].result()
                }
    #
    @property
    def metrics(self):
        # `reset_states()` at beginning of each epoch
        return train_loss_trackers_list + test_loss_trackers_list

class Reg_Seg_128(tf.keras.Model):
    def __init__(self, size_128_registration_network, size_128_segmentation_network_1, size_128_segmentation_network_2, mixed_precision, ndims, name='Reg_Seg_128', **kwargs):
        super(Reg_Seg_128, self).__init__(name=name, **kwargs)
        self.size_128_registration_network = size_128_registration_network
        self.size_128_segmentation_network_1 = size_128_segmentation_network_1
        self.size_128_segmentation_network_2 = size_128_segmentation_network_2
        self.mixed_precision = mixed_precision
        self.ndims = ndims
        #
    def compile(self, registration_optimizer, segmentation_1_optimizer, segmentation_2_optimizer):
        super(Reg_Seg_128, self).compile()
        self.registration_optimizer = registration_optimizer
        self.segmentation_1_optimizer = segmentation_1_optimizer
        self.segmentation_2_optimizer = segmentation_2_optimizer
        #
    def get_data(self, data):
        x, y = data
        fixed =  tf.cast(tf.expand_dims(x[...,0], axis=-1), tf.float32)
        moving = tf.cast(tf.expand_dims(x[...,1], axis=-1), tf.float32)
        fixed_label = tf.cast(tf.expand_dims(y[0][...,0], axis=-1), tf.float32)
        moving_label = tf.cast(tf.expand_dims(y[0][...,1], axis=-1), tf.float32)
        mask_for_dice_loss_label = tf.cast(tf.expand_dims(y[0][...,2], axis=-1), tf.float32)
        mask_for_NCC_deformable_loss_label = tf.cast(tf.expand_dims(y[0][...,3], axis=-1), tf.float32)
        true_affine_params = tf.cast(y[1], tf.float32)
        true_affine_matrix = make_affine_matrix_from_params(true_affine_params, ndims=self.ndims)
        return fixed, moving, fixed_label, moving_label, mask_for_dice_loss_label, mask_for_NCC_deformable_loss_label, true_affine_params, true_affine_matrix
    #
    def train_step(self, data):
        fixed, moving, fixed_label, moving_label, mask_for_dice_loss_label, mask_for_NCC_deformable_loss_label, true_affine_params, true_affine_matrix = self.get_data(data)
        # registration model
        with tf.GradientTape() as tape:
            affine_params_32, affine_params_64, affine_params_128, affine_matrix_32, affine_matrix_64, affine_matrix_128, moved_affine_32, moved_affine_64, moved_affine_128, moved_label_affine_32, moved_label_affine_64, moved_label_affine_128, predicted_deformation_field_32, predicted_deformation_field_64, predicted_deformation_field_128, moved_deformable_32, moved_deformable_64, moved_deformable_128, moved_label_deformable_32, moved_label_deformable_64, moved_label_deformable_128 = self.size_128_registration_network(tf.concat([fixed, moving, moving_label], axis=-1), training=True)
            # Compute the loss
            registration_loss = total_loss_reg_128(fixed, fixed_label, true_affine_params, true_affine_matrix, affine_params_32, affine_params_64, affine_params_128, affine_matrix_32, affine_matrix_64, affine_matrix_128, moved_affine_32, moved_affine_64, moved_affine_128, moved_label_affine_32, moved_label_affine_64, moved_label_affine_128, predicted_deformation_field_32, predicted_deformation_field_64, predicted_deformation_field_128, moved_deformable_32, moved_deformable_64, moved_deformable_128, moved_label_deformable_32, moved_label_deformable_64, moved_label_deformable_128, mask_for_dice_loss_label, mask_for_NCC_deformable_loss_label)
            if self.mixed_precision == True:
                total_registration_loss = self.registration_optimizer.get_scaled_loss(registration_loss[0])
            else:
                total_registration_loss = registration_loss[0]
        # Compute gradients.
        grads_reg = tape.gradient(total_registration_loss, self.size_128_registration_network.trainable_variables)
        if self.mixed_precision == True:
            grads_reg = self.registration_optimizer.get_unscaled_gradients(grads_reg)
        # Update the trainable parameters.
        self.registration_optimizer.apply_gradients(zip(grads_reg, self.size_128_registration_network.trainable_variables))
        #
        # segmentation model 1
        with tf.GradientTape() as tape:
            pred_seg_128_final_1 = self.size_128_segmentation_network_1(fixed, training=True)
            # Compute the loss
            segmentation_loss_1 = total_loss_seg(fixed_label, pred_seg_128_final_1)
            if self.mixed_precision == True:
                total_segmentation_loss_1 = self.segmentation_1_optimizer.get_scaled_loss(segmentation_loss_1[0])
            else:
                total_segmentation_loss_1 = segmentation_loss_1[0]
        # Compute gradients.
        grads_seg_1 = tape.gradient(total_segmentation_loss_1, self.size_128_segmentation_network_1.trainable_variables)
        if self.mixed_precision == True:
            grads_seg_1 = self.segmentation_1_optimizer.get_unscaled_gradients(grads_seg_1)
        # Update the trainable parameters.
        self.segmentation_1_optimizer.apply_gradients(zip(grads_seg_1, self.size_128_segmentation_network_1.trainable_variables))
        #
        # segmentation model 2
        with tf.GradientTape() as tape:
            pred_seg_64_final_2 = self.size_128_segmentation_network_2(tf.concat([fixed, pred_seg_128_final_1, moved_deformable_128, moved_label_deformable_128], axis=-1), training=True)
            # Compute the loss
            segmentation_loss_2 = total_loss_seg(fixed_label, pred_seg_64_final_2)
            if self.mixed_precision == True:
                total_segmentation_loss_2 = self.segmentation_2_optimizer.get_scaled_loss(segmentation_loss_2[0])
            else:
                total_segmentation_loss_2 = segmentation_loss_2[0]
        # Compute gradients.
        grads_seg_2 = tape.gradient(total_segmentation_loss_2, self.size_128_segmentation_network_2.trainable_variables)
        if self.mixed_precision == True:
            grads_seg_2 = self.segmentation_2_optimizer.get_unscaled_gradients(grads_seg_2)
        # Update the trainable parameters.
        self.segmentation_2_optimizer.apply_gradients(zip(grads_seg_2, self.size_128_segmentation_network_2.trainable_variables))
        #
        return self.update_and_return_tracker_values(train_loss_trackers_list, registration_loss, segmentation_loss_1, segmentation_loss_2)
    #
    def test_step(self, data):
        fixed, moving, fixed_label, moving_label, mask_for_dice_loss_label, mask_for_NCC_deformable_loss_label, true_affine_params, true_affine_matrix = self.get_data(data)
        # registration model
        affine_params_32, affine_params_64, affine_params_128, affine_matrix_32, affine_matrix_64, affine_matrix_128, moved_affine_32, moved_affine_64, moved_affine_128, moved_label_affine_32, moved_label_affine_64, moved_label_affine_128, predicted_deformation_field_32, predicted_deformation_field_64, predicted_deformation_field_128, moved_deformable_32, moved_deformable_64, moved_deformable_128, moved_label_deformable_32, moved_label_deformable_64, moved_label_deformable_128 = self.size_128_registration_network(tf.concat([fixed, moving, moving_label], axis=-1), training=False)
        registration_loss = total_loss_reg_128(fixed, fixed_label, true_affine_params, true_affine_matrix, affine_params_32, affine_params_64, affine_params_128, affine_matrix_32, affine_matrix_64, affine_matrix_128, moved_affine_32, moved_affine_64, moved_affine_128, moved_label_affine_32, moved_label_affine_64, moved_label_affine_128, predicted_deformation_field_32, predicted_deformation_field_64, predicted_deformation_field_128, moved_deformable_32, moved_deformable_64, moved_deformable_128, moved_label_deformable_32, moved_label_deformable_64, moved_label_deformable_128, mask_for_dice_loss_label, mask_for_NCC_deformable_loss_label)
        # segmentation model 1
        pred_seg_128_final_1 = self.size_128_segmentation_network_1(fixed, training=False)
        segmentation_loss_1 = total_loss_seg(fixed_label, pred_seg_128_final_1)
        # segmentation model 2
        pred_seg_128_final_2 = self.size_128_segmentation_network_2(tf.concat([fixed, pred_seg_128_final_1, moved_deformable_128, moved_label_deformable_128], axis=-1), training=False)
        segmentation_loss_2 = total_loss_seg(fixed_label, pred_seg_128_final_2)
        #
        return self.update_and_return_tracker_values(test_loss_trackers_list, registration_loss, segmentation_loss_1, segmentation_loss_2)
    #
    def update_and_return_tracker_values(self, tracker_list, registration_loss, segmentation_loss_1, segmentation_loss_2):
        #update tracker states
        [mean_tracker.update_state(loss_value) for mean_tracker,loss_value in zip(tracker_list, registration_loss + segmentation_loss_1 + segmentation_loss_2)]
        return {
                "Reg_Loss": tracker_list[0].result(),
                "MSE": tracker_list[1].result(),
                "Local_NCC_Affine": tracker_list[2].result(),
                "Global_NCC_Affine": tracker_list[3].result(),
                "Local_NCC_Deformable": tracker_list[4].result(),
                "Global_NCC_Deformable": tracker_list[5].result(),
                "Reg_Dice": tracker_list[6].result(),
                "Reg_CE": tracker_list[7].result(),
                "Deformation": tracker_list[8].result(),
                "Seg_Loss_1": tracker_list[9].result(),
                "Seg_Dice_1": tracker_list[10].result(),
                "Seg_CE_1": tracker_list[11].result(),
                "Seg_Loss_2": tracker_list[12].result(),
                "Seg_Dice_2": tracker_list[13].result(),
                "Seg_CE_2": tracker_list[14].result()
                }
    #
    @property
    def metrics(self):
        # `reset_states()` at beginning of each epoch
        return train_loss_trackers_list + test_loss_trackers_list

#load in initial data
if train_32 == True:
    params_dict['input_image_names'] = ['T1_32.nii.gz']*2
    params_dict['ground_truth_label_names'] = ['label_32.nii.gz']*2
    params_dict['input_segmentation_masks'] = False
    params_dict['patch_size'] = [32,32,32]
    params_dict['batch_size'] = [32,32,32]
elif train_64 == True:
    params_dict['input_image_names'] = ['T1_64.nii.gz']*2
    params_dict['ground_truth_label_names'] = ['label_64.nii.gz']*2
    params_dict['input_segmentation_masks'] = False
    params_dict['patch_size'] = [64,64,64]
    params_dict['batch_size'] = [2,2,2]
elif train_128 == True:
    params_dict['input_image_names'] = ['T1_128.nii.gz']*2
    params_dict['ground_truth_label_names'] = ['label_128.nii.gz']*2
    params_dict['input_segmentation_masks'] = False
    params_dict['patch_size'] = [128,128,128]
    params_dict['batch_size'] = [1,1,1]
#
batch_size = params_dict['batch_size'][0]
#
train_patients = nested_folder_filepaths(params_dict['data_dir_train'], [params_dict['input_image_names'], params_dict['ground_truth_label_names']])
train_generator = DataGenerator(params_dict['data_dir_train'], train_patients, params_dict['batch_size'][0], params_dict['num_patches_per_patient'][0], params_dict['adaptive_full_image_patching'][0], 'train', params_dict, rng_index=0)
val_patients = nested_folder_filepaths(params_dict['data_dir_val'], [params_dict['input_image_names'], params_dict['ground_truth_label_names']])
val_generator = DataGenerator(params_dict['data_dir_val'], val_patients, params_dict['batch_size'][1], params_dict['num_patches_per_patient'][1], params_dict['adaptive_full_image_patching'][1], 'train', params_dict, rng_index=1)

iterations_per_epoch = np.ceil((len(train_patients) * params_dict['num_patches_per_patient'][0]) / params_dict['batch_size'][0])
#
#load tensorboard callback and specific learning rate logger
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=params_dict['tensorboard_dir'], write_graph=True, histogram_freq=1, profile_batch=0)
file_writer = tf.summary.create_file_writer(params_dict['tensorboard_dir'] + '/learning_rate')
file_writer.set_as_default()
callbacks = [tensorboard_callback]
#load in optimizer
if params_dict['custom_fit'][1] == True:
    exclude_from_weight_decay = None
else:
    exclude_from_weight_decay = ['normalization','bias']
try:
    optimizer1 = getattr(tf.keras.optimizers, params_dict['optimizer1'][0])(*params_dict['optimizer1'][1:], exclude_from_weight_decay=exclude_from_weight_decay)
    optimizer2 = getattr(tf.keras.optimizers, params_dict['optimizer2'][0])(*params_dict['optimizer2'][1:], exclude_from_weight_decay=exclude_from_weight_decay)
    optimizer3 = getattr(tf.keras.optimizers, params_dict['optimizer3'][0])(*params_dict['optimizer3'][1:], exclude_from_weight_decay=exclude_from_weight_decay)
except:
    optimizer1 = getattr(tfa_optimizers, params_dict['optimizer1'][0])(*params_dict['optimizer1'][1:], exclude_from_weight_decay=exclude_from_weight_decay)
    optimizer2 = getattr(tfa_optimizers, params_dict['optimizer2'][0])(*params_dict['optimizer2'][1:], exclude_from_weight_decay=exclude_from_weight_decay)
    optimizer3 = getattr(tfa_optimizers, params_dict['optimizer3'][0])(*params_dict['optimizer3'][1:], exclude_from_weight_decay=exclude_from_weight_decay)
if params_dict['mixed_precision'] == True:
    optimizer1 = tf.keras.mixed_precision.LossScaleOptimizer(optimizer1)
    optimizer2 = tf.keras.mixed_precision.LossScaleOptimizer(optimizer2)
    optimizer3 = tf.keras.mixed_precision.LossScaleOptimizer(optimizer3)
#load in learning rate schedule if specified
if params_dict['lr_schedule_type'] != 'None':
    learning_rate_schedule_callback1 = CustomLearningRateSchedules(params_dict, params_dict['learning_rate'][0], params_dict['weight_decay'][0], iterations_per_epoch, optimizer1)
    learning_rate_schedule_callback2 = CustomLearningRateSchedules(params_dict, params_dict['learning_rate'][1], params_dict['weight_decay'][1], iterations_per_epoch, optimizer2)
    learning_rate_schedule_callback3 = CustomLearningRateSchedules(params_dict, params_dict['learning_rate'][2], params_dict['weight_decay'][2], iterations_per_epoch, optimizer3)
    callbacks.append(learning_rate_schedule_callback1)
    callbacks.append(learning_rate_schedule_callback2)
    callbacks.append(learning_rate_schedule_callback3)
#
inputs = tf.keras.Input(shape=(None,None,None,32), batch_size=batch_size, name='fixed_moving_shared_encoder_1_32', dtype=tf.float16 if params_dict['mixed_precision'] == True else tf.float32)
outputs = shared_encoder_1()(inputs)
shared_encoder_1_model = tf.keras.Model(inputs=inputs, outputs=outputs)
#
inputs = tf.keras.Input(shape=(None,None,None,64), batch_size=batch_size, name='fixed_moving_shared_encoder_2_32', dtype=tf.float16 if params_dict['mixed_precision'] == True else tf.float32)
outputs = shared_encoder_2()(inputs)
shared_encoder_2_model = tf.keras.Model(inputs=inputs, outputs=outputs)
#
inputs = tf.keras.Input(shape=(32,32,32,3), batch_size=batch_size, name='fixed_moving_registration_32', dtype=tf.float16 if params_dict['mixed_precision'] == True else tf.float32)
outputs = size_32_registration_network(shared_encoder_1_model, shared_encoder_2_model, batch_size, params_dict['mixed_precision'])(inputs)
model32_1 = tf.keras.Model(inputs=inputs, outputs=outputs)
#
inputs = tf.keras.Input(shape=(32,32,32,1), batch_size=batch_size, name='fixed_moving_segmentation_1_32', dtype=tf.float16 if params_dict['mixed_precision'] == True else tf.float32)
outputs = size_32_segmentation_network_1(shared_encoder_1_model, shared_encoder_2_model, params_dict['mixed_precision'])(inputs)
model32_2 = tf.keras.Model(inputs=inputs, outputs=outputs)
#
inputs = tf.keras.Input(shape=(32,32,32,4), batch_size=batch_size, name='fixed_moving_segmentation_2_32', dtype=tf.float16 if params_dict['mixed_precision'] == True else tf.float32)
outputs = size_32_segmentation_network_2(params_dict['mixed_precision'])(inputs)
model32_3 = tf.keras.Model(inputs=inputs, outputs=outputs)
#
total_model_32 = Reg_Seg_32(model32_1, model32_2, model32_3, params_dict['mixed_precision'], ndims)
#print(total_model_32.summary(line_length=150))
total_model_32.compile(registration_optimizer=optimizer1, segmentation_1_optimizer=optimizer2, segmentation_2_optimizer=optimizer3)
params_dict['model_weights_save_path'] = params_dict['model_outputs_dir'] + 'tf_ckpts_32'
params_dict['output_file'] = params_dict['model_outputs_dir'] + 'logfile_32.txt'
ckpt = tf.train.Checkpoint(model=total_model_32, registration_optimizer=total_model_32.registration_optimizer, segmentation_1_optimizer=total_model_32.segmentation_1_optimizer, segmentation_2_optimizer=total_model_32.segmentation_2_optimizer)
manager = tf.train.CheckpointManager(ckpt, params_dict['model_weights_save_path'], max_to_keep=params_dict['max_number_checkpoints_keep'])
ckpt.restore(manager.latest_checkpoint)
#
#size 64
inputs = tf.keras.Input(shape=(64,64,64,3), batch_size=batch_size, name='fixed_moving_registration_64', dtype=tf.float16 if params_dict['mixed_precision'] == True else tf.float32)
outputs = size_64_registration_network(total_model_32, batch_size, params_dict['mixed_precision'])(inputs)
model64_1 = tf.keras.Model(inputs=inputs, outputs=outputs)
#
inputs = tf.keras.Input(shape=(64,64,64,1), batch_size=batch_size, name='fixed_moving_segmentation_1_64', dtype=tf.float16 if params_dict['mixed_precision'] == True else tf.float32)
outputs = size_64_segmentation_network_1(total_model_32, params_dict['mixed_precision'])(inputs)
model64_2 = tf.keras.Model(inputs=inputs, outputs=outputs)
#
inputs = tf.keras.Input(shape=(64,64,64,4), batch_size=batch_size, name='fixed_moving_segmentation_2_64', dtype=tf.float16 if params_dict['mixed_precision'] == True else tf.float32)
outputs = size_64_segmentation_network_2(total_model_32, params_dict['mixed_precision'])(inputs)
model64_3 = tf.keras.Model(inputs=inputs, outputs=outputs)
#
total_model_64 = Reg_Seg_64(model64_1, model64_2, model64_3, params_dict['mixed_precision'], ndims)
#print(total_model_64.summary(line_length=150))
total_model_64.compile(registration_optimizer=optimizer1, segmentation_1_optimizer=optimizer2, segmentation_2_optimizer=optimizer3)
params_dict['model_weights_save_path'] = params_dict['model_outputs_dir'] + 'tf_ckpts_64'
params_dict['output_file'] = params_dict['model_outputs_dir'] + 'logfile_64.txt'
ckpt = tf.train.Checkpoint(model=total_model_64, registration_optimizer=total_model_64.registration_optimizer, segmentation_1_optimizer=total_model_64.segmentation_1_optimizer, segmentation_2_optimizer=total_model_64.segmentation_2_optimizer)
manager = tf.train.CheckpointManager(ckpt, params_dict['model_weights_save_path'], max_to_keep=params_dict['max_number_checkpoints_keep'])
ckpt.restore(manager.latest_checkpoint)
#
#size 128
inputs = tf.keras.Input(shape=(128,128,128,3), batch_size=batch_size, name='fixed_moving_registration_128', dtype=tf.float16 if params_dict['mixed_precision'] == True else tf.float32)
outputs = size_128_registration_network(total_model_64, batch_size, params_dict['mixed_precision'])(inputs)
model128_1 = tf.keras.Model(inputs=inputs, outputs=outputs)
#
inputs = tf.keras.Input(shape=(128,128,128,1), batch_size=batch_size, name='fixed_moving_segmentation_1_128', dtype=tf.float16 if params_dict['mixed_precision'] == True else tf.float32)
outputs = size_128_segmentation_network_1(total_model_64, params_dict['mixed_precision'])(inputs)
model128_2 = tf.keras.Model(inputs=inputs, outputs=outputs)
#
inputs = tf.keras.Input(shape=(128,128,128,4), batch_size=batch_size, name='fixed_moving_segmentation_2_128', dtype=tf.float16 if params_dict['mixed_precision'] == True else tf.float32)
outputs = size_128_segmentation_network_2(total_model_64, params_dict['mixed_precision'])(inputs)
model128_3 = tf.keras.Model(inputs=inputs, outputs=outputs)
#
total_model_128 = Reg_Seg_128(model128_1, model128_2, model128_3, params_dict['mixed_precision'], ndims)
#print(total_model_128.summary(line_length=150))
total_model_128.compile(registration_optimizer=optimizer1, segmentation_1_optimizer=optimizer2, segmentation_2_optimizer=optimizer3)
params_dict['model_weights_save_path'] = params_dict['model_outputs_dir'] + 'tf_ckpts_128'
params_dict['output_file'] = params_dict['model_outputs_dir'] + 'logfile_128.txt'
ckpt = tf.train.Checkpoint(model=total_model_128, registration_optimizer=total_model_128.registration_optimizer, segmentation_1_optimizer=total_model_128.segmentation_1_optimizer, segmentation_2_optimizer=total_model_128.segmentation_2_optimizer)
manager = tf.train.CheckpointManager(ckpt, params_dict['model_weights_save_path'], max_to_keep=params_dict['max_number_checkpoints_keep'])
#ckpt.restore(manager.latest_checkpoint)
#load in saved model callback
(validation_data_x, validation_data_y) = val_generator.__getitem__(0, 123456)
save_model_callback = SaveModelCallback(manager, params_dict, validation_data_x, validation_data_y, save_model_every_n_epochs=[True, 50])
callbacks.append(save_model_callback)
#make model image
#save_model_image(params_dict)
#train model
if train_32 == True:
    history = total_model_32.fit(x=train_generator, epochs=params_dict['num_epochs'], validation_data=val_generator, callbacks=callbacks, workers=params_dict['workers'], max_queue_size=params_dict['max_queue_size'], verbose=params_dict['verbose'], shuffle=False, use_multiprocessing=False)
elif train_64 == True:
    history = total_model_64.fit(x=train_generator, epochs=params_dict['num_epochs'], validation_data=val_generator, callbacks=callbacks, workers=params_dict['workers'], max_queue_size=params_dict['max_queue_size'], verbose=params_dict['verbose'], shuffle=False, use_multiprocessing=False)
elif train_128 == True:
    history = total_model_128.fit(x=train_generator, epochs=params_dict['num_epochs'], validation_data=val_generator, callbacks=callbacks, workers=params_dict['workers'], max_queue_size=params_dict['max_queue_size'], verbose=params_dict['verbose'], shuffle=False, use_multiprocessing=False)
