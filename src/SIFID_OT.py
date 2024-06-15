#!/usr/bin/env python3
"""Calculates ***Single Image*** Frechet Inception Distance (SIFID) to evalulate Single-Image-GANs
Code was adapted from:
https://github.com/mseitzer/pytorch-fid.git
Which was adapted from the TensorFlow implementation of:

                                 
 https://github.com/bioinf-jku/TTUR

The FID metric calculates the distance between two distributions of images.
The SIFID calculates the distance between the distribution of deep features of a single real image and a single fake image.
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.                                                               
"""

import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
#from scipy.misc import imread
from matplotlib.pyplot import imread
from torch.nn.functional import adaptive_avg_pool2d
import ot

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from inception import InceptionV3
import torchvision
import numpy
import scipy
import pickle

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--path2real', type=str, help=('Path to the real images'))
parser.add_argument('--path2fake', type=str, help=('Path to generated images'))
parser.add_argument('-c', '--gpu', default='', type=str, help='GPU to use (leave blank for CPU only)')
parser.add_argument('--images_suffix', default='jpg', type=str, help='image file suffix')


def get_activations(files, model):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    # images = np.array([imread(str(f)).astype(np.float32)
    #                    for f in files[start:end]])
    images = imread(files).astype(np.float32)
    images = images.reshape(1, images.shape[0], images.shape[1], images.shape[2])
    

    images = images[:,:,:,0:3]
    # Reshape to (n_images, 3, height, width)
    images = images.transpose((0, 3, 1, 2))
    #images = images[0,:,:,:]
    images /= 255

    batch = torch.from_numpy(images).type(torch.FloatTensor)

    pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.

    #if pred.shape[2] != 1 or pred.shape[3] != 1:
    #    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    pred_arr = pred.cpu().data.numpy().transpose(0, 2, 3, 1).reshape(pred.shape[2]*pred.shape[3],-1)

    return pred_arr

def calculate_OT(act_real, act_generated):
    M = ot.dist(act_real, act_generated, 'euclidean')
    n_samples = act_real.shape[0]
    a, b = np.ones((n_samples,)) / n_samples, np.ones((n_samples,)) / n_samples  # uniform distribution on samples
    emd = ot.emd2(a, b, M)
    print(emd)
    return emd


def calculate_sifid_given_paths(path1, path2, cuda, dims):
    """Calculates the SIFID of two paths"""

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    files1 = [path1]
    files2 = [os.path.join(path2, file) for file in os.listdir(path2) if os.path.isfile(os.path.join(path2, file))]

    fid_values = []
    
    for i in tqdm(range(len(files2))):
        act_real = get_activations(files1[i], model)
        act_generated = get_activations(files2[i], model)
        fid_values.append(calculate_OT(act_real, act_generated))
    return fid_values


if __name__ == '__main__':
    args = parser.parse_args()

    path1 = "/home/infres/jsun-22/Documents/SinGAN/img/texture.png"
    path2 = "/home/infres/jsun-22/Documents/SinGAN/RandomSamples/texture.png/scale_factor=0.750000,alpha=10"

    sifid_values = calculate_sifid_given_paths(path1,path2,None,64)

    sifid_values = np.asarray(sifid_values,dtype=np.float32)
    # numpy.save('SIFID', sifid_values)
    print('SIFID: ', sifid_values.mean())