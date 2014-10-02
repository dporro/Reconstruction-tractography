# -*- coding: utf-8 -*-
"""
Created on Tue Oct 9 13:56:54 2012

@author: dianaporro
"""


import numpy as np
import nibabel as nib
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.io.dpy import Dpy
from dipy.data import get_sphere
from dipy.tracking.eudx import EuDX


def tractography(nii_filename, bval_filename, bvec_filename, seed, threshold):
    ''' Script to generate tractography. Uses the EuDX function from dipy. Returns tractography and FA.'''
    
    print "Loading data"
    img = nib.load(nii_filename)
    data = img.get_data()
    affine = img.get_affine()
    

    bvals = np.loadtxt(bval_filename)
    bvecs = np.loadtxt(bvec_filename).T # this is the unitary direction of the gradient


    #new version of dipy
    print "Computing tensor model"
    gradients = gradient_table(bvals,bvecs)
    tensor_model = dti.TensorModel(gradients)
    tensors = tensor_model.fit(data)

    print "Computing FA"
    FA = dti.fractional_anisotropy(tensors.evals)
    FA[np.isnan(FA)] = 0

    print "Computing evecs"
    evecs_img = nib.Nifti1Image(tensors.evecs.astype(np.float32), affine)
    evecs = evecs_img.get_data()

    sphere = get_sphere('symmetric724')
    peak_indices = dti.quantize_evecs(evecs, sphere.vertices)
    
    print "Computing EuDX reconstruction."
    streamlines = EuDX(FA.astype('f8'),
                        ind=peak_indices,
			seeds=seed,
                        odf_vertices= sphere.vertices,
                        a_low=threshold)
    return streamlines, FA


def save_trk(streamlines, voxel_size, dimensions, filename):
    '''Save tractography to a .trk file'''
    
    print "Save tracks as .trk"
    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] = voxel_size
    hdr['voxel_order'] = 'LAS'
    hdr['dim'] = dim
    strm = ((sl, None, None) for sl in streamlines)

    nib.trackvis.write(filename, strm, hdr, points_space='voxel')
    



