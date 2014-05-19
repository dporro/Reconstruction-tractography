# -*- coding: utf-8 -*-
"""
Created on Tue Oct 9 13:56:54 2012

@author: bao
"""


import os
import numpy as np
import nibabel as nib
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
#from dipy.reconst.gqi import GeneralizedQSampling
from dipy.io.dpy import Dpy
#this is new in dipy 0.6.0.dev
from dipy.data import get_sphere
from dipy.tracking.eudx import EuDX
import pdb


       
def create_save_tracks(anisotropy,indices, seed, sphere, low_thresh,filename):
    #this is new features in new dipy -current 121011 0.6.0.dev
    print "Computing EuDX reconstruction."
    euler = EuDX(anisotropy.astype('f8'),
                        ind=indices,
			seeds=seed,
                        odf_vertices= sphere.vertices,
                        a_low=low_thresh)
  
    
    print "Save tracks as .dpy"	
    tracks = [track for track in euler]
    pdb.set_trace()
    dpw = Dpy(filename, 'w')
    dpw.write_tracks(tracks)
    dpw.close()


    return euler

   
def tractography(directoryname):
    print "Loading data"
    dirname = directoryname
    filename = 'data'
    base_filename = dirname + filename

 
    nii_filename = base_filename + '.nii.gz'
    bvec_filename = dirname + 'bvecs'
    bval_filename = dirname + 'bvals'

    img = nib.load(nii_filename)
    data = img.get_data()
    affine = img.get_affine()
    

    bvals = np.loadtxt(bval_filename)
    bvecs = np.loadtxt(bvec_filename).T # this is the unitary direction of the gradient


    base_dir2 = dirname + 'DTI/'


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

    streamlines_10k = create_save_tracks(FA, peak_indices, 10**4, sphere, .1, base_dir2+'tracks_dti_10K.dpy')
    streamlines_3M = create_save_tracks(FA, peak_indices, 3*(10**6), sphere, .1, base_dir2+'tracks_dti_3M.dpy')

    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] = img.get_header().get_zooms()[:3]
    hdr['voxel_order'] = 'LAS'
    hdr['dim'] = FA.shape[:3]
    pdb.set_trace()
    streamlines_trk_10K = ((sl, None, None) for sl in streamlines_10k)
    sl_fname = base_dir2+'tracks_dti_10K.trk'

    nib.trackvis.write(sl_fname, streamlines_trk_10K, hdr, points_space='voxel')

    streamlines_trk_3M = ((sl, None, None) for sl in streamlines_3M)
    sl_fname = base_dir2+'tracks_dti_3M.trk'
    nib.trackvis.write(sl_fname, streamlines_trk_3M, hdr, points_space='voxel')

    mapfile = base_dir2+'FA_map.nii.gz'
    nib.save(nib.Nifti1Image(FA, img.get_affine()), mapfile)


#Calling the method for different subjects

#dirname = '/home/dporro/data/HCP_all/124422/T1w/Diffusion/'
dirname = "/home/dporro/data/Sarubbo/DTI_Niftii/"
tractography(dirname)


