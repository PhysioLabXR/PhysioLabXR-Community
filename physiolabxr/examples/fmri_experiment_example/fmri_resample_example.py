import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img

# Load the MRI and fMRI files
mri_file = 'structural.nii.gz'
fmri_file = 'fmri.nii.gz'

mri_img = nib.load(mri_file)
fmri_img = nib.load(fmri_file)
mri_img.header.get_zooms()
# Resample the fMRI data to match the MRI data's affine and shape
fmri_resampled = resample_to_img(fmri_img, mri_img, interpolation='linear')

# Save the resampled fMRI data
nib.save(fmri_resampled, 'resampled_fmri.nii.gz')



