import matplotlib.pyplot as plt
import numpy as np

import pyvista as pv
from pyvista import examples
import nilearn
import nibabel as nib


fmri_img = nib.load('structural.nii.gz')
volume = fmri_img.get_fdata()
print()

