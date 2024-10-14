
import nibabel as nib


fmri_img = nib.load('structural.nii.gz')
volume = fmri_img.header.get_zooms()
print(fmri_img)

# (0.9375, 0.9375, 1.5)