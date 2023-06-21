import nibabel as nib
import numpy as np


# # Load the NIfTI image
# image_path = 'avg152T1_LR_nifti.nii.gz'
# nifti_image = nib.load(image_path)
#
# # Access the image data and header
# image_data = nifti_image.get_fdata()
# image_header = nifti_image.header
#
# # Convert the image data to a NumPy array
# numpy_array = np.array(image_data)
#
# # Now you can work with the NumPy array
# print(numpy_array.shape)  # Print the shape of the array


def load_nii_gz_file(file_path: str):
    nifti_image = nib.load(file_path)

    # Access the image data and header
    image_data = nifti_image.get_fdata()
    image_header = nifti_image.header

    # Convert the image data to a NumPy array
    numpy_array = np.array(image_data)

    return image_header, numpy_array
