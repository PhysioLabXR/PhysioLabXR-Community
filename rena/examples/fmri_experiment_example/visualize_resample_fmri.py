import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img

# Load the MRI and fMRI files
mri_file = 'structural.nii.gz'
fmri_file = 'fmri.nii.gz'

mri_img = nib.load(mri_file)
fmri_img = nib.load(fmri_file)






# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # Generate a 3D volume (example data)
# volume_shape = (50, 50, 50)
# volume = np.random.rand(*volume_shape)
#
# # Define the plane parameters
# normal = np.array([1, 1, 1])  # Normal vector of the plane
# point = np.array([25, 25, 25])  # Point on the plane
#
# # Create a meshgrid for the volume
# x, y, z = np.meshgrid(np.arange(volume_shape[0]), np.arange(volume_shape[1]), np.arange(volume_shape[2]))
#
# # Calculate the signed distances from each point in the meshgrid to the plane
# distances = np.dot(np.array([x.flatten(), y.flatten(), z.flatten()]).T - point, normal)
#
# # Reshape the distances to match the volume shape
# distances = distances.reshape(volume_shape)
#
# # Cut the volume with the plane by setting values outside the plane to 0
# cut_volume = np.where(distances >= 0, volume, 0)
#
# # Visualize the cut volume
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.voxels(cut_volume, edgecolor='k')
# plt.show()
