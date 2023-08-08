# x_size, y_size, z_size = self.volume_data.shape
# self.coronal_view_slider.setRange(minValue=0, maxValue=x_size - 1)
# self.sagittal_view_slider.setRange(minValue=0, maxValue=y_size - 1)
# self.axial_view_slider.setRange(minValue=0, maxValue=z_size - 1)
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import functions as fn
import nibabel as nib
from nilearn.image import resample_img
from nilearn import plotting
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def get_mri_coronal_view_dimension(volume_data: np.ndarray):
    return volume_data.shape[0]


def get_mri_coronal_view_slice(volume_data, index):
    return volume_data[index, :, :]


def get_fmri_coronal_view_slice(volume_data, index, timestamp):
    return volume_data[index, :, :, timestamp]


def get_mri_sagittal_view_dimension(volume_data: np.ndarray) -> int:
    return volume_data.shape[1]


def get_mri_sagittal_view_slice(volume_data, index):
    return volume_data[:, index, :]


def get_fmri_sagittal_view_slice(volume_data, index, timestamp):
    return volume_data[:, index, :, timestamp]


def get_mri_axial_view_dimension(volume_data: np.ndarray) -> int:
    return volume_data.shape[2]


def get_mri_axial_view_slice(volume_data, index):
    return volume_data[:, :, index]


def get_fmri_axial_view_slice(volume_data, index, timestamp):
    return volume_data[:, :, index, timestamp]


def load_nii_gz_file(file_path: str, normalized=True, zoomed=False):
    nifti_image = nib.load(file_path)

    # Access the image data and header
    image_data = nifti_image.get_fdata()
    image_header = nifti_image.header

    # Convert the image data to a NumPy array
    # volume_data = np.array(image_data)

    if normalized:
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    if zoomed:
        scale_factor = nifti_image.header.get_zooms()

        # Rescale the matrix using zoom
        image_data = zoom(image_data, scale_factor)

    return image_header, image_data


def volume_to_gl_volume_item(volume_data, alpha_interpolate=True, centralized=True, non_linear_interpolation_factor=1):
    volume_data = (volume_data - np.min(volume_data)) / (np.max(volume_data) - np.min(volume_data)) * non_linear_interpolation_factor # TODO: change the interpolation range
    x_size, y_size, z_size = volume_data.shape

    # cut off the top half data
    volume_data = volume_data[:, :, : 105]

    if alpha_interpolate:
        alpha_channel = np.interp(volume_data, (0, 1), (0, 255))
    else:
        alpha_channel = 225

    # Add an alpha channel to the volume data
    volume_data_rgba = np.zeros(volume_data.shape + (4,), dtype=np.ubyte)
    volume_data_rgba[..., 0] = volume_data * 255  # R channel
    volume_data_rgba[..., 1] = volume_data * 255  # G channel
    volume_data_rgba[..., 2] = volume_data * 255  # B channel
    volume_data_rgba[..., 3] = alpha_channel  # Alpha channel

    # Get the volume dimensions

    v = gl.GLVolumeItem(volume_data_rgba)

    # Shift the center of the volume to (0, 0, 0)
    if centralized:
        v.translate(-x_size / 2, -y_size / 2, -z_size / 2)
    return v


def gray_to_heatmap(image, threshold=1.0):
    cmap = plt.cm.hot  # Choose the colormap for the heatmap (you can change it as needed)

    # Apply thresholding
    masked_image = np.ma.masked_less(image, threshold)
    heatmap = cmap(masked_image)

    return heatmap


if __name__ == '__main__':
    _, mri_data = load_nii_gz_file(file_path='structural.nii.gz')
    _, fmri_data = load_nii_gz_file(file_path='resampled_fmri.nii.gz')

    print("finished loading")
    # pass
    # mri_data = nib.load('structural.nii.gz')
    # fmri_data = nib.load('fmri.nii.gz')
    # reference_image = mri_data
    # # resampled_fmri = resample_to_img(stat_img, template)
    # resampled_fmri = resample_img(fmri_data, target_affine=reference_image.affine,
    #                               target_shape=reference_image.shape[:3])
    #
    #
    #
    # fmri_t = resampled_fmri.get_fdata()[:,:,:, 50]
    # fmri_slice = fmri_t[:,:, 100]
    # plt.imshow(fmri_slice, cmap='hot', alpha=0.5)
    # plt.show()
    # plotting.plot_stat_map(
    #     resampled_fmri,
    #     bg_img=mri_data,
    #     cut_coords=(36, -27, 66),
    #     threshold=3,
    #     title="Resampled t-map",
    # )
    # plotting.show()
