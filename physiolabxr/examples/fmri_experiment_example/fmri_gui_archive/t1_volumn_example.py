import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import functions as fn

from physiolabxr.examples.fmri_experiment_example.mri_utils import load_nii_gz_file, gray_to_heatmap, get_mri_axial_view_slice

app = pg.mkQApp("GLVolumeItem Example")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('pyqtgraph example: GLVolumeItem')
w.setCameraPosition(distance=200)

g = gl.GLGridItem()
g.scale(100, 100, 100)
w.addItem(g)

## Hydrogen electron probability density
def psi(i, j, k, offset=(50,50,100)):
    x = i-offset[0]
    y = j-offset[1]
    z = k-offset[2]
    th = np.arctan2(z, np.hypot(x, y))
    r = np.sqrt(x**2 + y**2 + z **2)
    a0 = 2
    return (
        (1.0 / 81.0)
        * 1.0 / (6.0 * np.pi) ** 0.5
        * (1.0 / a0) ** (3 / 2)
        * (r / a0) ** 2
        * np.exp(-r / (3 * a0))
        * (3 * np.cos(th) ** 2 - 1)
    )


# data = np.fromfunction(psi, (100,100,200))
_, volume_data = load_nii_gz_file(file_path='structural_brain.nii.gz')
# volume_data = volume_data[:, :, :, 100]
# with np.errstate(divide = 'ignore'):
#     positive = np.log(fn.clip_array(data, 0, data.max())**2)
#     negative = np.log(fn.clip_array(-data, 0, -data.min())**2)
#
# d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
#
# # Original Code
# # d2[..., 0] = positive * (255./positive.max())
# # d2[..., 1] = negative * (255./negative.max())
#
# # Reformulated Code
# # Both positive.max() and negative.max() are negative-valued.
# # Thus the next 2 lines are _not_ bounded to [0, 255]
# positive = positive * (255./positive.max())
# negative = negative * (255./negative.max())
# # When casting to ubyte, the original code relied on +Inf to be
# # converted to 0. On arm64, it gets converted to 255.
# # Thus the next 2 lines change +Inf explicitly to 0 instead.
# positive[np.isinf(positive)] = 0
# negative[np.isinf(negative)] = 0
# # When casting to ubyte, the original code relied on the conversion
# # to do modulo 256. The next 2 lines do it explicitly instead as
# # documentation.
# d2[..., 0] = positive.astype(int) % 256
# d2[..., 1] = negative.astype(int) % 256
#
# d2[..., 2] = d2[...,1]
# d2[..., 3] = d2[..., 0]*0.3 + d2[..., 1]*0.3
# d2[..., 3] = (d2[..., 3].astype(float) / 255.) **2 * 255
#
# d2[:, 0, 0] = [255,0,0,100]
# d2[0, :, 0] = [0,255,0,100]
# d2[0, 0, :] = [0,0,255,100]

# shape = (50, 50, 50)
# volume_data = np.random.randint(0, 255, shape + (4,), dtype=np.ubyte)


volume_data = (volume_data - np.min(volume_data)) / (np.max(volume_data) - np.min(volume_data))
alpha_channel = np.interp(volume_data, (0, 1), (0, 255))

# Add an alpha channel to the volume data
volume_data_rgba = np.zeros(volume_data.shape + (4,), dtype=np.ubyte)
volume_data_rgba[..., 0] = volume_data * 255  # R channel
volume_data_rgba[..., 1] = volume_data * 255  # G channel
volume_data_rgba[..., 2] = volume_data * 255  # B channel
volume_data_rgba[..., 3] = alpha_channel  # Alpha channel

# Get the volume dimensions
x_size, y_size, z_size = volume_data.shape




v = gl.GLVolumeItem(volume_data_rgba)


# Shift the center of the volume to (0, 0, 0)
v.translate(-x_size/2, -y_size/2, -z_size/2)

# Set camera position and orientation
# view.setCameraPosition(distance=200)

# v.translate(-50,-50,-100)
w.addItem(v)
fmri_slice = get_mri_axial_view_slice(volume_data, 25)
image_data = (gray_to_heatmap(fmri_slice, threshold=0) * 255).astype(np.uint8)
# image_data = np.transpose(image_data, (1, 0, 2))
fmri_axial_view_image_item = gl.GLImageItem(image_data, smooth=False, glOptions='translucent')
#
# fmri_axial_view_image_item = gl.GLImageItem(image_data)  # np.zeros((256, 256, 4), dtype=np.uint8)
# apply the xz plane transform
fmri_axial_view_image_item.translate(-x_size / 2, -y_size / 2, 100)
w.addItem(fmri_axial_view_image_item)



ax = gl.GLAxisItem()
w.addItem(ax)

if __name__ == '__main__':
    pg.exec()

