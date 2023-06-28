import sys

import numpy as np
from PyQt6.QtWidgets import QApplication
from pyqtgraph.opengl import GLViewWidget, GLImageItem

# This Python file uses the following encoding: utf-8
# get_mri_coronal_view_dimension, get_mri_sagittal_view_dimension, \
#     get_mri_axial_view_dimension
# This Python file uses the following encoding: utf-8

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Create a GLViewWidget
    view = GLViewWidget()


    # Create a grayscale image (8-bit depth)
    # width, height, depth = 128, 128, 128
    # image_data = np.random.randint(0, 256, size=(width, height, depth, 4), dtype=np.ubyte)
    image_data = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
    image_data[:, :, 3] = 100

    # data = gray_to_heatmap()
    # Create a GLImageItem and assign the image data
    image_item = GLImageItem(image_data)
    # image_item = GLImageItem(data=image_data)
    # Add the image item to the view
    view.addItem(image_item)

    # Show the view
    view.show()

    sys.exit(app.exec_())

# import cv2
# import numpy as np
#
# # Image dimensions
# width = 100
# height = 100
#
# # Create a random RGBA image
# random_image = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
#
# # Set the alpha channel to maximum (255) for all pixels
# random_image[:, :, 3] = 100
#
# # Display the image
# cv2.imshow('Random Image', random_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()