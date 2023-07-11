# import matplotlib.pyplot as plt
# import numpy as np
#
# import pyvista as pv
# from pyvista import examples
#
#
# mesh = examples.load_channels()
# # define a categorical colormap
# cmap = plt.cm.get_cmap("viridis", 4)
#
# mesh.plot(cmap=cmap)
#
#
#
# slices = mesh.slice_orthogonal()
#
# slices.plot(cmap=cmap)
#
# slices = mesh.slice_orthogonal(x=20, y=20, z=30)
# slices.plot(cmap=cmap)
#
# # Single slice - origin defaults to the center of the mesh
# single_slice = mesh.slice(normal=[1, 1, 0])
#
# p = pv.Plotter()
# p.add_mesh(mesh.outline(), color="k")
# p.add_mesh(single_slice, cmap=cmap)
# p.show()
#
#


