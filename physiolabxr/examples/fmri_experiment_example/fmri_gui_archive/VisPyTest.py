import numpy as np

from vispy import io, plot as vp

fig = vp.Fig(bgcolor='k', size=(800, 800), show=False)

vol_data = np.load(io.load_data_file('brain/mri.npz'))['data']
vol_data = np.flipud(np.rollaxis(vol_data, 1))
vol_data = vol_data.astype(np.float32)

clim = [32, 192]
texture_format = "auto"  # None for CPUScaled, "auto" for GPUScaled

vol_pw = fig[0, 0]
v = vol_pw.volume(vol_data, clim=clim, texture_format=texture_format)
vol_pw.view.camera.elevation = 30
vol_pw.view.camera.azimuth = 30
vol_pw.view.camera.scale_factor /= 1.5

shape = vol_data.shape
fig[1, 0].image(vol_data[:, :, shape[2] // 2], cmap='grays', clim=clim,
                fg_color=(0.5, 0.5, 0.5, 1), texture_format=texture_format)
fig[0, 1].image(vol_data[:, shape[1] // 2, :], cmap='grays', clim=clim,
                fg_color=(0.5, 0.5, 0.5, 1), texture_format=texture_format)
fig[1, 1].image(vol_data[shape[0] // 2, :, :].T, cmap='grays', clim=clim,
                fg_color=(0.5, 0.5, 0.5, 1), texture_format=texture_format)
fig[1, 2].image(vol_data[shape[0] // 2, :, :].T, cmap='grays', clim=clim,
                fg_color=(0.5, 0.5, 0.5, 1), texture_format=texture_format)



if __name__ == '__main__':
    fig.show(run=True)
# #
# #
# # import numpy as np
# # from PyQt6 import QtWidgets
# #
# # from vispy.scene import SceneCanvas, visuals
# # from vispy.app import use_app
# #
# # IMAGE_SHAPE = (600, 800)  # (height, width)
# # CANVAS_SIZE = (800, 600)  # (width, height)
# # NUM_LINE_POINTS = 200
# #
# #
# # class MyMainWindow(QtWidgets.QMainWindow):
# #     def __init__(self, *args, **kwargs):
# #         super().__init__(*args, **kwargs)
# #
# #         central_widget = QtWidgets.QWidget()
# #         main_layout = QtWidgets.QHBoxLayout()
# #
# #         self._controls = Controls()
# #         main_layout.addWidget(self._controls)
# #         self._canvas_wrapper = CanvasWrapper()
# #         main_layout.addWidget(self._canvas_wrapper.canvas.native)
# #
# #         central_widget.setLayout(main_layout)
# #         self.setCentralWidget(central_widget)
# #
# #
# # class Controls(QtWidgets.QWidget):
# #     def __init__(self, parent=None):
# #         super().__init__(parent)
# #         layout = QtWidgets.QVBoxLayout()
# #         self.colormap_label = QtWidgets.QLabel("Image Colormap:")
# #         layout.addWidget(self.colormap_label)
# #         self.colormap_chooser = QtWidgets.QComboBox()
# #         self.colormap_chooser.addItems(["viridis", "reds", "blues"])
# #         layout.addWidget(self.colormap_chooser)
# #
# #         self.line_color_label = QtWidgets.QLabel("Line color:")
# #         layout.addWidget(self.line_color_label)
# #         self.line_color_chooser = QtWidgets.QComboBox()
# #         self.line_color_chooser.addItems(["black", "red", "blue"])
# #         layout.addWidget(self.line_color_chooser)
# #
# #         layout.addStretch(1)
# #         self.setLayout(layout)
# #
# #
# # class CanvasWrapper:
# #     def __init__(self):
# #         self.canvas = SceneCanvas(size=CANVAS_SIZE)
# #         self.grid = self.canvas.central_widget.add_grid()
# #
# #         self.view_top = self.grid.add_view(0, 0, bgcolor='cyan')
# #         image_data = _generate_random_image_data(IMAGE_SHAPE)
# #         self.image = visuals.Image(
# #             image_data,
# #             texture_format="auto",
# #             cmap="viridis",
# #             parent=self.view_top.scene,
# #         )
# #         self.view_top.camera = "panzoom"
# #         self.view_top.camera.set_range(x=(0, IMAGE_SHAPE[1]), y=(0, IMAGE_SHAPE[0]), margin=0)
# #
# #         self.view_bot = self.grid.add_view(1, 0, bgcolor='#c0c0c0')
# #         line_data = _generate_random_line_positions(NUM_LINE_POINTS)
# #         self.line = visuals.Line(line_data, parent=self.view_bot.scene, color='black')
# #         self.view_bot.camera = "panzoom"
# #         self.view_bot.camera.set_range(x=(0, NUM_LINE_POINTS), y=(0, 1))
# #
# #
# # def _generate_random_image_data(shape, dtype=np.float32):
# #     rng = np.random.default_rng()
# #     data = rng.random(shape, dtype=dtype)
# #     return data
# #
# #
# # def _generate_random_line_positions(num_points, dtype=np.float32):
# #     rng = np.random.default_rng()
# #     pos = np.empty((num_points, 2), dtype=np.float32)
# #     pos[:, 0] = np.arange(num_points)
# #     pos[:, 1] = rng.random((num_points,), dtype=dtype)
# #     return pos
# #
# #
# # if __name__ == "__main__":
# #     app = use_app("PyQt6")
# #     app.createuse_app()
# #     win = MyMainWindow()
# #     win.show()
# #     app.run()
#
#
# # class myWindow(QtWidgets.QMainWindow):
# #     def __init__(self):
# #         super(myWindow, self).__init__()
# #         self._ui = Ui_MainWindow()
# #         self._ui.setupUi(self)
# #
# #         canvas = vispy.app.Canvas()
# #         lay = QtWidgets.QVBoxLayout(self._ui.frameFor3d) # create layout
# #         lay.addWidget(canvas.native)
# #
# # if __name__ == '__main__':
# #     app = QtWidgets.QApplication([])
# #     application = myWindow()
# #     application.show()
# #     vispy.app.run()
# #     # sys.exit(app.exec())'
#
#
# import sys
# from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget
# from PyQt6.QtCore import QTimer
# from vispy import app, scene
#
# # Create a custom QWidget class for embedding the Vispy canvas
# class VispyWidget(QWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.canvas = scene.SceneCanvas(keys='interactive', show=False)
#         self.native = self.canvas.native
#         self.layout = QVBoxLayout()
#         self.layout.addWidget(self.native)
#         self.setLayout(self.layout)
#
#     def start(self):
#         self.canvas.show()
#         app.run()
#
#     def stop(self):
#         app.quit()
#
#     def paintEvent(self, event):
#         self.canvas.update()
#
# # Create a Vispy scene
# def create_scene():
#     canvas = scene.SceneCanvas(keys='interactive', show=False)
#     view = canvas.central_widget.add_view()
#     # Add your Vispy scene objects here
#     # e.g., visuals, lights, etc.
#
#     return canvas
#
# if __name__ == '__main__':
#     # Create the PyQt application
#     app = QApplication(sys.argv)
#
#     # Create the Vispy widget
#     vispy_widget = VispyWidget()
#
#     # Create the Vispy scene
#     vispy_canvas = create_scene()
#
#     # Set the Vispy canvas as a central widget in the Vispy widget
#     vispy_widget.canvas = vispy_canvas
#
#     # Create a QTimer to periodically refresh the Vispy canvas
#     timer = QTimer()
#     timer.timeout.connect(vispy_widget.update)
#     timer.start(16)  # Update at approximately 60 FPS
#
#     # Show the PyQt window containing the Vispy widget
#     vispy_widget.show()
#
#     # Start the event loop
#     sys.exit(app.exec_())
#
#     # Clean up
#     timer.stop()
#     vispy_widget.stop()