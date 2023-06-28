import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import functions as fn

app = pg.mkQApp("GLVolumeItem Example")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('pyqtgraph example: GLVolumeItem')
w.setCameraPosition(distance=200)

g = gl.GLGridItem()
g.scale(10, 10, 100)
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


data = np.fromfunction(psi, (100,100,200))
with np.errstate(divide = 'ignore'):
    positive = np.log(fn.clip_array(data, 0, data.max())**2)
    negative = np.log(fn.clip_array(-data, 0, -data.min())**2)

d2 = np.empty(data.shape + (4,), dtype=np.ubyte)

# Original Code
# d2[..., 0] = positive * (255./positive.max())
# d2[..., 1] = negative * (255./negative.max())

# Reformulated Code
# Both positive.max() and negative.max() are negative-valued.
# Thus the next 2 lines are _not_ bounded to [0, 255]
positive = positive * (255./positive.max())
negative = negative * (255./negative.max())
# When casting to ubyte, the original code relied on +Inf to be
# converted to 0. On arm64, it gets converted to 255.
# Thus the next 2 lines change +Inf explicitly to 0 instead.
positive[np.isinf(positive)] = 0
negative[np.isinf(negative)] = 0
# When casting to ubyte, the original code relied on the conversion
# to do modulo 256. The next 2 lines do it explicitly instead as
# documentation.
d2[..., 0] = positive.astype(int) % 256
d2[..., 1] = negative.astype(int) % 256

d2[..., 2] = d2[...,1]
d2[..., 3] = d2[..., 0]*0.3 + d2[..., 1]*0.3
d2[..., 3] = (d2[..., 3].astype(float) / 255.) **2 * 255

d2[:, 0, 0] = [255,0,0,100]
d2[0, :, 0] = [0,255,0,100]
d2[0, 0, :] = [0,0,255,100]

v = gl.GLVolumeItem(d2)
v.translate(-50,-50,-100)
w.addItem(v)

ax = gl.GLAxisItem()
w.addItem(ax)

if __name__ == '__main__':
    pg.exec()


import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
from PyQt6.QtGui import QPainter
from PyQt6.QtCore import Qt
from OpenGL.GL import *


class MyGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def initializeGL(self):
        glClearColor(0.2, 0.2, 0.2, 1.0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)

        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_TRIANGLES)
        glVertex2f(-0.5, -0.5)
        glVertex2f(0.5, -0.5)
        glVertex2f(0.0, 0.5)
        glEnd()

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt OpenGL Example")
        self.setGeometry(100, 100, 400, 400)

        gl_widget = MyGLWidget(self)
        self.setCentralWidget(gl_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


# import sys
# import numpy as np
# from OpenGL.raw.GLU import gluPerspective, gluLookAt
# from PyQt6.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
# from PyQt6.QtGui import QPainter
# from PyQt6.QtCore import Qt
# from OpenGL.GL import *
#
#
# class MyGLWidget(QOpenGLWidget):
#     def __init__(self, volume_data, parent=None):
#         super().__init__(parent)
#         self.volume_data = volume_data
#
#     def initializeGL(self):
#         glClearColor(0.2, 0.2, 0.2, 1.0)
#
#         glEnable(GL_TEXTURE_3D)
#         glBindTexture(GL_TEXTURE_3D, glGenTextures(1))
#
#         glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
#         glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
#         glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
#         glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
#         glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
#
#         glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, self.volume_data.shape[0],
#                      self.volume_data.shape[1], self.volume_data.shape[2], 0, GL_RED, GL_UNSIGNED_BYTE,
#                      self.volume_data.tobytes())
#
#     def paintGL(self):
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#         glEnable(GL_DEPTH_TEST)
#
#         glMatrixMode(GL_PROJECTION)
#         glLoadIdentity()
#         gluPerspective(45, self.width() / self.height(), 0.1, 100.0)
#
#         glMatrixMode(GL_MODELVIEW)
#         glLoadIdentity()
#         gluLookAt(0, 0, -5, 0, 0, 0, 0, 1, 0)
#
#         # Render slices of the volume
#         slice_spacing = 2.0 / self.volume_data.shape[2]
#         for z in range(self.volume_data.shape[2]):
#             glBindTexture(GL_TEXTURE_3D, 1)
#             glBegin(GL_QUADS)
#             glTexCoord3f(0, 0, z * slice_spacing)
#             glVertex3f(-1, -1, z * slice_spacing)
#
#             glTexCoord3f(1, 0, z * slice_spacing)
#             glVertex3f(1, -1, z * slice_spacing)
#
#             glTexCoord3f(1, 1, z * slice_spacing)
#             glVertex3f(1, 1, z * slice_spacing)
#
#             glTexCoord3f(0, 1, z * slice_spacing)
#             glVertex3f(-1, 1, z * slice_spacing)
#             glEnd()
#
#     def resizeGL(self, width, height):
#         glViewport(0, 0, width, height)
#
#
# class MainWindow(QMainWindow):
#     def __init__(self, volume_data):
#         super().__init__()
#
#         self.setWindowTitle("PyQt OpenGL Volume Example")
#         self.setGeometry(100, 100, 400, 400)
#
#         gl_widget = MyGLWidget(volume_data, self)
#         self.setCentralWidget(gl_widget)
#
#
# if __name__ == '__main__':
#     # Generate sample volume data (random noise)
#     volume_data = np.random.randint(0, 256, size=(32, 32, 32), dtype=np.uint8)
#
#     app = QApplication(sys.argv)
#     window = MainWindow(volume_data)
#     window.show()
