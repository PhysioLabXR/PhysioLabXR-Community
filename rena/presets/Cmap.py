from enum import Enum
import numpy as np
import pyqtgraph as pg

class Cmap(Enum):
    VIRIDIS = 0
    GRAY = 1
    HOT = 2
    PLASMA = 3
    SPRING = 4
    SUMMER = 5
    AUTUMN = 6
    WINTER = 7
    COOL = 8

    def get_lookup_table(self):
        # pos = np.array([0.0, 0.5, 1.0])  # absolute scale here relative to the expected data not important I believe
        # color = np.array([[255, 0, 0, 255], [255, 255, 0, 255], [0, 255, 0, 255]], dtype=np.ubyte)
        # colmap = pg.ColorMap(pos, color)
        # lut = colmap.getLookupTable(0, 1.0, 2000)

        if self == Cmap.VIRIDIS:
            pos = np.array([0.0, 0.5, 1.0])
            color = np.array([[68, 1, 84, 255], [26, 152, 80, 255], [253, 231, 37, 255]], dtype=np.ubyte)
        elif self == Cmap.COOL:
            pos = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            color = np.array([[0, 255, 255, 255],
                              [0, 191, 255, 255],
                              [0, 128, 255, 255],
                              [0, 64, 255, 255],
                              [0, 0, 255, 255],
                              [0, 0, 127, 255]], dtype=np.ubyte)

        elif self == Cmap.GRAY:
            pos = np.array([0.0, 1.0])
            color = np.array([[0, 0, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte)

        elif self == Cmap.HOT:
            pos = np.array([0.0, 0.5, 1.0])
            color = np.array([[0, 0, 0, 255], [255, 0, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte)

        elif self == Cmap.PLASMA:
            pos = np.array([0.0, 0.5, 1.0])
            color = np.array([[12, 7, 134, 255], [247, 252, 253, 255], [255, 0, 255, 255]], dtype=np.ubyte)

        elif self == Cmap.SPRING:
            pos = np.array([0.0, 1.0])
            color = np.array([[255, 0, 255, 255], [0, 255, 127, 255]], dtype=np.ubyte)

        elif self == Cmap.SUMMER:
            pos = np.array([0.0, 1.0])
            color = np.array([[255, 255, 102, 255], [0, 255, 127, 255]], dtype=np.ubyte)
        elif self == Cmap.AUTUMN:
            pos = np.array([0.0, 1.0])
            color = np.array([[255, 0, 0, 255], [255, 255, 0, 255]], dtype=np.ubyte)
        elif self == Cmap.WINTER:
            pos = np.array([0.0, 0.5, 1.0])
            color = np.array([[0, 0, 255, 255], [0, 255, 255, 255], [255, 255, 255, 255]], dtype=np.ubyte)
        else:
            raise ValueError('Unknown colormap')
        return pg.ColorMap(pos, color).getLookupTable(0, 1.0, 2000)