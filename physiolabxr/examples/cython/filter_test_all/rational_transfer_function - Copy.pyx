# tag: numpy
# You can ignore the previous line.
# It's for internal testing of the cython documentation.
import cython
import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
np.import_array()

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float32_t DTYPE_t


def process_sample_vanilla(x_tap, y_tap, a, b, sample):
    # perform realtime filter with tap

    # push x
    x_tap[:, 1:] = x_tap[:, : -1]
    x_tap[:, 0] = sample
    # push y
    y_tap[:, 1:] = y_tap[:, : -1]
    # calculate new y
    y_tap[:, 0] = np.sum(np.multiply(x_tap, b), axis=1) - \
                       np.sum(np.multiply(y_tap[:, 1:], a[1:]), axis=1)

    sample = y_tap[:, 0]
    return sample

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def process_sample_cython(np.ndarray[DTYPE_t, ndim=2] x_tap, np.ndarray[DTYPE_t, ndim=2] y_tap,
                          np.ndarray[DTYPE_t, ndim=1] a, np.ndarray[DTYPE_t, ndim=1] b, np.ndarray[DTYPE_t, ndim=1] sample):
    # perform realtime filter with tap

    # push x
    x_tap[:, 1:] = x_tap[:, : -1]
    x_tap[:, 0] = sample
    # push y
    y_tap[:, 1:] = y_tap[:, : -1]
    # calculate new y
    y_tap[:, 0] = np.sum(np.multiply(x_tap, b), axis=1) - \
                       np.sum(np.multiply(y_tap[:, 1:], a[1:]), axis=1)

    sample = y_tap[:, 0]
    return sample

# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)
# def cython_process_buffer():
