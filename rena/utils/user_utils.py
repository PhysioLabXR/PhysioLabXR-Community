import os
import pickle
from pathlib import Path
import scipy.io

from rena.utils.RNStream import RNStream
from rena.utils.data_utils import CsvStoreLoad
from rena.utils.xdf_utils import load_xdf


def stream_in(file_path):
    '''

    :param file_path:
    :return:
    '''
    file_path_obj = Path(file_path)
    if file_path_obj.is_dir():
        csv_loader = CsvStoreLoad()
        return csv_loader.load_csv(file_path)
    else:
        file_extension = file_path_obj.suffix
        if file_extension == '.dats':
            stream = RNStream(file_path)
            return stream.stream_in()
        elif file_extension == '.mat':
            return scipy.io.loadmat(file_path)
        elif file_extension == '.pickle' or file_extension == '.pkl' or file_extension == '.p':
            return pickle.load(open(file_path, 'rb'))
        elif file_extension == '.xdf':
            return load_xdf(file_path)