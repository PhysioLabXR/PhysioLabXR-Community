import os
import pickle
from pathlib import Path
import scipy.io

from physiolabxr.utils.RNStream import RNStream
from physiolabxr.utils.data_utils import CsvStoreLoad
from physiolabxr.utils.xdf_utils import load_xdf


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
        elif file_extension == '.mat' or file_extension == '.m':
            buffer = scipy.io.loadmat(file_path)
            data = {}
            for stream_type_label, data_array in buffer.items():
                if stream_type_label.endswith(' timestamp'):
                    stream_type_label = stream_type_label.replace(' timestamp', '')
                    data[stream_type_label] = [buffer[stream_type_label], data_array.squeeze()]
            return data
        elif file_extension == '.pickle' or file_extension == '.pkl' or file_extension == '.p':
            return pickle.load(open(file_path, 'rb'))
        elif file_extension == '.xdf':
            return load_xdf(file_path)
        else:
            raise Exception('Unknown file extension: {}'.format(file_extension))