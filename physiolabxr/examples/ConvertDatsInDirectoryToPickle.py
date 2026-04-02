import os
import pickle

from physiolabxr.utils.RNStream import RNStream

my_file = r'C:\Data\Wingman\03_24_2026_12_04_40-Exp_wingman_us3_c-Sbj_43-Ssn_1.dats'


def single_dats_to_pickle(file_path):
    if not file_path.endswith('.dats'):
        print(f"File {file_path} is not a .dats file.")
        return

    converted_file_path = file_path.replace('.dats', '.p')
    print('Working on file:', file_path)
    test_rns = RNStream(file_path)
    reloaded_data = test_rns.stream_in(jitter_removal=False, ignore_stream=['monitor1', '1'])
    with open(converted_file_path, 'wb') as outfile:
        pickle.dump(reloaded_data, outfile)
    print('File converted to:', converted_file_path)


if __name__ == '__main__':
    single_dats_to_pickle(my_file)
