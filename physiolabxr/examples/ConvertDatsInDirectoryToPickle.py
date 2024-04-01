import os
import pickle

from physiolabxr.utils.RNStream import RNStream

my_directory = f'/Users/apocalyvec/PycharmProjects/Temp/AOIAugmentation/Participants'

def directory_dats_to_pickle_recursive(directory):
    files_to_convert = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.dats'):
                files_to_convert.append(os.path.join(dirpath, filename))

    converted_file_paths = [x.replace('.dats', '.p') for x in files_to_convert]

    for i, (f, fc) in enumerate(zip(files_to_convert, converted_file_paths)):
        print('Working on file {} of {}'.format(i + 1, len(files_to_convert)))
        test_rns = RNStream(f)
        reloaded_data = test_rns.stream_in(jitter_removal=False, ignore_stream=['monitor1', '1'])
        with open(fc, 'wb') as outfile:
            pickle.dump(reloaded_data, outfile)


if __name__ == '__main__':
    directory_dats_to_pickle_recursive(my_directory)