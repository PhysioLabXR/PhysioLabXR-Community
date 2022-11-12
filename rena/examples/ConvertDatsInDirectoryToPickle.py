import os
import pickle

from rena.utils.data_utils import RNStream

directory = 'C:/Users/LLINC-Lab/Dropbox/ReNa/data/ReNaPilot-2022Fall/Subjects-Test-IncompleteBlock/0'

files_to_convert = [os.path.join(directory, x) for x in os.listdir(directory) if x.endswith('.dats')]
converted_file_paths = [x.replace('.dats', '.p') for x in files_to_convert]

for i, (f, fc) in enumerate(zip(files_to_convert, converted_file_paths)):
    print('Woring on file {} of {}'.format(i+1, len(files_to_convert)))
    test_rns = RNStream(f)
    reloaded_data = test_rns.stream_in(jitter_removal=False, ignore_stream=['monitor1', '1'])
    pickle.dump(reloaded_data, open(fc, 'wb'))

