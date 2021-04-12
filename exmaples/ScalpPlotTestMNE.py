"""
from https://mne.tools/dev/auto_tutorials/evoked/plot_20_visualize_evoked.html?highlight=scalp
"""
import os
import numpy as np
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_evk_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis-ave.fif')
evokeds_list = mne.read_evokeds(sample_data_evk_file, baseline=(None, 0),
                                proj=True, verbose=False)
# show the condition names
for e in evokeds_list:
    print(e.comment)

conds = ('aud/left', 'aud/right', 'vis/left', 'vis/right')
evks = dict(zip(conds, evokeds_list))

evks['aud/left'].plot(exclude=[])

times = np.linspace(0.05, 0.13, 5)
evks['aud/left'].plot_topomap(ch_type='mag', times=times, colorbar=True)