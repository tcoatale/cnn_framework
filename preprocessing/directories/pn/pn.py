# -*coding: utf-8 -*-
import os
import glob

app_raw_data_root = os.path.join('..', '..', '..', 'data', 'raw', 'pn')
gabor_dir = os.path.join('data', 'augmented', 'pn', 'gabor')
hog_dir = os.path.join('data', 'augmented', 'pn', 'hog')


#%%
seq = '20160707'
files = glob.glob(os.path.join(app_raw_data_root, 'frames', '*' + seq + '*'))
file_numbers = list(map(lambda f: int(f.split('_')[-1][:-4]), files))

#%%
file_numbers = sorted(file_numbers)