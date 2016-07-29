# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from scipy import ndimage as ndi

from skimage.filters import gabor_kernel
from skimage import color
from skimage import io

from datetime import datetime
import pandas as pd

#%%
data_dir = os.path.join('..', 'raw', 'driver', 'train')
image_files = glob.glob(os.path.join(data_dir, '*', '*'))

#%%
thetas = np.arange(0, 1, 0.33) * np.pi
sigmas = np.arange(1, 10, 3)
freqs = np.arange(0, 10, 3)

#%%
def compute_feats(image, kernels):
  feats = np.zeros((len(kernels), 2), dtype=np.double)
  for k, kernel in enumerate(kernels):
    filtered = ndi.convolve(image, kernel, mode='wrap')
    feats[k, 0] = filtered.mean()
    feats[k, 1] = filtered.var()
  return feats
  
def get_kernels(thetas, sigmas, freqs):
  kernels = []
  for theta in thetas:
      for sigma in sigmas:
          for frequency in freqs:
              kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
              kernels += [kernel]
              
  return kernels
  
def load(file, shrink):
  image = color.rgb2gray(io.imread(file))
  return image[shrink]
  
def get_reference(image_files, index, kernels):
  if index % 50 == 0:
    print(datetime.now(), index)
  
  image_file = image_files[index]
  shrink = (slice(0, None, 4), slice(0, None, 4))
  im = load(image_file, shrink)
  features = compute_feats(im, kernels)
  features = features.reshape([-1])
  return features

#%%
kernels = get_kernels(thetas, sigmas, freqs)
features = np.array(list(map(lambda i: get_reference(image_files, i, kernels), range(len(image_files)))))

df_dict = {}
cols = list(range(features.shape[1]))

for col in cols:
  df_dict[col] = features[:, col]
df_dict['file'] = image_files
df = pd.DataFrame(df_dict)

df.to_csv('gabor.csv', index=False)


'''
from sklearn.manifold import Isomap
dimen_reductor = Isomap()
features = dimen_reductor.fit_transform(ref_20)
'''