# -*- coding: utf-8 -*-
import skimage.io
import skimage.feature
import skimage.util
import numpy as np
import itertools
from tqdm import tqdm
import sklearn.decomposition
import os
import glob

class BriefExtractor:
  def __init__(self, file, freq=16):
    self.file = file
    size = skimage.io.imread(file).shape[0]
    num = int(size / freq)
    axis_0 = np.array(list(range(num))) * freq + freq / 2
    axis_1 = np.array(list(range(num))) * freq + freq / 2
    self.keypoints = np.array(list(itertools.product(axis_0,axis_1)), dtype=np.uint8)
    
  def get_brief_descriptor(self):
    image = skimage.io.imread(self.file)
    brief = skimage.feature.BRIEF(descriptor_size=256, patch_size=17)
    brief.extract(image, self.keypoints)
    return brief.descriptors
  
class BriefExtractionManager:
  def __init__(self, frames_dir, dest_dir, freq=16, n_components=3):
    self.frames = glob.glob(os.path.join(dir, '*'))
    self.dest_dir = dest_dir
    self.n_components = n_components
    self.freq = freq
    size = skimage.io.imread(self.frames[0]).shape[0]
    self.num = int(size / self.freq)
    
  def get_brief_wrapper(self, file):
    brief_extractor = BriefExtractor(file, freq=self.freq)
    return brief_extractor.get_brief_descriptor()
    
  def get_all_briefs(self):
    briefs = []
    for file in tqdm(self.frames):
      briefs += [self.get_brief_wrapper(file)]
    return briefs
    
  def train_pca(self, briefs):
    indices = np.random.choice(list(range(len(briefs))), 10000)
    chosen_brief_descriptors = list(map(lambda i: briefs[i], indices))
    pca_learning_examples = np.vstack(chosen_brief_descriptors)
    pca = sklearn.decomposition.PCA(n_components=self.n_components, whiten=True)
    pca.fit(pca_learning_examples)
    return pca
    
  def get_brief_reduced_descriptors(self, briefs, pca):
    briefs = np.vstack(briefs)
    reduction = pca.transform(briefs)
    reduction -= reduction.min(axis=0)
    reduction /= reduction.max(axis=0)
    reduction *= 2
    reduction -= 1.0
    return reduction
    
  def write_as_image(self, line):
    path, image = line
    dir, file = os.path.split(path)
    dest_path = os.path.join(self.dest_dir, file)
    skimage.io.imsave(dest_path, image)
    
  def run_extraction(self):
    briefs = self.get_all_briefs()
    pca = self.train_pca(briefs)
    reduced_descriptors = self.get_brief_reduced_descriptors(briefs, pca)
    image_reductions = reduced_descriptors.reshape([len(self.frames), self.num, self.num, self.n_components])
    lines = list(zip(self.frames, image_reductions))
    list(map(self.write_as_image, lines))

