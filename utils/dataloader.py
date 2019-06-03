import pickle
import numpy as np
import os
from utils.equalizer import *
from config import *

#dataset_path: path to *.np dataset that has been preprocessed
#normdata_path: path to directory where norm and equalization data is stored ('normalization.np','equalization.np')
#img_res: the shape of the orthotopres (cubes) being processed
class DataLoader():
    def __init__(self, dataset_path, normdata_path, img_res=None):
        self.normdata_path = normdata_path
        if img_res is not None:
            self.img_res = img_res
        else:
            self.img_res = config['cube_size']

        self.m_xlims = config['mask_xlims']
        self.m_ylims = config['mask_ylims']
        self.m_zlims = config['mask_zlims']
        print("loading preprocessed dataset...")
        self.data_train = np.load(dataset_path)
        # format for nerual net
        self.data_train = self.data_train.reshape((len(self.data_train), self.img_res[0], self.img_res[1], self.img_res[2], 1))
        # shuffle
        np.random.shuffle(self.data_train)

    def load_data(self, batch_size=1, is_testing=False):
        if is_testing == False:
            idx = np.random.permutation(len(self.data_train))
            batch_images = self.data_train[idx[:batch_size]]
        else:
            idx = np.random.permutation(len(self.data_train))
            batch_images = self.data_train[idx[:batch_size]]
        imgs_A = []
        imgs_B = []
        for i, img in enumerate(batch_images):
            imgs_A.append(img)
            img_out = np.copy(img)
            img_out[self.m_zlims[0]:self.m_zlims[1], self.m_xlims[0]:self.m_xlims[1], self.m_ylims[0]:self.m_ylims[1]] = 0
            imgs_B.append(img_out)

        return np.array(imgs_A), np.array(imgs_B)

    def load_batch(self, batch_size=1, is_testing=False):
        if is_testing == False:
            self.n_batches = int(len(self.data_train) / batch_size)
        else:
            self.n_batches = int(len(self.data_train) / batch_size)

        for i in range(self.n_batches - 1):
            if is_testing == False:
                batch = self.data_train[i * batch_size:(i + 1) * batch_size]
            else:
                batch = self.data_train[i * batch_size:(i + 1) * batch_size]
            imgs_A = []
            imgs_B = []
            for i, img in enumerate(batch):
                imgs_A.append(img)
                img_out = np.copy(img)
                img_out[self.m_zlims[0]:self.m_zlims[1], self.m_xlims[0]:self.m_xlims[1], self.m_ylims[0]:self.m_ylims[1]] = 0
                imgs_B.append(img_out)
            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)
            yield imgs_A, imgs_B

