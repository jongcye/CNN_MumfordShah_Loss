from __future__ import print_function, division, absolute_import, unicode_literals
import scipy.io as sio
import h5py
import numpy as np
from PIL import Image
from scipy import misc, ndimage
import os, os.path
import ipdb
import matplotlib.pyplot as plt
from image_util_unet import BaseDataProvider
from augment_method_unet import ElasticTransform, Zoom
from scipy.ndimage.interpolation import rotate

class DataProvider_LiTS(BaseDataProvider):

    n_class = 2

    def __init__(self, inputSize,fineSize, segtype, semi_rate, input_nc, path, a_min=0, a_max=100, mode=None):
        super(DataProvider_LiTS, self).__init__(a_min, a_max)
        self.nx       = inputSize
        self.ny       = inputSize
        self.nx_f = fineSize
        self.ny_f = fineSize
		self.semi_rate = semi_rate
        self.segtype = segtype
        self.channels = input_nc
        self.path     = path
        self.mode     = mode
        self.data_idx = -1
        self.n_data = self._load_data()

    def _load_data(self):
        path_ = os.path.join(self.path, self.mode)
        filefolds = os.listdir(path_)
        self.imageNum = []
        self.filePath = []

        for isub, filefold in enumerate(filefolds):
            foldpath = os.path.join(path_, filefold)
            dataFold = sorted(os.listdir(foldpath))
            for inum, idata in enumerate(dataFold):
                dataNum = int(idata.split('.')[0])
                dataFold[inum] = dataNum
            dataFile = sorted(dataFold)
            for islice in range(1, len(dataFile)-1):
                filePath = os.path.join(foldpath, str(dataFile[islice]) + '.mat')
                file = sio.loadmat(filePath)
                data = file['imdb']['data'][0][0]
                label = file['imdb']['labels'][0][0]
                if np.amax(data) == 0: continue
                if np.amax(label) == 0: continue
                if self.segtype == "tumor":
                    if np.amax(label)!=2: continue
                self.imageNum.append((foldpath, dataFile[islice], isub))

        if self.mode == "train":
            np.random.shuffle(self.imageNum)

        return len(self.imageNum)

    def _shuffle_data_index(self):
        self.data_idx += 1
        if self.data_idx >= self.n_data:
            self.data_idx = 0
            if self.mode =="train":
                np.random.shuffle(self.imageNum)

    def _next_data(self):
        self._shuffle_data_index()
        filePath = self.imageNum[self.data_idx]
        data = np.zeros((self.nx, self.ny, self.channels))
        labels = np.zeros((self.nx, self.ny, self.channels))
		
        for ich in range(self.channels):
            fileName = os.path.join(filePath[0], str(filePath[1]-1+ich) + '.mat')
            file = sio.loadmat(fileName)
            data[:,:,ich] = file['imdb']['data'][0][0]
			if filePath[-1] % self.semi_rate == 0:
	            labels[:, :, ich] = file['imdb']['labels'][0][0]

        data = np.clip(data+124, 0, 400)
		
        path = filePath[0] + str(filePath[1])
        return data, labels, path

    def _augment_data(self, data, labels):
        if self.mode == "train":
            if self.segtype == "liver":
                labels = (labels[...,1]>0).astype(float)
            else:
                for ich in range(self.channels):
                    data[..., ich] = data[..., ich] * (labels[..., ich] > 0)
                labels = (labels[...,1] ==2).astype(float)

            # downsampling x2
            op = np.random.randint(0, 4)
            if op == 0:
                data = data[::2, ::2]
                labels = labels[::2, ::2]
            elif op == 1:
                data = data[::2, 1::2]
                labels = labels[::2, 1::2]
            elif op == 2:
                data = data[1::2, ::2]
                labels = labels[1::2, ::2]
            elif op == 3:
                data = data[1::2, 1::2]
                labels = labels[1::2, 1::2]

            # Rotation 90
            op = np.random.randint(0, 4)  # 0, 1, 2, 3
            data, labels = np.rot90(data, op), np.rot90(labels, op)
            
            # Flip horizon / vertical
            op = np.random.randint(0, 3)  # 0, 1
            if op < 2:
                data, labels = np.flip(data, op), np.flip(labels, op)

        return data, labels

