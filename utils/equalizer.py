import numpy as np
import pickle

class histEq:
    def __init__(self,X_sample,levels=10000,nbins=10000, path=[]): #X_sample is a list of one or more images sampled from the distribution, levels is the ourput norm range
        if len(path) > 0:
            self.load(path)
        else:
            self.nbins = nbins
            self.init(X_sample,levels,nbins)

    def init(self,X_sample,levels,nbins):
        #format X
        X_f = np.concatenate(X_sample)
        self.N =  np.prod(X_f.shape) #number of pixels
        self.L = levels #grey levels to eq to

        hist, bins = np.histogram(X_f.ravel(), nbins)
        self.hist = hist/len(X_sample) #normalize to thenumber of images
        self.bins = bins + ((max(bins)-min(bins))/len(bins))/2 #center of each bin
        self.bins = self.bins[:nbins]
        self.cdf = np.cumsum(self.hist)
        self.cdf_min = np.min(self.cdf)

    def equalize(self,image):
        out = np.interp(image.flat, self.bins, self.cdf)
        return out.reshape(image.shape)

    def dequalize(self,image):
        out = np.interp(image.flat, self.cdf, self.bins)
        return out.reshape(image.shape)

    def save(self,path=None):
        if path is None:
            return [self.bins,self.cdf]
        with open(path, 'wb') as f:
            pickle.dump([self.bins,self.cdf], f)

    def load(self,path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.bins = data[0]
        self.cdf = data[1]

