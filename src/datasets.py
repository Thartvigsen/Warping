import torch
from src.utils import *
import numpy as np
from scipy.interpolate import interp1d

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]
    
    def __len__(self):
        return len(self.labels)

class CosSin(Dataset):
    def __init__(self, T=50, N=500, prewarp=True):
        super(CosSin, self).__init__()
        self.N = N
        self.T = T
        self.prewarp = prewarp
        self.data, self.labels = self.loadData()
    
    def warp(self, time_series):
        # Get warp function (monotonic increasing, range \in [0, 1], gamma(0) = 0.0, gamma(1) = 1.0)
        gamma = exponentialWarp(self.T)
        f = interp1d(np.linspace(0, 1, self.T), time_series)
        warped_time_series = f(gamma)
        return warped_time_series
    
    def loadData(self):
        data = []
        labels = np.concatenate(([0]*int(self.N//2), [1]*int(self.N//2)))
        values = np.zeros((self.N, self.T))
        #values = np.random.normal(0, 0.001, (self.N, self.T))
        for i in range(self.N):
            if i <= int(self.N/2): # Class 1 is a sin wave
                time_series = np.sin(np.linspace(0, 2*np.pi, self.T))
            else: # Class 2 is a cosine wave
                time_series = np.cos(np.linspace(0, 2*np.pi, self.T))
            if self.prewarp:
                time_series = self.warp(time_series)
            values[i, :] += time_series
        self.train_ix = np.random.choice(np.arange(self.N), int(self.N*0.8), replace=False)
        self.test_ix = np.array(list(set(np.arange(self.N)) - set(self.train_ix)))
        return torch.tensor(values, dtype=torch.float), torch.tensor(labels, dtype=torch.long)

class UCR(Dataset):
    def __init__(self, dataset="ECGFiveDays"):
        super(UCR, self).__init__()
        self.dataset = dataset
        self.data, self.labels = self.loadData()
        self.nclasses = len(np.unique(self.labels))
    
    def loadData(self):
        train = np.loadtxt("/home/tom/Documents/data/UCR_TS_Archive_2015/{}/{}_TRAIN".format(self.dataset, self.dataset), delimiter=",")
        test = np.loadtxt("/home/tom/Documents/data/UCR_TS_Archive_2015/{}/{}_TEST".format(self.dataset, self.dataset), delimiter=",")
        data = np.concatenate((train, test))
        labels = (1+data[:, 0])/2 # ECG200
        #labels = data[:, 0] - 1 # ECGFiveDays
        values = data[:, 1:]
        self.N = len(data)
        self.T = values.shape[1]
        self.train_ix = np.random.choice(np.arange(self.N), int(self.N*0.8), replace=False)
        self.test_ix = np.array(list(set(np.arange(self.N)) - set(self.train_ix)))
        return torch.tensor(values, dtype=torch.float), torch.tensor(labels, dtype=torch.long)

# Now what if classes are warped differently?
class DiffWarps(torch.utils.data.Dataset):
    def __init__(self, T=50, N=500):
        super(DiffWarps, self).__init__()
        self.N = N
        self.T = T
        self.data, self.labels = self.loadData()

    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]

    def __len__(self):
        return len(self.labels)

    def logisticWarp(self):
        gamma = 1/(0.5+np.exp(-10*np.linspace(0.000001, 1, self.T)))
        gamma = (gamma-np.min(gamma))/(np.max(gamma)-np.min(gamma))
        return gamma

    def exponentialWarp(self):
        gamma = np.exp(np.linspace(0, 5, self.T))
        gamma = gamma/np.max(gamma)
        return gamma

    def identityWarp(self):
        return np.linspace(0, 1, self.T)

    def warp(self, time_series, class1=True):
        if class1:
            # Sin
            gamma = self.exponentialWarp()
            #gamma = self.logisticWarp()
        else:
            # Cos
            #gamma = self.logisticWarp()
            gamma = self.identityWarp()
        gamma[0] = 0.0
        gamma[-1] = 1.0
        f = interp1d(np.linspace(0, 1, self.T), time_series)
        warped_time_series = f(gamma)
        return warped_time_series

    def loadData(self):
        data = []
        labels = []
#         values = np.zeros((self.N, self.T))
        values = np.random.normal(0, 0.10, (self.N, self.T))
        for i in range(self.N):
            if i <= int(self.N/2): # Class 1
                time_series = np.sin(np.linspace(0, 2*np.pi, self.T))
                values[i, :] += self.warp(time_series, class1=True)
                labels.append(0)
            else: # Class 2
                time_series = np.cos(np.linspace(0, 2*np.pi, self.T))
                values[i, :] += self.warp(time_series, class1=False)
                labels.append(1)
        self.train_ix = np.random.choice(np.arange(self.N), int(self.N*0.8), replace=False)
        self.test_ix = np.array(list(set(np.arange(self.N)) - set(self.train_ix)))
        return torch.tensor(values, dtype=torch.float), torch.tensor(labels, dtype=torch.long)

class CosSinStretch(Dataset):
    def __init__(self, T=50, N=500):
        super(CosSinStretch, self).__init__()
        self.N = N
        self.T = T
        self.data, self.labels = self.loadData()
    
    def warp(self, time_series):
        # Get warp function (monotonic increasing, range \in [0, 1], gamma(0) = 0.0, gamma(1) = 1.0)
        gamma = exponentialWarp(self.T)
        f = interp1d(np.linspace(0, 1, self.T), time_series)
        warped_time_series = f(gamma)
        return warped_time_series
    
    def loadData(self):
        data = []
        labels = np.concatenate(([0]*int(self.N//2), [1]*int(self.N//2)))
        starts = np.random.uniform(0, 3*np.pi/4, self.N)
        ends = np.random.uniform(5*np.pi/4, 2*np.pi, self.N)
        values = np.zeros((self.N, self.T))
        #values = np.random.normal(0, 0.1, (self.N, self.T))
        for i in range(self.N):
            if i <= int(self.N/2): # Class 1 is a sin wave
                time_series = np.sin(np.linspace(starts[i], ends[i], self.T))
            else: # Class 2 is a cosine wave
                time_series = np.cos(np.linspace(starts[i], ends[i], self.T))
            values[i, :] += time_series
        self.train_ix = np.random.choice(np.arange(self.N), int(self.N*0.8), replace=False)
        self.test_ix = np.array(list(set(np.arange(self.N)) - set(self.train_ix)))
        return torch.tensor(values, dtype=torch.float), torch.tensor(labels, dtype=torch.long)
