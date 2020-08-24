import torch
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from src.models import WarpClassifier
from src.datasets import *
from src.utils import *

if __name__ == "__main__":
    # Run code here!

    # Load data
    T = 1000 # Number of timesteps per time series
    N = 100 # Number of time series to generate
    bsz = 10 # batch size
    lr = 1e-05 # learning rate
    nepoch = 200

    #D = CosSin(T=T, N=N, prewarp=False)
    #D = UCR("ECG200")
    #D = DiffWarps(T=T, N=N)
    D = CosSinStretch(T=T, N=N)
    T = D.T
    train_sampler = SubsetRandomSampler(D.train_ix)
    train_loader = torch.utils.data.DataLoader(D, batch_size=bsz, sampler=train_sampler, drop_last=True)
    test_sampler = SubsetRandomSampler(D.test_ix)
    test_loader = torch.utils.data.DataLoader(D, batch_size=bsz, sampler=test_sampler, drop_last=True)

    M = WarpClassifier(ninp=T, nclasses=2)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, M.parameters()), lr=lr, weight_decay=1e-05)

    # Train the model
    loss_values = train(M, optimizer, train_loader, nepoch)
    plotLoss(loss_values)

    # Test the model
    labels, time_series, warping_paths, warped_time_series = test(M, test_loader)

    # Plot the results
    plotResults(time_series, warping_paths, warped_time_series, labels)
