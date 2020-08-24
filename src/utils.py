import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
pink = "#FA4563"
dark_blue = "#065472"
light_blue = "#63bae7"
gray = "#808080"

# --- warp functions ---
def logisticWarp(T):
    gamma = 1/(0.5+np.exp(-10*np.linspace(0.000001, 1, T)))
    gamma = (gamma-np.min(gamma))/(np.max(gamma)-np.min(gamma))
    gamma[0] = 0.0
    gamma[-1] = 1.0
    return gamma

def exponentialWarp(T):
    gamma = np.exp(np.linspace(0, 5, T))
    gamma = gamma/np.max(gamma) 
    gamma[0] = 0.0
    gamma[-1] = 1.0
    return gamma

def identityWarp(T):
    gamma = np.linspace(0, 1, T)
    gamma[0] = 0.0
    gamma[-1] = 1.0
    return gamma

def uniformWarp(T):
    gamma = np.sort(np.random.uniform(0, 1, T))
    gamma[0] = 0.0
    gamma[-1] = 1.0
    return gamma

# --- training and testing the model ---
def train(model, optimizer, loader, nepoch):
    loss_values = []
    for e in range(nepoch):
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss_values.append([loss.item()])
            loss.backward()
            optimizer.step()
    return loss_values

def test(model, loader):
    predictions = []
    labels = []
    warping_paths = []
    warped_time_series = []
    time_series = []
    for x, y in loader:
        logits = model(x)
        warping_paths.append(model.gamma.squeeze())
        [time_series.append(i) for i in x]
        labels.append(y)
        warped_time_series.append(model.warped_time_series.squeeze())
        y_hat = torch.softmax(logits, 1).argmax(1) # Record what class the model predicted
        predictions.append(y_hat)
    labels = np.stack(labels).squeeze().reshape(-1)
    predictions = np.stack(predictions).squeeze().reshape(-1)
    warping_paths = torch.stack(warping_paths).squeeze()
    warping_paths = warping_paths.reshape(-1, warping_paths.shape[-1]).detach().numpy()
    time_series = torch.stack(time_series).squeeze().detach().numpy()
    warped_time_series = torch.stack(warped_time_series)
    warped_time_series = warped_time_series.reshape(-1, warped_time_series.shape[-1]).detach().numpy()
    print("Test Accuracy: {}%".format(100*np.mean(labels == predictions)))
    return labels, time_series, warping_paths, warped_time_series

# --- plotting functions ---
def plotResults(time_series, warping_paths, warped_time_series, labels=None):
    """
    Parameters
    ----------
    time_series : numpy array of shape (instances x number of timesteps)
        These are the time series that have been fed to the model.

    warping_paths : numpy array of shape (instances x number of timesteps)
        These are the predicted warps (i.e., vectors of length T that are
        monotonically increasing from 0 to 1).

    warped_time_series : numpy array of shape (instances x number of timesteps)
        These are the time series AFTER warping has been applied so we can see
        the effects when comparing to time_series.

    labels (optional) : numpy array of shape (instances)
        These are the labels associated with the input time series
    """
    fig, ax = plt.subplots(1, 3, figsize=(32, 6))

    #colors = ["k", "r"]
    colors = [light_blue, gray]

    for i in range(len(time_series)):
        if type(labels) == np.ndarray: # If labels were input, color the time series
            ax[0].plot(time_series[i], c=colors[int(labels[i])])
            ax[1].plot(warping_paths[i], c=colors[int(labels[i])], lw=0.5)
            ax[2].plot(warped_time_series[i], c=colors[int(labels[i])])
        else: # Otherwise, no colors
            ax[0].plot(time_series[i], c="k")
            ax[1].plot(warping_paths[i], c="k")
            ax[2].plot(warped_time_series[i], c="k")

    if type(labels) == np.ndarray:
        # Add cluster centroids
        c1_mean = warping_paths[np.where(labels == 0.)].mean(0)
        c2_mean = warping_paths[np.where(labels == 1.)].mean(0)

        ax[1].plot(c1_mean, c=dark_blue, lw=2, label="Class 1")
        ax[1].plot(c2_mean, c="k", lw=2, label="Class 2")

    ax[0].set_title("Raw Time Series", fontfamily="serif", fontsize=24)
    ax[0].set_xlabel("Time", fontfamily="serif", fontsize=18)
    ax[0].set_ylabel("Values", fontfamily="serif", fontsize=18)
    ax[1].legend(loc="upper left")
    ax[1].set_title("Predicted Warping Paths", fontfamily="serif", fontsize=24)
    ax[1].set_xlabel("Time", fontfamily="serif", fontsize=18)
    ax[1].set_xticks(np.round(np.linspace(0, time_series.shape[1], 11), 2))
    ax[1].set_xticklabels(np.round(np.linspace(0, 1, 11), 2))
    ax[1].set_ylabel(r"Time$^\prime$", fontfamily="serif", fontsize=18)
    ax[2].set_title("Warped Time Series", fontfamily="serif", fontsize=24)
    ax[2].set_xlabel("Time", fontfamily="serif", fontsize=18)
    ax[2].set_ylabel("Values",
    fontfamily="serif", fontsize=18)
    plt.show()

def plotLoss(loss_values):
    plt.plot(loss_values)
    plt.xlabel("Iterations", fontfamily="serif", fontsize=24)
    plt.ylabel("Cross Entropy", fontfamily="serif", fontsize=24)
    plt.show()
