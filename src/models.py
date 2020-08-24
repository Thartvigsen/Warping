import torch
import torch.nn.functional as F
from src.interpolator import Interp1d

# Now let's get a machine learning example running... learn gamma!
class WarpClassifier(torch.nn.Module):
    def __init__(self, ninp, nclasses):
        # Expects time series of shape (instances x number of timesteps)
        super(WarpClassifier, self).__init__()

        self.fc = torch.nn.Linear(ninp, nclasses)
        self.w = torch.nn.Linear(ninp, ninp)
        self.Interpolator = Interp1d()

    def warp(self, x):
        self.gamma = torch.cumsum(torch.softmax(self.w(x), 1), 1)
        x = self.Interpolator(torch.linspace(0, 1, x.shape[1]), x, self.gamma)
        return x

    def forward(self, time_series):
        # Apply warping
        self.warped_time_series = self.warp(time_series)

        # Compute model's predictions
        logits = self.fc(self.warped_time_series)
        return logits
