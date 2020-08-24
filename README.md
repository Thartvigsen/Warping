### Predicting Warping Paths with a Neural Network

This code is all written in Python 3, so set up a virtual environment in this
directory:
```
python3 -m venv ./env
```
Next, install the packages using the requirements file:
```
pip install -r requirements.txt
```

To run the code from command line, run
```
python main.py
```

This will make some plots pop up showing the results!

The code is also written as a [jupyter notebook](./main.ipynb), though it is
messier.

One hiccup is the "torchsearchsorted" package -- which has to be installed
manually from: https://github.com/aliutkus/torchsearchsorted
