# Imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sklearn
from scipy import stats
import pandas as pd
from math import sqrt
import sys
from itertools import cycle
%matplotlib inline
# Install a conda package in the current Jupyter kernel
#!conda install --yes --prefix {sys.prefix} ruptures
#RUPTURES PACKAGE
import ruptures as rpt


def doCPD(data, method = "PELT", MAXBREAKPOINTS=5, show_graph = True):
    bkps = []
    title=''
    if method == "Binary Segmentation":    #Binary Segmentation search method
        model = "l2"  
        method = rpt.Binseg(model=model).fit(data)
        bkps = method.predict(n_bkps=MAXBREAKPOINTS)
        title = "Binary Segmentation Search"
    elif method == "Window":    #Window-based search method
        model = "l2"  
        method = rpt.Window(width=10, model=model).fit(data)
        bkps = method.predict(n_bkps=MAXBREAKPOINTS)
        title = "Window-Based Search"
    elif method == "Dynamic":    #Dynamic programming search method
        model = "l1"  
        method = rpt.Dynp(model=model, min_size=3, jump=4).fit(data)
        bkps = method.predict(n_bkps=MAXBREAKPOINTS)
        title = "Dynamic Programming Search"
    else:    # Pelt - Pruned Exact Linear Time
        model="rbf"
        method = rpt.Pelt(model=model).fit(data)
        bkps  = method.predict(pen=MAXBREAKPOINTS)
        title = "Pelt Search"
        
    # show results
    if show_graph:
        rpt.show.display(data, bkps, figsize=(10, 6))
        plt.suptitle(title, y=1.1, size=28)
        plt.show()
        print("Breaks at: " + str(bkps[:-1]))
    return bkps[:-1]
