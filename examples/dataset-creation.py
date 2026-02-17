import numpy as np
import pandas as pd

"""
This script creates two playground datasets to test our models.
The first one are two classes of points ubicated in a square regions.
The second one are two classes of points in a circle distribution.
"""

# ============================
# 1. Square regions dataset
# ============================

def square_regions_ds(n):

    # The points will be stored in a tuple ((x,y),label)
    X = []
    Y = []
    label = []
    labels = np.random.choice([0,1], n)

    mean_0_1, mean_0_2 = (-1,-1), (1, 1)
    mean_1_1, mean_1_2 = (-1, 1), (1,-1)
    means = [(mean_0_1, mean_0_2), (mean_1_1, mean_1_2)]
    cov = 0.25*np.eye(2)

    for i in labels:
        j = np.random.binomial(1,0.5)
        mean = means[i][j]
        point = np.random.multivariate_normal(mean,cov)
        X.append(point[0])
        Y.append(point[1])
        label.append(i)

    df = pd.DataFrame({"x": X, "y": Y, "label": label})
    df.to_csv("examples/datasets/square_regions.csv",index=False)
    
# ===========================
# 2. Circle regions dataset
# ===========================

# We take random points over a segment and apply the function r*x/norm(x)

def circle_regions_dataset(n):

    X = []
    Y = []
    label = []

    labels = np.random.choice([0,1],n)
    means_rad = [0.5,1]
    
    cov = np.eye(2)
    points = np.random.multivariate_normal((0,0),cov,n)

    for i in range(n):
        while points[i][0] == 0 and points[i][1] == 0:
            points[i] = np.random.multivariate_normal((0,0), np.eye(2))
        radius = np.random.normal(means_rad[labels[i]],0.2)
        point = radius*points[i]/np.linalg.norm(points[i])

        X.append(point[0])
        Y.append(point[1])
        label.append(labels[i])
    
    df = pd.DataFrame({"x":X, "y":Y, "label":label})
    df.to_csv("examples/datasets/circle_regions.csv",index=False)

circle_regions_dataset(1000)


