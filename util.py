import matplotlib.pyplot as plt
import joblib
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from skimage import data, color
from skimage.transform import rescale
import numpy as np

global digits
digits = datasets.load_digits()

global X_train, X_test, y_train, y_test

def train_data(data,digits,test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=test_size, shuffle=False)
    return X_train, X_test, y_train, y_test