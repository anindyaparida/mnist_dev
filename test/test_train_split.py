from collections import defaultdict
import random
import pytest
from sklearn import datasets, svm
from plot_graph import preprocess, create_splits, test
import utils
import os.path
from pathlib import Pat

def test_create_splits(digits,targets):
    train = 70
    test = 15
    val = 15
    total = test + test + val
    digits = datasets.load_digits()
    split_sum = create_splits(digits, targets,test, val)
    assert split_sum == sum()
    assert total == sum(train,test,val)

def split_data(digits,target):
    digits = datasets.load_digits()
    train = 0.70
    test = 0.15
    val = 0.15
    total = train + test + val
    assert total == sum(train,test,val)

