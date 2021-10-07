#Test Mnist mode taking train and validation dataset.
from collections import defaultdict
import random
import pytest
from sklearn import datasets, svm
from plot_graph import preprocess, create_splits, test
import utils
import os.path
from pathlib import Path

def test_model_writing(data, targets, test_size, valid_size):
    digits = datasets.load_digits()
    images = digits.images
    rescale_factor = [1]
    resized_images = utils.preprocess(images, rescale_factor)
    create_splits = utils.create_splits(data, targets, test_size, valid_size)
    path_to_file = 'models/tt_0.15_val_0.15_rescale_1_gamma_0.01/model.joblib'
    path = Path(path_to_file)
    assert os.path.is_file(path)

def test_small_data_overfit_checking(data, accuracy, f1 ):
    clf = svm.SVC(gamma=gamma)
    digits = datasets.load_digits()
    images = digits.images
    X_train, X_test, X_valid, y_train, y_test, y_valid = create_splits(
        data, digits.target, test_size, valid_size)

    train_metrics = utils.test(clf, X_valid, y_valid)

