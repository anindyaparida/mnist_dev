from numpy.lib import utils
from utils import get_random_acc,test,run_classification_experiment

import numpy as np

def test_digit_correct_0():
    y=np.array([0])
    assert get_random_acc(y)==0

def test_digit_correct_1():
    y=np.array([1])
    assert get_random_acc(y)==1

def test_digit_correct_2():
    y=np.array([2])
    assert get_random_acc(y)==2

def test_digit_correct_3():
    y=np.array([7,3,1,2,8,9])
    assert get_random_acc(y)==3

def test_digit_correct_4():
    y=np.array([6,3,4,1,5])
    assert get_random_acc(y)==4

def test_digit_correct_5():
    y=np.array([1,2,3,4,5,6])
    assert get_random_acc(y)==5

def test_digit_correct_6():
    y=np.array([6,1,2,3,4,5])
    assert get_random_acc(y)==6

def test_digit_correct_7():
    y=np.array([1,2,3,7,6,8])
    assert get_random_acc(y)==7

def test_digit_correct_8():
    y=np.array([1,2,3,4,8,2])
    assert get_random_acc(y)==8

def test_digit_correct_9():
    y=np.array([1,2,3,9,7,6])
    assert get_random_acc(y)==9