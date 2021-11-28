# mnist_dev
Scikit-learn mnist dataset
Run code - svm_mnist.py

For this data is splitted across -  80:10:10 train:dev:test.
The gamma hyper parameter is tuned for changing the gamma values [0.01,0.001,0.0001]
Classification report :  SVC(gamma=0.0001):
              precision    recall  f1-score   support

           0       1.00      0.95      0.97        19
           1       0.88      0.88      0.88        17
           2       1.00      1.00      1.00        18
           3       1.00      0.84      0.91        19
           4       0.94      0.94      0.94        17
           5       0.95      1.00      0.97        19
           6       1.00      0.95      0.97        19
           7       0.84      0.94      0.89        17
           8       0.75      0.75      0.75        16
           9       0.81      0.89      0.85        19

    accuracy                           0.92       180
   macro avg       0.92      0.91      0.91       180
weighted avg       0.92      0.92      0.92       180


Confusion matrix:
[[18  0  0  0  1  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0  2]
 [ 0  0 18  0  0  0  0  0  0  0]
 [ 0  0  0 16  0  0  0  2  1  0]
 [ 0  0  0  0 16  0  0  0  1  0]
 [ 0  0  0  0  0 19  0  0  0  0]
 [ 0  1  0  0  0  0 18  0  0  0]
 [ 0  0  0  0  0  0  0 16  1  0]
 [ 0  1  0  0  0  1  0  0 12  2]
 [ 0  0  0  0  0  0  0  1  1 17]]
 
 
 
