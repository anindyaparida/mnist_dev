
print(__doc__)

import os
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
from skimage import data, color
from skimage.transform import rescale
import numpy as np

from joblib import dump, load

###############################################################################
# Digits dataset
# --------------

digits = datasets.load_digits()

n_samples = len(digits.images)

# rescale_factors = [0.25, 0.5, 1, 2, 3]
rescale_factors = [1]
for test_size, valid_size in [(0.10, 0.10)]:
    for rescale_factor in rescale_factors:
        model_candidates = []
        model_candidates1 = []
        maxdepth = [2,4,6,8,10]
        gammavalue = [0.01,0.001,0.0001]
        for gamma in gammavalue:
            resized_images = []
            for d in digits.images:
                resized_images.append(rescale(d, rescale_factor, anti_aliasing=False))

            resized_images = np.array(resized_images)
            data = resized_images.reshape((n_samples, -1))

            # Create a classifier: a support vector classifier
            clf = svm.SVC(gamma=gamma)
            #clf = tree.DecisionTreeClassifier(max_depth=4)

            X_train, X_test_valid, y_train, y_test_valid = train_test_split(
                data, digits.target, test_size=test_size + valid_size, shuffle=False
            )

            X_test, X_valid, y_test, y_valid = train_test_split(
                X_test_valid,
                y_test_valid,
                test_size=valid_size / (test_size + valid_size),
                shuffle=False,
            )

            # print("Number of samples: Train:Valid:Test = {}:{}:{}".format(len(y_train),len(y_valid),len(y_test)))

            # Learn the digits on the train subset
            clf.fit(X_train, y_train)
            predicted_valid = clf.predict(X_valid)
            acc_valid = metrics.accuracy_score(y_pred=predicted_valid, y_true=y_valid)
            f1_valid = metrics.f1_score(
                y_pred=predicted_valid, y_true=y_valid, average="macro")
            # we will ensure to throw away some of the models that yield random-like performance.
            if acc_valid < 0.11:
                print("Skipping for {}".format(gamma))
                continue

            candidate = {
                "acc_valid": acc_valid,
                "f1_valid": f1_valid,
                "gamma": gamma,
            }

            model_candidates.append(candidate)
            print(candidate)
            output_folder = "./models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, valid_size, rescale_factor, gamma
            )
            #os.mkdir(output_folder)
            #dump(clf, os.path.join(output_folder, "model.joblib"))

        # Predict the value of the digit on the test subset

        max_valid_f1_model_candidate = max(
            model_candidates, key=lambda x: x["f1_valid"]
        )

        best_model_folder = "./models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
            test_size, valid_size, rescale_factor, max_valid_f1_model_candidate['gamma']
        )
        #clf = load(os.path.join(best_model_folder, "model.joblib"))
        predicted = clf.predict(X_test)

        _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
        for ax, image, prediction in zip(axes, X_test, predicted):
            ax.set_axis_off()
            image = image.reshape(8, 8)
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            ax.set_title(f'Prediction: {prediction}')

        acc = metrics.accuracy_score(y_pred=predicted, y_true=y_test)
        f1 = metrics.f1_score(y_pred=predicted, y_true=y_test, average="macro")
        # print(
        #     "{:15s}\t{}x{}\t{}\t{}:{}\t{:.3f}\t{:.3f}".format('SVM',
        #         resized_images[0].shape[0],
        #         resized_images[0].shape[1],
        #         max_valid_f1_model_candidate["gamma"],
        #         (1 - test_size) * 100,
        #         test_size * 100,
        #         acc,
        #         f1,
        #     )
        # )
        print(f"Classification report :  {clf}:\n"
              f"{metrics.classification_report(y_test, predicted)}\n")

        disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
        disp.figure_.suptitle("Confusion Matrix")
        print(f"Confusion matrix:\n{disp.confusion_matrix}")