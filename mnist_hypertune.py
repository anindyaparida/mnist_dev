print(__doc__)
import os
import matplotlib.pyplot as plt
from numpy import mean
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from skimage import data, color
from skimage.transform import rescale
import numpy as np

from joblib import dump, load


digits = datasets.load_digits()

n_samples = len(digits.images)

rescale_factors = [1]
for i in range(3):
    for test_size, valid_size in [(0.15, 0.15)]:
        for rescale_factor in rescale_factors:
            model_candidates = []
            for gamma in [10 ** exponent for exponent in range(-7, 0)]:
                resized_images = []
                for d in digits.images:
                    resized_images.append(rescale(d, rescale_factor, anti_aliasing=False))

                resized_images = np.array(resized_images)
                data = resized_images.reshape((n_samples, -1))

            # SVM classifier

                clf = svm.SVC(gamma=gamma, C=2, degree=3)

                X_train, X_test_valid, y_train, y_test_valid = train_test_split(
                    data, digits.target, test_size=test_size + valid_size, shuffle=False
                )

                X_test, X_valid, y_test, y_valid = train_test_split(
                    X_test_valid,
                    y_test_valid,
                    test_size=valid_size / (test_size + valid_size),
                    shuffle=False,
                )

                clf.fit(X_train, y_train)
                predicted_valid = clf.predict(X_valid)
                acc_valid = metrics.accuracy_score(y_pred=predicted_valid, y_true=y_valid)
                f1_valid = metrics.f1_score(
                    y_pred=predicted_valid, y_true=y_valid, average="macro"
                )

                if acc_valid < 0.11:
                    print("Skipping for {}".format(gamma))
                    continue

                candidate = {
                    "acc_valid": acc_valid,
                    "f1_valid": f1_valid,
                    "gamma": gamma,
                }
                model_candidates.append(candidate)
                output_folder = "./models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                   test_size, valid_size, rescale_factor, gamma
                )
                #os.mkdir(output_folder)
                #dump(clf, os.path.join(output_folder, "model.joblib"))

        # Predict the value of the digit

            max_valid_f1_model_candidate = max(
                model_candidates, key=lambda x: x["f1_valid"]
            )
            best_model_folder = "./models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, valid_size, rescale_factor, max_valid_f1_model_candidate['gamma']
            )
            clf = load(os.path.join(best_model_folder, "model.joblib"))
            predicted = clf.predict(X_test)

            acc = metrics.accuracy_score(y_pred=predicted, y_true=y_test)
            f1 = metrics.f1_score(y_pred=predicted, y_true=y_test, average="macro")
            mean= y_train.data.mean(), y_train.data.std()
            print(mean)
            print(
                "{}x{}\t{}\t{}:{}\t{:.3f}\t{:.3f}".format(
                    resized_images[0].shape[0],
                    resized_images[0].shape[1],
                    max_valid_f1_model_candidate["gamma"],
                    (1 - test_size) * 100,
                    test_size * 100,
                    acc,
                    f1,
                )
            )
