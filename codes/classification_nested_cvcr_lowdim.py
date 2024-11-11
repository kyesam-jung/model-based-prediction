import numpy as np
import scipy
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import sklearn

# The scikit-learn version is 1.3.2. (15th March 2024)
# The numpy version is 1.26.4.
# The scipy version is 1.12.0.
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The numpy version is {}.'.format(np.__version__))
print('The scipy version is {}.'.format(scipy.__version__))

# scikit-learn preparation (MacOS Monterey 12.7.1 / homebrew package manager) ----
# brew install python
# pip3 install pandas
# pip3 install sklearn
# pip3 install scipy
# /Library/Developer/CommandLineTools/usr/bin/python3 -m pip install --upgrade pip
# --------------------------------------------------------------------------------
# On juseless in INM-7 (https://docs.inm7.de/learning/programming_python/)
# python3 -m venv ~/.venvs/classification
# source ~/.venvs/classification/bin/activate
# ~/.venvs/classification/bin/python3 -m pip install --upgrade pip
# pip install numpy
# pip install scipy
# pip install scikit-learn
# deactivate
# ---------------------------------------------------------------------------------------------------------------
# Features and targets
# ---------------------------------------------------------------------------------------------------------------
# X: features [Schaefer_eFCeSC, HarvOxf_eFCeSC, Schaefer_sFCeFC, HarvOxf_sFCeFC, Schaefer_sFCeSC, HarvOxf_sFCeSC]
# y: targets (0: female, 1: male)
# V: brain volumes = cortical surface + subcortical areas + white matter (in cubic centimeter, cc)
# ---------------------------------------------------------------------------------------------------------------
X = np.loadtxt('../data/Feature_for_Classification_N270_LowDim.csv', delimiter=',')
y = np.loadtxt('../data/Female_Male_for_Classification_N270_LowDim.csv', delimiter=',')
V = np.loadtxt('../data/BrainVolume_for_Classification_N270_LowDim.csv', delimiter=',')
C = np.c_[np.ones((V.size, )), V]

# Number of repetitions of nested 5-fold CV
num_reps = 100
num_fold = 5

# Empirical features (eFC vs. eSC)
X = X[:, (0,1)]
CONDITION = "EMP"

# Simulated features (sFC vs. eFC)
# X = X[:, (2,3)]
# CONDITION = "SIM"

# Emprical and simulated features (eFC vs. eSC, sFC vs. eFC)
# X = X[:, (0,1,2,3)]
# CONDITION = "EMPSIM"

# X = X[:, [0]]
# CONDITION = "EMP_SCH"
# X = X[:, [1]]
# CONDITION = "EMP_HO"
# X = X[:, [2]]
# CONDITION = "SIM_SCH"
# X = X[:, [3]]
# CONDITION = "SIM_HO"

# Nested cross-validation
# -----------------------
PROBABILITY = np.zeros([num_reps * num_fold, y.shape[0]])
TRAIN_INDEX = np.zeros([num_reps * num_fold, y.shape[0]])
REPORT_TRAIN = np.zeros([num_reps * num_fold, 3])
REPORT_TEST = np.zeros([num_reps * num_fold, 3])
BETA = np.zeros([num_reps * num_fold, X.shape[1]])
nFold = -1
for nIterRepet in range(num_reps):
    skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=nIterRepet)
    for nIterOuter, (train_index, test_index) in enumerate(skf.split(X, y)):
        nFold = nFold + 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # CVCR-ZAC
        # CVCR (cross-validated confound removal) - ZAC (z-scoring after confound removal)
        C_train = C[train_index, :]
        B_train = np.dot(np.dot(np.linalg.inv(np.dot(C_train.T, C_train)), C_train.T), X_train)
        X_residuals = X - np.dot(C, B_train)
        X_residuals_train = X_residuals[train_index]
        Z = (X_residuals - X_residuals_train.mean(axis=0)) / X_residuals_train.std(axis=0, ddof=1)
        Z_train, Z_test = Z[train_index], Z[test_index]
        # CV Training (l1 or l2 penalty)
        # clf = LogisticRegressionCV(cv=num_fold, penalty='l1', solver='liblinear', max_iter=1000, random_state=(nIterOuter + (nIterRepet + 1) * skf.get_n_splits())).fit(Z_train, y_train)
        clf = LogisticRegressionCV(cv=num_fold, penalty='l2', solver='lbfgs', max_iter=1000, random_state=(nIterOuter + (nIterRepet + 1) * skf.get_n_splits())).fit(Z_train, y_train)
        P = clf.predict_proba(Z)
        PROBABILITY[nFold, :] = P[:, 1]
        TRAIN_INDEX[nFold, train_index] = 1
        train_predictions = clf.predict(Z_train)
        test_predictions = clf.predict(Z_test)
        # PERFORMANCE - TRAIN
        # acc_train = accuracy_score(y_train, train_predictions)
        acc_train = balanced_accuracy_score(y_train, train_predictions)
        # PERFORMANCE - TEST
        # acc_test = accuracy_score(y_test, test_predictions)
        acc_test = balanced_accuracy_score(y_test, test_predictions)
        # SAVE and PRINT
        REPORT_TRAIN[nFold, :] = [nIterRepet, nIterOuter, acc_train]
        REPORT_TEST[nFold, :] = [nIterRepet, nIterOuter, acc_test]
        BETA[nFold, :] = clf.coef_
        print("PARAMETER 1/C = " f'{1 / clf.C_[0]:>25}' + ", TRAIN ACCURACY = " f'{acc_train:>20}' + ", TEST ACCURACY = " f'{acc_test:>20}')

# Sanity check (it should be zero or infinitesimally small)
# ---------------------------------------------------------
for n in range(X_residuals_train.shape[1]):
    scipy.stats.pearsonr(C_train[:, 1], X_residuals_train[:, n])

# Print mean performace
# ---------------------
acc_test_mean = np.mean(REPORT_TEST[:, 2])
acc_train_mean = np.mean(REPORT_TRAIN[:, 2])
print("Mean TEST  = " + f'{acc_test_mean:>25}')
print("Mean TRAIN = " + f'{acc_train_mean:>25}')

# Save results
# ------------
# np.savetxt('../classification/HCP_N270_LowDim_NestedCV_' + CONDITION + '_ACCURACY_TRAIN.txt', REPORT_TRAIN)
# np.savetxt('../classification/HCP_N270_LowDim_NestedCV_' + CONDITION + '_ACCURACY_TEST.txt', REPORT_TEST)
# np.savetxt('../classification/HCP_N270_LowDim_NestedCV_' + CONDITION + '_CV_TRAIN_INDEX.txt', TRAIN_INDEX)
# np.savetxt('../classification/HCP_N270_LowDim_NestedCV_' + CONDITION + '_PROBA.txt', PROBABILITY)
# np.savetxt('../classification/HCP_N270_LowDim_NestedCV_' + CONDITION + '_BETA.txt', BETA)
