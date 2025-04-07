import numpy as np
import scipy
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import sklearn
import sys
import shap

print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The numpy version is {}.'.format(np.__version__))
print('The scipy version is {}.'.format(scipy.__version__))
print('The shap version is {}.'.format(shap.__version__))

# Input arguments
# ---------------
CONDITION = sys.argv[1]
print("CONDITION = " + CONDITION)
PARAMETER_DIM = sys.argv[2]
print("PARAMETER_DIM = " + PARAMETER_DIM)
if CONDITION == "EMP":
    PARAMETER_DIM = "LowDim"

# ------------------------------------------------------------------------------------------------
# Features and targets
# ------------------------------------------------------------------------------------------------
# X: features [Schaefer_eFCeSC, HarvOxf_eFCeSC, Schaefer_sFCeFC, HarvOxf_sFCeFC]
# y: targets (0: female, 1: male)
# V: brain volumes = cortical surface + subcortical areas + white matter (in cubic centimeter, cc)
# ------------------------------------------------------------------------------------------------
X = np.loadtxt('../data/Feature_for_Classification_N270_' + PARAMETER_DIM + '.csv', delimiter=',')
y = np.loadtxt('../data/Female_Male_for_Classification_N270_' + PARAMETER_DIM + '.csv', delimiter=',')
V = np.loadtxt('../data/BrainVolume_for_Classification_N270_' + PARAMETER_DIM + '.csv', delimiter=',')

if CONDITION == "EMP":
    X = X[:, (0,1)]

if CONDITION == "SIM":
    X = X[:, (2,3)]

if CONDITION == "EMPSIM":
    X = X[:, (0,1,2,3)]

# Confounds
# ---------
C = np.c_[np.ones((V.size, )), V]

# Number of repetitions of nested 5-fold CV
# -----------------------------------------
num_reps = 100
num_fold = 5

# Nested cross-validation
# -----------------------
PROBABILITY = np.zeros([num_reps * num_fold, y.shape[0]])
TRAIN_INDEX = np.zeros([num_reps * num_fold, y.shape[0]])
REPORT_TRAIN = np.zeros([num_reps * num_fold, 3])
REPORT_TEST = np.zeros([num_reps * num_fold, 3])
BETA = np.zeros([num_reps * num_fold, X.shape[1]])
SHAP_SUBJECT = []
SHAP_BASE = []
nFold = -1
for nIterRepet in range(num_reps):
    np.random.seed(nIterRepet) # Random seed for permutation test (0, 1, 2, ..., 99)
    y = np.random.permutation(y)
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
        acc_train = balanced_accuracy_score(y_train, train_predictions)
        # PERFORMANCE - TEST
        acc_test = balanced_accuracy_score(y_test, test_predictions)
        # SHAP
        explainer = shap.Explainer(clf.predict, Z_train)                # Initialize the Permutation explainer for the best model and training data
        max_evals = 2 ** np.shape(Z_train)[1] + 1                       # Number of evaluations for the permutation importance
        shap_explainer_object = explainer(Z_test, max_evals=max_evals)  # Calculate feature importances via permutation importance using the test data
        shap_values = shap_explainer_object.values                      # Get the SHAP values
        base_value = shap_explainer_object.base_values[0]               # Get the base value
        SHAP_SUBJECT.append(np.column_stack((np.full((test_index.shape[0], 1), nIterRepet), np.full((test_index.shape[0], 1), nIterOuter), test_index, shap_values)))
        SHAP_BASE.append([nIterRepet, nIterOuter, base_value])
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
acc_train_mean = np.mean(REPORT_TRAIN[:, 2])
acc_test_mean = np.mean(REPORT_TEST[:, 2])
print("Mean TRAIN = " + f'{acc_train_mean:>25}')
print("Mean TEST  = " + f'{acc_test_mean:>25}')

# Change data type to Numpy array
# -------------------------------
SHAP_SUBJECT_NP = np.vstack(SHAP_SUBJECT)
SHAP_BASE_NP = np.vstack(SHAP_BASE)

# Save results
# ------------
np.savetxt('../classification_permutation_test/HCP_N270_' + PARAMETER_DIM + '_NestedCV_' + CONDITION + '_ACCURACY_TRAIN.txt', REPORT_TRAIN, fmt=["%.1f", "%.1f", "%.18f"])
np.savetxt('../classification_permutation_test/HCP_N270_' + PARAMETER_DIM + '_NestedCV_' + CONDITION + '_ACCURACY_TEST.txt', REPORT_TEST, fmt=["%.1f", "%.1f", "%.18f"])
np.savetxt('../classification_permutation_test/HCP_N270_' + PARAMETER_DIM + '_NestedCV_' + CONDITION + '_CV_TRAIN_INDEX.txt', TRAIN_INDEX, fmt="%.1f")
np.savetxt('../classification_permutation_test/HCP_N270_' + PARAMETER_DIM + '_NestedCV_' + CONDITION + '_PROBA.txt', PROBABILITY)
np.savetxt('../classification_permutation_test/HCP_N270_' + PARAMETER_DIM + '_NestedCV_' + CONDITION + '_BETA.txt', BETA)

if CONDITION == "EMPSIM":
    np.savetxt('../classification_permutation_test/HCP_N270_' + PARAMETER_DIM + '_NestedCV_' + CONDITION + '_SHAP_SUBJECT.txt', SHAP_SUBJECT_NP, fmt=["%.1f", "%.1f", "%.1f", "%.18f", "%.18f", "%.18f", "%.18f"])

if CONDITION == "EMP" or CONDITION == "SIM":
    np.savetxt('../classification_permutation_test/HCP_N270_' + PARAMETER_DIM + '_NestedCV_' + CONDITION + '_SHAP_SUBJECT.txt', SHAP_SUBJECT_NP, fmt=["%.1f", "%.1f", "%.1f", "%.18f", "%.18f"])

np.savetxt('../classification_permutation_test/HCP_N270_' + PARAMETER_DIM + '_NestedCV_' + CONDITION + '_SHAP_BASE.txt', SHAP_BASE_NP, fmt=["%.1f", "%.1f", "%.18f"])
