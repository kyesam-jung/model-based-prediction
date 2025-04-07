import numpy as np
import scipy
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import make_scorer
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

# ---------------------------------------------------------------------------------------------------------------
# Features and targets
# ---------------------------------------------------------------------------------------------------------------
# X: features [Schaefer_eFCeSC, HarvOxf_eFCeSC, Schaefer_sFCeFC, HarvOxf_sFCeFC, Schaefer_sFCeSC, HarvOxf_sFCeSC]
# y: cognitive scores [90-160]
# V: brain volumes = cortical surface + subcortical areas + white matter (in cubic centimeter, cc)
# ---------------------------------------------------------------------------------------------------------------
X = np.loadtxt('../data/Feature_for_CogTotalComp_Unadj_N268_' + PARAMETER_DIM + '.csv', delimiter=',')
y = np.loadtxt('../data/CogTotalComp_Unadj_for_CogTotalComp_Unadj_N268_' + PARAMETER_DIM + '.csv', delimiter=',')
V = np.loadtxt('../data/BrainVolume_for_CogTotalComp_Unadj_N268_' + PARAMETER_DIM + '.csv', delimiter=',')
SG = np.loadtxt('../data/SG_CogTotalComp_Unadj_for_CogTotalComp_Unadj_N268_' + PARAMETER_DIM + '.csv', delimiter=',')
AGE = np.loadtxt('../data/Age_for_CogTotalComp_Unadj_N268_' + PARAMETER_DIM + '.csv', delimiter=',')

if CONDITION == "EMP":
    X = X[:, (0,1)]

if CONDITION == "SIM":
    X = X[:, (2,3)]

if CONDITION == "EMPSIM":
    X = X[:, (0,1,2,3)]

# Confound matrix
# ---------------
C = np.c_[np.ones((AGE.size, )), AGE, V]

# Goal function
# -------------
def pearson_correlation(y, y_hat):
    r, p = scipy.stats.pearsonr(y, y_hat)
    return np.nan_to_num(r)

r_scorer = make_scorer(pearson_correlation)

# Number of repetitions of nested 5-fold CV
# -----------------------------------------
num_reps = 100
num_fold = 5

# Regularization strengths in the Ridge (L2 penalty) regression
# -------------------------------------------------------------
alpha = np.array([1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 1e+01, 1e+02, 1e+03, 1e+04, 1e+05, 1e+06])

# Nested cross-validation
# -----------------------
PREDICTED_Y = np.zeros([num_reps * num_fold, y.shape[0]])
TRAIN_INDEX = np.zeros([num_reps * num_fold, y.shape[0]])
REPORT_TRAIN = np.zeros([num_reps * num_fold, 3])
REPORT_TEST = np.zeros([num_reps * num_fold, 3])
BETA = np.zeros([num_reps * num_fold, X.shape[1]])
SHAP_SUBJECT = []
SHAP_BASE = []
nFold = -1
for nIterRepet in range(num_reps):
    skf_out = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=nIterRepet)
    for nIterOuter, (train_index, test_index) in enumerate(skf_out.split(X, SG)):
        kf_in = KFold(n_splits=num_fold, shuffle=True, random_state=(nIterOuter + (nIterRepet + 1) * skf_out.get_n_splits()))
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
        # CV Training (Ridge)
        reg = RidgeCV(cv=kf_in, alphas=alpha, scoring=r_scorer).fit(Z_train, y_train)
        P = reg.predict(Z)
        PREDICTED_Y[nFold, :] = P[:, ]
        TRAIN_INDEX[nFold, train_index] = 1
        train_predictions = reg.predict(Z_train)
        test_predictions = reg.predict(Z_test)
        # PERFORMANCE - TRAIN
        r_train = pearson_correlation(y_train, train_predictions)
        # PERFORMANCE - TEST
        r_test = pearson_correlation(y_test, test_predictions)
        # SHAP
        explainer = shap.Explainer(reg.predict, Z_train)                # Initialize the Permutation explainer for the best model and training data
        max_evals = 2 ** np.shape(Z_train)[1] + 1                       # Number of evaluations for the permutation importance
        shap_explainer_object = explainer(Z_test, max_evals=max_evals)  # Calculate feature importances via permutation importance using the test data
        shap_values = shap_explainer_object.values
        base_value = shap_explainer_object.base_values[0]
        SHAP_SUBJECT.append(np.column_stack((np.full((test_index.shape[0], 1), nIterRepet), np.full((test_index.shape[0], 1), nIterOuter), test_index, shap_values)))
        SHAP_BASE.append([nIterRepet, nIterOuter, base_value])
        # SAVE and PRINT
        REPORT_TRAIN[nFold, :] = [nIterRepet, nIterOuter, r_train]
        REPORT_TEST[nFold, :] = [nIterRepet, nIterOuter, r_test]
        BETA[nFold, :] = reg.coef_
        print("Alpha = " + f'{reg.alpha_:>20}' + ", TRAIN r = " f'{r_train:>20}' + ", TEST r = " f'{r_test:>20}')

# Sanity check (it should be zero or infinitesimally small)
# ---------------------------------------------------------
for n in range(X_residuals_train.shape[1]):
    scipy.stats.pearsonr(C_train[:, 1], X_residuals_train[:, n])

# Print mean performace
# ---------------------
pred_train_mean = np.mean(REPORT_TRAIN[:, 2])
pred_test_mean = np.mean(REPORT_TEST[:, 2])
print("Mean TRAIN = " + f'{pred_train_mean:>25}')
print("Mean TEST  = " + f'{pred_test_mean:>25}')

# Change data type to Numpy array
# -------------------------------
SHAP_SUBJECT_NP = np.vstack(SHAP_SUBJECT)
SHAP_BASE_NP = np.vstack(SHAP_BASE)

# Save results
# ------------
np.savetxt('../prediction/HCP_N268_' + PARAMETER_DIM + '_NestedCV_CORR_COG_' + CONDITION + '_PEARSON_R_TRAIN.txt', REPORT_TRAIN, fmt=["%.1f", "%.1f", "%.18f"])
np.savetxt('../prediction/HCP_N268_' + PARAMETER_DIM + '_NestedCV_CORR_COG_' + CONDITION + '_PEARSON_R_TEST.txt', REPORT_TEST, fmt=["%.1f", "%.1f", "%.18f"])
np.savetxt('../prediction/HCP_N268_' + PARAMETER_DIM + '_NestedCV_CORR_COG_' + CONDITION + '_CV_TRAIN_INDEX.txt', TRAIN_INDEX, fmt="%.1f")
np.savetxt('../prediction/HCP_N268_' + PARAMETER_DIM + '_NestedCV_CORR_COG_' + CONDITION + '_PREDICTED_Y.txt', PREDICTED_Y)
np.savetxt('../prediction/HCP_N268_' + PARAMETER_DIM + '_NestedCV_CORR_COG_' + CONDITION + '_BETA.txt', BETA)

if CONDITION == "EMPSIM":
    np.savetxt('../prediction/HCP_N268_' + PARAMETER_DIM + '_NestedCV_CORR_COG_' + CONDITION + '_SHAP_SUBJECT.txt', SHAP_SUBJECT_NP, fmt=["%.1f", "%.1f", "%.1f", "%.18f", "%.18f", "%.18f", "%.18f"])

if CONDITION == "EMP" or CONDITION == "SIM":
    np.savetxt('../prediction/HCP_N268_' + PARAMETER_DIM + '_NestedCV_CORR_COG_' + CONDITION + '_SHAP_SUBJECT.txt', SHAP_SUBJECT_NP, fmt=["%.1f", "%.1f", "%.1f", "%.18f", "%.18f"])

np.savetxt('../prediction/HCP_N268_' + PARAMETER_DIM + '_NestedCV_CORR_COG_' + CONDITION + '_SHAP_BASE.txt', SHAP_BASE_NP, fmt=["%.1f", "%.1f", "%.18f"])
