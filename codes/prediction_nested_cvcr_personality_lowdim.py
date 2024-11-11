import numpy as np
import scipy
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import make_scorer, accuracy_score, r2_score, mean_squared_error
import sklearn
import sys

# The scikit-learn version is 1.3.2. (15th March 2024)
# The numpy version is 1.26.4.
# The scipy version is 1.12.0.
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The numpy version is {}.'.format(np.__version__))
print('The scipy version is {}.'.format(scipy.__version__))

# Argument
TRAIT_NAME = sys.argv[1]
print("Traits Personal " + TRAIT_NAME)

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
# -------------------------------------------------------------------------------------------------------------------------------------------------
# Features and targets
# -------------------------------------------------------------------------------------------------------------------------------------------------
# X: features [Schaefer_eFCeSC, HarvOxf_eFCeSC, Schaefer_sFCeFC, HarvOxf_sFCeFC, Schaefer_sFCeSC, HarvOxf_sFCeSC], Schaefer_sFCePL, HarvOxf_sFCePL]
# y: cognitive scores [90-160]
# V: brain volumes = cortical surface + subcortical areas + white matter (in cubic centimeter, cc)
# -------------------------------------------------------------------------------------------------------------------------------------------------
X = np.loadtxt('../data/Feature_for_Personality_N269_LowDim.csv', delimiter=',')
y = np.loadtxt('../data/Personal_' + TRAIT_NAME + '_for_Personality_N269_LowDim.csv', delimiter=',')
V = np.loadtxt('../data/BrainVolume_for_Personality_N269_LowDim.csv', delimiter=',')
SG = np.loadtxt('../data/SG_Personal_' + TRAIT_NAME + '_for_Personality_N269_LowDim.csv', delimiter=',')
AGE = np.loadtxt('../data/Age_for_Personality_N269_LowDim.csv', delimiter=',')
S = np.loadtxt('../data/Sex_Personal_for_Personality_N269_LowDim.csv', delimiter=',')
C = np.c_[np.ones((AGE.size, )), AGE, V]

def pearson_correlation(y, y_hat):
    r, p = scipy.stats.pearsonr(y, y_hat)
    return np.nan_to_num(r)

r_scorer = make_scorer(pearson_correlation)

# Number of repetitions of nested 5-fold CV
num_reps = 100
num_fold = 5

# Empirical features (eFC vs. eSC)
# X = X[:, (0,1)]
# CONDITION = "EMP"

# Simulated features (sFC vs. eFC)
# X = X[:, (2,3)]
# CONDITION = "SIM"

# Emprical and simulatied features (eFC vs. eSC, sFC vs. eFC)
# X = X[:, (0,1,2,3)]
# CONDITION = "EMPSIM"

# X = X[:, [0]]
# CONDITION = "EMP_SCH"
# X = X[:, [1]]
# CONDITION = "EMP_HO"
# X = X[:, [2]]
# CONDITION = "SIM_SCH"
X = X[:, [3]]
CONDITION = "SIM_HO"

# Regularization strengths in the Ridge (L2 penalty) regression
alpha = np.array([1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 1e+01, 1e+02, 1e+03, 1e+04, 1e+05, 1e+06])

# Nested cross-validation
# -----------------------
PREDICTED_Y = np.zeros([num_reps * num_fold, y.shape[0]])
TRAIN_INDEX = np.zeros([num_reps * num_fold, y.shape[0]])
REPORT_TRAIN = np.zeros([num_reps * num_fold, 3])
REPORT_TEST = np.zeros([num_reps * num_fold, 3])
BETA = np.zeros([num_reps * num_fold, X.shape[1]])
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
        # --- Multivariate CVCR ---
        C_train = C[train_index, :]
        B_train = np.dot(np.dot(np.linalg.inv(np.dot(C_train.T, C_train)), C_train.T), X_train)
        X_residuals = X - np.dot(C, B_train)
        # -------------------------
        # --- Univariate CVCR for multiple confounds ---
        # C_train1 = C[train_index, :][:, (0, 1)]
        # B_train1 = np.dot(np.dot(np.linalg.inv(np.dot(C_train1.T, C_train1)), C_train1.T), X_train)
        # X_residuals1 = X - np.dot(C[:, (0, 1)], B_train1)
        # C_train = C[train_index, :][:, (0, 2)]
        # B_train2 = np.dot(np.dot(np.linalg.inv(np.dot(C_train.T, C_train)), C_train.T), X_residuals1[train_index])
        # X_residuals = X_residuals1 - np.dot(C[:, (0, 2)], B_train2)
        # ----------------------------------------------
        X_residuals_train = X_residuals[train_index]
        Z = (X_residuals - X_residuals_train.mean(axis=0)) / X_residuals_train.std(axis=0, ddof=1)
        Z_train, Z_test = Z[train_index], Z[test_index]
        # CV Training (Ridge)
        # reg = LassoCV(cv=kf_in, n_alphas=100, random_state=(nIterOuter + (nIterRepet + 1) * skf_out.get_n_splits() * 2)).fit(Z_train, y_train)
        # reg = RidgeCV(cv=kf_in, alphas=alpha, scoring='neg_mean_squared_error').fit(Z_train, y_train)
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
pred_test_mean = np.mean(REPORT_TEST[:, 2])
pred_train_mean = np.mean(REPORT_TRAIN[:, 2])
print("Mean TEST  = " + f'{pred_test_mean:>25}')
print("Mean TRAIN = " + f'{pred_train_mean:>25}')

# Save results
# ------------
# np.savetxt('../prediction/HCP_N269_LowDim_NestedCV_CORR_Personal_' + TRAIT_NAME + '_' + CONDITION + '_PEARSON_R_TRAIN.txt', REPORT_TRAIN)
# np.savetxt('../prediction/HCP_N269_LowDim_NestedCV_CORR_Personal_' + TRAIT_NAME + '_' + CONDITION + '_PEARSON_R_TEST.txt', REPORT_TEST)
# np.savetxt('../prediction/HCP_N269_LowDim_NestedCV_CORR_Personal_' + TRAIT_NAME + '_' + CONDITION + '_CV_TRAIN_INDEX.txt', TRAIN_INDEX)
# np.savetxt('../prediction/HCP_N269_LowDim_NestedCV_CORR_Personal_' + TRAIT_NAME + '_' + CONDITION + '_PREDICTED_Y.txt', PREDICTED_Y)
# np.savetxt('../prediction/HCP_N269_LowDim_NestedCV_CORR_Personal_' + TRAIT_NAME + '_' + CONDITION + '_BETA.txt', BETA)
