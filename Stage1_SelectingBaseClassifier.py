'Phase 1: Selecting best classifier'
'Record performance on each run'

import math
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from sklearn.model_selection import cross_val_predict, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.metrics import precision_recall_fscore_support as score, f1_score, precision_score, recall_score, \
    accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler


from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.metrics import geometric_mean_score
from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier




from sklearn.ensemble import VotingClassifier



####################################
'STEP 1: Import Dataset (Imbalance)'
####################################
MyDataset = pd.read_excel('path..../Data/Cleveland0vs4/Cleveland0vs4_Cleaned.xlsx')
# MyDataset = pd.read_excel('path..../Data/eColi4/eColi4_Cleaned.xlsx')
# MyDataset = pd.read_excel('path..../Data/Yeast3/Yeast3_Cleaned.xlsx')
# MyDataset = pd.read_excel('path..../Data/SPECT/SPECT_Cleaned.xlsx')
# MyDataset = pd.read_excel('path..../Data/SPECTF/SPECTF_Cleaned.xlsx')
# MyDataset = pd.read_excel('path..../Data/Parkinson/Parkinson_Cleaned.xlsx')
# MyDataset = pd.read_excel('path..../Data/Parkinson/MIMIC/MIMIC_Cleaned.xlsx')
# MyDataset = pd.read_excel('path..../Data/Parkinson/Stroke/Stroke_Cleaned.xlsx')


#########################################################################
'STEP 2: Assign X(data) & y (target)'
#########################################################################
#AllData
X = MyDataset.drop('Class', axis=1)
y = MyDataset['Class']


####################################
'STEP 3: Experimental Models, uncomment to use'
####################################

'Exp 1'
LR_model = LogisticRegression(solver='lbfgs',max_iter=2500)
# LSVM_model = SVC(kernel='linear')
# RSVM_model = SVC(kernel='rbf')
# DT_model = DecisionTreeClassifier(random_state=7)


############################################################
'Exp 2'
LR_model = LogisticRegression(solver='lbfgs',max_iter=2500)
LSVM_model = SVC(kernel='linear')
RSVM_model = SVC(kernel='rbf')
DT_model = DecisionTreeClassifier(random_state=7)

estimators = []
estimators.append(('LR', LR_model))
estimators.append(('LSVM', LSVM_model))
estimators.append(('RSVM', RSVM_model))
estimators.append(('DT', DT_model))

ensemble_EXP2 = VotingClassifier(estimators, voting = 'hard')



############################################################
'Exp 3'
DT_model = DecisionTreeClassifier(random_state=7)
XGB_model = XGBClassifier(n_estimators=100, random_state=7, n_jobs=-1)
RSVM_model = SVC(kernel='rbf', probability=True)

# ## Ensemble method: Majority voting
estimators = []
estimators.append(('RSVM', RSVM_model))
estimators.append(('DT', DT_model))
estimators.append(('XGB', XGB_model))

ensemble_base = VotingClassifier(estimators=estimators)
# Apply Self-paced Ensemble
ensemble_EXP3 = SelfPacedEnsembleClassifier(random_state=42, base_estimator=ensemble_base, n_jobs=-1)




####################################
'Extra: Cross Validation'
####################################
'CV Method: (LOOCV)'
# --------------------------------------------------------------------------------------------
'Store Results: Average'
y_test_dec=[] #Store y_test for every split
y_pred_dec=[] #Store y_pred for every split

'LOOCV'
loo_cv = LeaveOneOut()
for fold, (train_index, test_index) in enumerate(loo_cv.split(X,y), 1):
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    'Standardize, Train/Test each fold'
    standard_scaler = StandardScaler()
    standard_scaler.fit(X_train)
    x_train_scaler = standard_scaler.transform(X_train)
    x_test_scaler = standard_scaler.transform(X_test)
    TheModel = LR_model  #Insert Experimental models here (eg; Exp1: LR, L.SVM, R.SVM, DT| Exp2: ensemble_EXP2| Exp3: ensemble_EXP3)
    TheModel.fit(x_train_scaler, y_train)
    y_pred = TheModel.predict(x_test_scaler)
    '-------------------------------------'

    print(f'Fold {fold}')

    'Store in list'
    y_test_dec.append(y_test.to_numpy()[0])
    y_pred_dec.append(y_pred)



my_acc = (accuracy_score(y_test_dec,y_pred_dec))*100
my_prec = (precision_score(y_test_dec,y_pred_dec))*100
my_Sens = (recall_score(y_test_dec,y_pred_dec))*100
my_Spec = (recall_score(y_test_dec,y_pred_dec, pos_label=0))*100
my_F1 = (f1_score(y_test_dec,y_pred_dec))*100
my_gmean = (geometric_mean_score(np.array(y_test_dec),np.array(y_pred_dec).ravel()))*100
my_auroc = (roc_auc_score(y_test_dec,y_pred_dec))*100

print("Results: \n")
print("{:.2f}%".format(my_acc), "{:.2f}%".format(my_prec), "{:.2f}%".format(my_Sens),"{:.2f}%".format(my_Spec),"{:.2f}%".format(my_F1),"{:.2f}%".format(my_gmean),"{:.2f}%".format(my_auroc))

