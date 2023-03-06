'Phase 2: Selecting best rebalancing strategy'
'Record performance on each rebalancing strategies and choose the best'

import pandas as pd
import numpy as np
import pickle
from collections import Counter
import time
from numpy import mean
from numpy import std

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
import math


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import ConfusionMatrixDisplay, make_scorer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from matplotlib import pyplot
from numpy import where
from pycm import *
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.metrics import geometric_mean_score

'Import metric'
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneOut




'Import Rebalancing'
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

'Analysis Library'
from yellowbrick.classifier import ROCAUC





from sklearn.ensemble import VotingClassifier

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn import metrics

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


#########################################################################
'STEP 3: Assign Model Experiments'
#########################################################################
'Model Exp1'
LR_model = LogisticRegression(max_iter=2500)
# LSVM_model = SVC(kernel='linear')
# RSVM_model = SVC(kernel='rbf')
# DT_model = DecisionTreeClassifier(random_state=7)

# CSL: On for CSL
# weights = {0:1.5}
# LR_model = LogisticRegression(class_weight=weights, max_iter=2500)
# RSVM_model = SVC(kernel='rbf', class_weight=weights)
# LSVM_model = SVC(kernel='linear', class_weight=weights)
# DT_model = DecisionTreeClassifier(random_state=7, class_weight=weights)

#-----------------------------------------------#

'Model Exp2'
LR_model = LogisticRegression(max_iter=2500)
LSVM_model = SVC(kernel='linear')
RSVM_model = SVC(kernel='rbf')
DT_model = DecisionTreeClassifier(random_state=7)
 
# # CSL: On for CSL
# weights = {1:0.25}
# LR_model = LogisticRegression(max_iter=2500, class_weight=weights)
# LSVM_model = SVC(kernel='linear', class_weight=weights)
# RSVM_model = SVC(kernel='rbf', class_weight=weights)
# DT_model = DecisionTreeClassifier(random_state=7, class_weight=weights)
# 
estimators = []
estimators.append(('LR', LR_model))
estimators.append(('LSVM', LSVM_model))
estimators.append(('RSVM', RSVM_model))
estimators.append(('DT', DT_model))
ensemble_EXP2 = VotingClassifier(estimators)



#-----------------------------------------------#
'Model Exp3'
RSVM_model = SVC(kernel='rbf', probability = True)
DT_model = DecisionTreeClassifier(random_state=7)
XGB_model = XGBClassifier(n_estimators=100, random_state=7, n_jobs=-1)


# CSL: On for CSL
#weights = {0:0.5}
#RSVM_model = SVC(kernel='rbf', probability = True, class_weight= weights)
#DT_model = DecisionTreeClassifier(random_state=7, class_weight= weights)
#XGB_model = XGBClassifier(n_estimators=100, random_state=7, n_jobs=-1)
#
estimators = []
estimators.append(('RSVM', RSVM_model))
estimators.append(('DT', DT_model))
estimators.append(('XGB', XGB_model))
ensemble_base = VotingClassifier(estimators=estimators)
# # Apply SPE+
ensemble_EXP3 = SelfPacedEnsembleClassifier(random_state=42, base_estimator=ensemble_base, n_jobs=-1)



#########################################################################
'STEP 4: Rebalancing, Adjust by parameters'
#########################################################################
'Oversampling'
# strategy_setup = 0.75 #Minority
# samplingtype = RandomOverSampler(sampling_strategy=strategy_setup)

'Undersampling'
# strategy_setup = 0.28 #Majority
# samplingtype = RandomUnderSampler(sampling_strategy=strategy_setup)

'SMOTE with Undersample'
# smote_strategy = 0.68 #Minority
# smoted = SMOTE(sampling_strategy=smote_strategy)
# under_strategy = {0:496} #Majority
# undersample = RandomUnderSampler(sampling_strategy=under_strategy)





#########################################################################
'STEP 4: Cross Validation'
'Train/split by cv, rebalance on train set and test by each fold'
#########################################################################
'CV Method: (LOOCV)'
y_test_dec=[] #Store y_test for every split
y_pred_dec=[] #Store y_pred for every split

loo_cv = LeaveOneOut()
for fold, (train_index, test_index) in enumerate(loo_cv.split(X,y), 1):
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    

    'Over&under-sampling: uncomment this to use'
    X_train_rebalance, y_train_rebalance = samplingtype.fit_resample(X_train, y_train)
    'SMOTE: uncomment this to use'
    # X_smote, y_smote = smoted.fit_resample(X_train, y_train)
    # X_train_rebalance, y_train_rebalance = undersample.fit_resample(X_smote, y_smote)

    'Standardize -> Rebalance train set -> Test each fold -> repeat next fold'
    standard_scaler = StandardScaler()
    standard_scaler.fit(X_train_rebalance)
    x_train_scaler = standard_scaler.transform(X_train_rebalance)
    x_test_scaler = standard_scaler.transform(X_test)
    TheModel = ensemble_EXP3 #Insert Experimental models here (eg; Exp1: LR, L.SVM, R.SVM, DT| Exp2: ensemble_EXP2| Exp3: ensemble_EXP3)
    TheModel.fit(x_train_scaler, y_train_rebalance)
    y_pred = TheModel.predict(x_test_scaler)
    '-------------------------------------'

    print(f'Fold {fold}')

    'Store in list'
    y_test_dec.append(y_test.to_numpy()[0])
    y_pred_dec.append(y_pred)




# Stroke: prec, rec, f1 = average='weighted'
my_acc = (accuracy_score(y_test_dec,y_pred_dec))*100
my_prec = (precision_score(y_test_dec,y_pred_dec))*100
my_Sens = (recall_score(y_test_dec,y_pred_dec))*100
my_Spec = (recall_score(y_test_dec,y_pred_dec, pos_label=0))*100
my_F1 = (f1_score(y_test_dec,y_pred_dec))*100
my_gmean = (geometric_mean_score(np.array(y_test_dec),np.array(y_pred_dec).ravel()))*100
my_auroc = (roc_auc_score(y_test_dec,y_pred_dec))*100

print("Results: \n")
print("{:.2f}%".format(my_acc), "{:.2f}%".format(my_prec), "{:.2f}%".format(my_Sens),"{:.2f}%".format(my_Spec),"{:.2f}%".format(my_F1),"{:.2f}%".format(my_gmean),"{:.2f}%".format(my_auroc))
print(classification_report(y_test_dec,y_pred_dec,labels=[1,0]))










