import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, BayesianRidge, ridge_regression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    precision_recall_curve,
    auc,
    cohen_kappa_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
    roc_curve,
    brier_score_loss,
    accuracy_score
)
import shap
import pyreadstat
from alibi.explainers import ALE, plot_ale
from sklearn.compose import make_column_selector as selector
import copy
import lightgbm as lgb

# from imbens.utils._plot import plot_2Dprojection_and_cardinality
from sklearn.model_selection import cross_validate

dataset = pd.read_csv("../final_data.csv")
X = dataset[['hypertension', 'dyslipidemia', 'liver_disease', 'kidney_disease', 'arthritis', 'digestive_disease', 
                 'sleep_duration', 'smoking', 'drinking', 'BMI_status']]
Y = dataset['successful_aging']


# 明确指定类别变量和数值变量
categorical_columns = ['hypertension', 'dyslipidemia', 'liver_disease', 'kidney_disease', 
                       'arthritis', 'digestive_disease', 'smoking', 'drinking', 'BMI_status']
numerical_columns = ['sleep_duration'] 

# minmax 处理连续变量，放缩0-1区间，不影响哑变量编码
numerical_preprocessor = MinMaxScaler()
dataset[numerical_columns] = numerical_preprocessor.fit_transform(dataset[numerical_columns])

# pd dummy 处理分类变量，哑变量第一列删除
cate_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True, dtype=float)

datasetX = copy.deepcopy(cate_encoded)

Models = {
    'XGB':XGBClassifier(max_depth=9,
    min_child_weight=6,
    gamma=3,
    learning_rate=0.3417813803467016,
    subsample=0.820724210008386,
    colsample_bytree=0.5116227290149833,
    eval_metric='logloss',
    scale_pos_weight=2,
    max_delta_step=2
),

    'GBDT':GradientBoostingClassifier(n_estimators=10,
    max_depth=29,
    min_samples_split=7,
    min_samples_leaf=3,
    min_weight_fraction_leaf=0.06618028848966474
),

    'RF':RandomForestClassifier(n_estimators=18,
    max_depth=22,
    min_samples_split=4,
    min_samples_leaf=9,
    class_weight=None
),

    'LR':LogisticRegression(
    tol=0.00019052345948076022,
    C=0.9164334852811108,
    fit_intercept=True,
    random_state=42,
    class_weight='balanced'

),

    'LightGBM':lgb.LGBMClassifier(random_state=555,
    n_estimators=5978,
    reg_alpha=2.739959780469101,
    reg_lambda=0.058179694801695096,
    colsample_bytree=0.6,
    subsample=1.0,
    learning_rate=0.006,
    max_depth=100,
    num_leaves=245,
    min_child_samples=39,
    min_data_per_groups=99
),
    #'AdaBoost': AdaBoostClassifier(random_state=23333),
    'CatBoost':CatBoostClassifier(eval_metric=None,
    l2_leaf_reg=0.16824334777612132,
    learning_rate=0.007347588201707157,
    n_estimators=278,
    depth=4,
    scale_pos_weight=55,
    min_data_in_leaf=145
)
}

minmaxnorm = False
SMOTE_Process = False

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
X_train, X_test, y_train, y_test = train_test_split(datasetX,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=23333,
                                                    shuffle = True)
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23333)
compare_result = ['Model', 'Pre_Mean', 'Pre_Std','Recall_Mean', 'Recall_Std',
                  'F1_Mean', 'F1_Std', 'ROCAUC_Mean', 'ROCAUC_Std', 
                  'PRAUC_Mean', 'PRAUC_Std','Brier_Mean', 'Brier_Std',
                  'ACC_Mean', 'ACC_Std']
for model_name in Models:
    print(model_name)
    model = Models[model_name]
    mid_results = ['precision', 'recall', 'f1', 'ROC_AUC', 'aucpr', 'brier', 'acc']
    for train, test in kfold.split(X_train, y_train):
        x_train = X_train.values[train]
        y_train = y_train.values[train]
        x_test = X_train.values[test]
        y_test = y_train[test]
        print('Positive in y_test: ', y_test.sum()/y_test.shape[0])
        print('Positive in y_train: ',y_train.sum()/y_train.shape[0])
 
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred_1_proba = model.predict_proba(x_test)[:, 1]
        y_pred_proba = model.predict_proba(x_test)
        tn, fp, fn, tp  = confusion_matrix(y_test, y_pred_proba.argmax(axis=1)).ravel()
        f2 = fbeta_score(y_test, y_pred, average='binary', beta=2)
        f0_5 = fbeta_score(y_test, y_pred, average='binary', beta=0.5)
        f1 = f1_score(y_test, y_pred, average='binary')
        precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        ROC_AUC = roc_auc_score(y_test, y_pred_1_proba, average='weighted')
        (precisions, recalls, _) = precision_recall_curve(y_test, y_pred_1_proba)
        aucpr = auc(recalls, precisions)
        AP = average_precision_score(y_test, y_pred)
        
        brier = brier_score_loss(y_test, y_pred_1_proba)
        acc = accuracy_score(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])
        mid_results = np.row_stack((mid_results,[precision, recall, f1, ROC_AUC, aucpr, brier, acc]))
    compare_result = np.row_stack((compare_result, [model_name,
                                                    mid_results[1:, 0].astype(float).mean().round(3),
                                                    mid_results[1:, 0].astype(float).std().round(3),
                                                    mid_results[1:, 1].astype(float).mean().round(3),
                                                    mid_results[1:, 1].astype(float).std().round(3),
                                                    mid_results[1:, 2].astype(float).mean().round(3),
                                                    mid_results[1:, 2].astype(float).std().round(3),
                                                    mid_results[1:, 3].astype(float).mean().round(3),
                                                    mid_results[1:, 3].astype(float).std().round(3),
                                                    mid_results[1:, 4].astype(float).mean().round(3),
                                                    mid_results[1:, 4].astype(float).std().round(3),
                                                    mid_results[1:, 5].astype(float).mean().round(3),
                                                    mid_results[1:, 5].astype(float).std().round(3),
                                                    mid_results[1:, 6].astype(float).mean().round(3),
                                                    mid_results[1:, 6].astype(float).std().round(3),
                                                    ]))
import datetime
filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_CV_results_Minmax_'+str(minmaxnorm)+'_SMOTE_'+str(SMOTE_Process)+'.csv'
np.savetxt(filename, compare_result, delimiter=',', fmt='%s')