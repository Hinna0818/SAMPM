## train model after hyper-params optimization
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
# np.random.seed(3407)
# random.seed(3407)
# np.random.seed(2333)
# random.seed(2333)

def process_no_ordered(dataset):
    return dataset

minmaxnorm = False
SMOTE_Process = False
SHAP_Analysis = False #False True
ALE_Analysis = False
print('minmaxnorm: ', minmaxnorm)
print('SMOTE_Process: ', SMOTE_Process)
print('SHAP_Analysis: ', SHAP_Analysis)
print('ALE_Analysis: ', ALE_Analysis)

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

x_train, x_test, y_train, y_test = train_test_split(datasetX,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=23333,
                                                    shuffle = True)

print('Baseline task')
print('Positive in y_test: ', y_test.sum()/y_test.shape[0])
print('Positive in y_train: ',y_train.sum()/y_train.shape[0])

# if SMOTE_Process==True:
#     from imblearn.over_sampling import SMOTE
#     sm = SMOTE(random_state=42)
#     x_train, y_train = sm.fit_resample(x_train, y_train)

compare_results = ['model_name','dataset', 'TN','FP','FN','TP','Pre','Rec','F0.5','F1',
                   'F2','AP','ROC_AUC','AUCPR','Brier','ACC']



# ## train set
# plt.figure(figsize=(10, 10))  # 用于训练集ROC
# for i in Models.keys():
#     print(i)
    
#     model = Models[i]
#     model.fit(x_train, y_train)

#     # 绘制训练集ROC曲线
#     y_train_pred_proba = model.predict_proba(x_train)[:, 1]
#     train_fpr, train_tpr, _ = roc_curve(y_train, y_train_pred_proba)
#     train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
#     plt.plot(train_fpr, train_tpr, label='%s Train ROC (AUC = %0.3f)' % (i, train_roc_auc))

# # 添加训练集图例
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('1-Specificity(False Positive Rate)')
# plt.ylabel('Sensitivity(True Positive Rate)')
# plt.title('Train Set - Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.savefig('optimized_Train_ROC.png', dpi=330, bbox_inches='tight')
# plt.show()  


# ## test set
# plt.figure(figsize=(10, 10))
# for i in Models.keys():
#     print(i)
#     if i == 'Tab':
#         y_train = y_train.values
#         y_test = y_test.values
#         x_train = x_train.values
#         x_test = x_test.values

#     model = Models[i]
#     model.fit(x_train, y_train)

#     # test set
#     y_pred = model.predict(x_test)
#     y_pred_1_proba = model.predict_proba(x_test)[:, 1]
#     y_pred_proba = model.predict_proba(x_test)
#     tn, fp, fn, tp  = confusion_matrix(y_test, y_pred_proba.argmax(axis=1)).ravel()
#     f2 = fbeta_score(y_test, y_pred, average='binary', beta=2)
#     f0_5 = fbeta_score(y_test, y_pred, average='binary', beta=0.5)
#     f1 = f1_score(y_test, y_pred, average='binary')
#     precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
#     ROC_AUC = roc_auc_score(y_test, y_pred_1_proba, average='weighted')
#     (precisions, recalls, _) = precision_recall_curve(y_test, y_pred_1_proba)
#     aucpr = auc(recalls, precisions)
#     AP = average_precision_score(y_test, y_pred)
    
#     brier = brier_score_loss(y_test, y_pred_1_proba)
#     acc = accuracy_score(y_test, y_pred)
#     fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])
#     plt.plot(fpr, tpr, label='%s ROC (area = %0.3f)' % (i, ROC_AUC))
#     compare_results = np.row_stack((compare_results,[i, 'test', tn, fp, fn, tp, precision, recall, 
#                                    f0_5, f1, f2, AP, ROC_AUC, aucpr, brier, acc]))

# import datetime

# # 保存测试集结果
# filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_Compare_results_Minmax_'+str(minmaxnorm)+'_SMOTE_'+str(SMOTE_Process)+'.csv'
# np.savetxt(filename, compare_results, delimiter=',', fmt='%s')
# print(f"Results saved to {filename}")

# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('1-Specificity(False Positive Rate)')
# plt.ylabel('Sensitivity(True Positive Rate)')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.savefig('optimized_Test_ROC.png', dpi=330, bbox_inches='tight')
# plt.show()   # Display
# print(compare_results)

## LR coefficient
# nomogram
LR_model = Models['LR']
LR_model.fit(x_train, y_train)
nomo_list = ['name', 'coef', 'min',	'max']
for var in range(x_train.shape[1]):
    nomo_list = np.row_stack((nomo_list, [x_train.columns[var],
                                          LR_model.coef_[0][var],
                                          x_train[x_train.columns[var]].min(),
                                          x_train[x_train.columns[var]].max()]
                                                 ))
nomo_list = pd.DataFrame(nomo_list[1:,:], columns=nomo_list[0,:])
nomo_list['coef'] = pd.to_numeric(nomo_list['coef'])
nomo_list['abs_values'] = nomo_list['coef'].abs()
nomo_list_sorted = nomo_list.sort_values(by='abs_values', ascending=False)
intercept = LR_model.intercept_[0]
threshold = 0.5

# 构造与 nomo_list_sorted 列数一致的两行
intercept_row = pd.DataFrame([{
    "feature": "intercept",
    "coef": intercept,
    "min": 0,
    "max": 0,
    "abs_values": abs(intercept),
    "main_feature": "intercept",
    "sub_feature": "",
    "position": "",
    "type": "continuous"
}])

threshold_row = pd.DataFrame([{
    "feature": "threshold",
    "coef": threshold,
    "min": 0,
    "max": 0,
    "abs_values": abs(threshold),
    "main_feature": "threshold",
    "sub_feature": "",
    "position": "",
    "type": "continuous"
}])

# 拼接
nomo_list_sorted = pd.concat([intercept_row, threshold_row, nomo_list_sorted], ignore_index=True)


# 显示排序后的DataFrame
print(nomo_list_sorted)
print(LR_model.intercept_)
print(LR_model.coef_)

def extract_main_feature(feature_name):
    feature_name = str(feature_name)
    if feature_name.startswith(("BMI_status_", "drinking_")):
        return feature_name.split("_")[0]
    else:
        return feature_name

nomo_list_sorted["main_feature"] = nomo_list_sorted["feature"].astype(str).apply(extract_main_feature)
nomo_list_sorted["sub_feature"] = nomo_list_sorted["feature"]
nomo_list_sorted["type"] = "nominal"  # 你也可以根据变量实际情况手动改，例如 BMI 就是 continuous
nomo_list_sorted["position"] = "default"
nomo_list_sorted["sub_feature"] = nomo_list_sorted["name"]

import os

# 更稳妥的文件夹名，避免设备名冲突
folder_name = "output_results"
os.makedirs(folder_name, exist_ok=True)

# 保存路径
output_path = os.path.join(folder_name, "nomogram_input.xlsx")
nomo_list_sorted.to_excel(output_path, index=False)

print(f"✅ 文件已保存至：{output_path}")


# # # model_best = Models['XGB']
if SHAP_Analysis==True:
    # import scipy as sp
    # partition_tree = shap.utils.partition_tree(datasetX)
    # plt.figure(figsize=(15, 6))
    # sp.cluster.hierarchy.dendrogram(partition_tree, labels=datasetX.columns, 
    #                                 leaf_font_size=10)
    # plt.title("Hierarchical Clustering Dendrogram")
    # # plt.xlabel("feature")
    # # plt.ylabel("distance")
    # plt.savefig('SHAP_result/Hierarchical Clustering Dendrogram.png', dpi=330, bbox_inches='tight')
    # plt.show()


    model_best = Models['XGB']
    model_best.fit(x_train, y_train) ## 补充
    y_pred_train  = model_best.predict(x_train)
    y_pred_train_proba  = model_best.predict_proba(x_train)[:, 1]
    Analysis_data = copy.deepcopy(x_train)
    Analysis_data['GT'] = y_train
    Analysis_data['Pre'] = y_pred_train    
    Analysis_data['Proba'] = y_pred_train_proba   
    Analysis_True = Analysis_data[Analysis_data['Pre']==Analysis_data['GT']]
    # x_train = x_train.reset_index(drop=True)
    # # https://catboost.ai/docs/concepts/shap-values
    # y_pred = model_best.predict(x_train)
    # For GBDT
    # explainer = shap.TreeExplainer(model_best, x_train, model_output='probability')
    # shap_values = explainer.shap_values(x_train)
    # For CAT
    # explainer = shap.TreeExplainer(model_best)
    # shap_values = explainer(x_train)
    # For ensemble lib https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Census%20income%20classification%20with%20scikit-learn.html
    # For ensemble lib https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Simple%20California%20Demo.html
    # def f(x):
    #     return model_best.predict_proba(x)[:, 1]
    # med = x_train.median().values.reshape((1, x_train.shape[1]))
    # explainer = shap.Explainer(f, med)
    # shap_values = explainer(x_train)
    
    def f(x):
        return model_best.predict_proba(x)[:, 1]    
    explainer = shap.Explainer(f, x_train)
    shap_values = explainer(x_train)

    
    # TP: 5 TN: 3, 9
    # shap.plots.waterfall(shap_values[5], show=False)
    # plt.savefig('SHAP_result/Instance_TP_SHAP.png', dpi=330, bbox_inches='tight')
    # plt.show()
 
    # shap.plots.waterfall(shap_values[3], show=False)
    # plt.savefig('SHAP_result/Instance_TN_SHAP.png', dpi=330, bbox_inches='tight')
    # plt.show()
    
    # # https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/heatmap.html
    # # https://shap.readthedocs.io/en/latest/generated/shap.plots.heatmap.html
    # shap.plots.heatmap(shap_values[:500], show=False)
    # plt.savefig('SHAP_result/heatmap.png', dpi=330, bbox_inches='tight')
    # plt.show()
    
    ## shap画图
    # # 修改一下要展示的列名
    # selected_feature_names = ['sleep_duration', 'hypertension_1', 'arthritis_1', 'dyslipidemia_1', 'BMI_status_2', 'digestive_disease_1', 'kidney_disease_1',
    #                  'smoking_1', 'drinking_2', 'liver_disease_1']
    # shap_selected = shap_values[:,:,0]
    # shap.summary_plot(shap_selected, x_train, plot_type="bar", show=False,
    #                   feature_names=datasetX.columns, cmap='plasma')
    # plt.savefig('SHAP_result/Summary_bar_XGB.png', dpi=330, bbox_inches='tight')
    # plt.show()
    # plt.close ()
    # shap.summary_plot(shap_selected, x_train, show=False,
    #                   feature_names=datasetX.columns, cmap='plasma')
    # plt.savefig('SHAP_result/Summary_ori_XGB.png', dpi=330, bbox_inches='tight')
    # plt.show()
    

    ## ALE
    vals= np.abs(shap_values.values).mean(0)
    feature_importance = pd.DataFrame(list(zip(datasetX.columns,vals)),columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
    feature_importance.head()
    
    explainer = shap.TreeExplainer(model_best)
    shap_values = explainer(x_train)
    
    shap.summary_plot(shap_values, x_train, plot_type="bar", show=False,
                      feature_names=datasetX.columns)

if ALE_Analysis == True:
    model_ale = ALE(model_best.predict, feature_names=datasetX.columns, 
                    target_names=['Violence'])
    model_exp = model_ale.explain(x_train.values) 
    for i in list(feature_importance['col_name'][:7]):
        plot_ale(model_exp, features=[i], n_cols=2, fig_kw={'figwidth':6, 'figheight': 3})
        path = 'SHAP_result/ALE' + i + '.png'
        plt.savefig(path, dpi=330, bbox_inches='tight')
    