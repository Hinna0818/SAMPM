## GBDT sensitivity analysis

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, brier_score_loss, accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import copy
import os

output_dir = "Sensitivity_Analysis"
os.makedirs(output_dir, exist_ok=True)

def calculate_metrics_for_splits(dataset, test_sizes):
    auc_scores, f1_scores, brier_scores, acc_scores = [], [], [], []

    X = dataset[['hypertension', 'dyslipidemia', 'liver_disease', 'kidney_disease',
                 'arthritis', 'digestive_disease', 'sleep_duration', 'smoking',
                 'drinking', 'BMI_status']]
    y = dataset['successful_aging']

    # 数据预处理
    categorical_columns = ['hypertension', 'dyslipidemia', 'liver_disease', 'kidney_disease',
                           'arthritis', 'digestive_disease', 'smoking', 'drinking', 'BMI_status']
    numerical_columns = ['sleep_duration']
    dataset[numerical_columns] = MinMaxScaler().fit_transform(dataset[numerical_columns])
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True, dtype=float)

    for ratio in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=ratio, random_state=23333)

        model = GradientBoostingClassifier(n_estimators=10,
        max_depth=29,
        min_samples_split=7,
        min_samples_leaf=3,
        min_weight_fraction_leaf=0.06618028848966474
    )
        model.fit(X_train, y_train)

        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc_scores.append(roc_auc_score(y_test, y_pred_prob))
        f1_scores.append(f1_score(y_test, y_pred))
        brier_scores.append(brier_score_loss(y_test, y_pred_prob))
        acc_scores.append(accuracy_score(y_test, y_pred))

    return auc_scores, f1_scores, brier_scores, acc_scores

# 主运行
test_sizes = np.round(np.arange(0.1, 0.26, 0.015), 3)
dataset = pd.read_csv("../final_data.csv")
auc, f1, brier, acc = calculate_metrics_for_splits(dataset, test_sizes)

# 画图
plt.figure(figsize=(10, 6))
plt.plot(test_sizes, auc, marker='o', label='AUC')
plt.plot(test_sizes, f1, marker='s', label='F1-score')
plt.plot(test_sizes, brier, marker='^', label='Brier score')
plt.plot(test_sizes, acc, marker='d', label='Accuracy')

plt.xlabel('Test Set Proportion')
plt.ylabel('Metric Value')
plt.title('GBDT Sensitivity Analysis')
plt.grid(True)
plt.yticks(np.arange(0.2, 1.05, 0.1))
plt.legend(loc='best')
plt.tight_layout()

# 保存图像
path = os.path.join(output_dir, "GBDTSensitivity_AUROCC_F1_Brier_ACC.png")
plt.savefig(path, dpi=330, bbox_inches='tight')
plt.show()
# 可选打印

for i, ratio in enumerate(test_sizes):
    print(f"Test={ratio:.3f} | AUC={auc[i]:.3f}, F1={f1[i]:.3f}, Brier={brier[i]:.3f}, ACC={acc[i]:.3f}")
