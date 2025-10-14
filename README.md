# 🧬 SAMPM: Successful Aging Prediction Models based on Machine Learning
## 📖 Introduction
This repository provides the implementation of machine learning models for predicting successful aging (SA) outcomes, based on health conditions, lifestyle factors, and chronic disease history.
The project evaluates multiple algorithms, compares predictive performance, and applies interpretability techniques to understand feature contributions.

---

## 🚀 Implemented Models
The following models are implemented as baselines for predictive analysis and hyperparameter optimization:

- **CatBoost** (Categorical Boosting)  
- **Random Forest (RF)**  
- **Logistic Regression (LR)**  
- **Gradient Boosting Decision Tree (GBDT)**  
- **XGBoost**  
- **LightGBM**  

---

## 📊 Evaluation Metrics
Model performance is assessed using multiple metrics for robustness:

- **F1 Score**  
- **AUC (Area Under ROC Curve)**  
- **Precision & Recall**  
- **AUPRC (Area Under Precision–Recall Curve)**  
- **Brier Score**  
- **Accuracy**  

In addition, **ROC curves** are plotted for both training and test sets to provide detailed comparison.

---

## 🔍 Model Interpretability
To gain insights into model decisions, the following interpretability techniques are applied:

- **SHAP (SHapley Additive exPlanations):** quantifies individual feature contributions.  
- **ALE Plots (Accumulated Local Effects):** captures feature effects while accounting for correlations.  
- **Restricted Cubic Splines (RCS) with logistic regression:** explores nonlinear associations, e.g., between SA and sleep duration.  

---

## 📜 Citation
If you use this repository in your research, please cite:
```{}
10.1038/s41598-025-24154-w
```

---

## 📬 Contact
For questions, please feel free to contact nanh302311@gmail.com.

