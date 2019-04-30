from sklearn.metrics import cohen_kappa_score
import pandas as pd

def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

result_opt = pd.read_csv('./submission.csv')
result_lcv = pd.read_csv('./submission_lcv.csv')
result_cat = pd.read_csv('./submission_cat.csv')
result_lgb = pd.read_csv('./submission_lgb.csv')
result_xgb = pd.read_csv('./submission_xgb.csv')
Groundtruth = pd.read_csv('./Groundtruth.csv')

def score(RE, GT):
    y_pred_valid = RE['AdoptionSpeed']
    y_valid = GT['AdoptionSpeed']
    score = kappa(y_valid, y_pred_valid)
    return score

print('Fold kappa for optimal model:', score(result_opt, Groundtruth))
print('Fold kappa for LogisticRegressionCV model:', score(result_lcv, Groundtruth))
print('Fold kappa for Catboost model:', score(result_cat, Groundtruth))
print('Fold kappa for lightgbm model:', score(result_lgb, Groundtruth))
print('Fold kappa for xgboost model:', score(result_xgb, Groundtruth))

