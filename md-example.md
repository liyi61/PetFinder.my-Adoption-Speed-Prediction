---
layout: page
title: Modeling
subtitle: XGBoost, LightGBM, and Logistic Regression
bigimg: /img/main2.jpg
---

## Feature Selection

There are some feautres which were added during EDA process: score, No_name, Pure_breed, desc_length, desc_words, magnitude. 

```python
cat_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'State', 'RescuerID', 'Fee', 'Age',
            'VideoAmt', 'PhotoAmt']

more_cols = []
for col1 in cat_cols:
    for col2 in cat_cols:
        if col1 != col2 and col1 not in ['RescuerID', 'State'] and col2 not in ['RescuerID', 'State']:
            train[col1 + '_' + col2] = train[col1].astype(str) + '_' + train[col2].astype(str)
            test[col1 + '_' + col2] = test[col1].astype(str) + '_' + test[col2].astype(str)
            more_cols.append(col1 + '_' + col2)
cat_cols = cat_cols + more_cols
```

In order for our models to have better performances, I decided to combine features in cat_cols which contains categorical features. Then I appended the combined features to the original list to obtain more features. 


## XGBoost

XGBoost does not support categorical features, categorical features had to be loaded as array and then one-hot encoding was performed. Early stopping was used to find the optimal number of boosting rounds.  

```python
model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                              verbose_eval=500, params=params)
```
```python
xgb_params = {'eta': 0.01, 'max_depth': 9, 'subsample': 0.9, 'colsample_bytree': 0.9,
          'objective': 'multi:softprob', 'eval_metric': 'merror', 'silent': True, 'nthread': 4, 'num_class': 5}
```

![XG1](/img/XG1.png)
![XG2](/img/XG2.png)

## LightGBM

Similar procedure was done for LightGBM except for one-hot encoding wasn’t needed. LightGBM can use categorical features as input directly. 

```python
model = lgb.train(params,
                              train_data,
                              num_boost_round=20000,
                              valid_sets=[train_data, valid_data],
                              verbose_eval=500,
                              early_stopping_rounds=200)

```
```python
params = {'num_leaves': 512,
        #  'min_data_in_leaf': 60,
         'objective': 'multiclass',
         'max_depth': -1,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 3,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "random_state": 42,
         "verbosity": -1,
         "num_class": 5}
```

![gbm](/img/GBM.png)

## Logistic Regression

```python
model = LogisticRegressionCV(scoring='neg_log_loss', cv=3, multi_class='multinomial')
```

![lcv](/img/LR.png)

## Testing Result

| Model | Training Kappa | Testing Kappa |
| :------: | :-----: |:-----: |
| XGBoost | 0.3553 | 0.3016794437089825 |
| LightGBM | 0.3530 | 0.3547277856966704 |
| LogisticRegression CV | 0.2283 | 0.2081662794270801 |

The results aren't very statisfying so I thought about using Back Propogation neural network algorithm. 

