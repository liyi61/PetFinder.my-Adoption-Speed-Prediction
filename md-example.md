---
layout: page
title: Modeling
subtitle: XGBoost, LightGBM, CatBoost, and Logistic Regression
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

In order for our models to have better performances, I decided to combine features in cat_cols which contains categorical features. Then I appended the combined features to the original list to obtain more features. Excessive number of features can cause model overfitting, however; in the original data, there were only 20-ish features so that adding more features won’t cause any trouble in this case. 


## XGBoost

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


