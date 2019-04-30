import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import lightgbm as lgb
import xgboost as xgb
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import  LogisticRegressionCV
import gc
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")
pd.set_option('max_colwidth', 500)
pd.set_option('max_columns', 500)
pd.set_option('max_rows', 100)
import os
from sklearn.metrics import cohen_kappa_score
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

breeds = pd.read_csv('./breed_labels.csv')
colors = pd.read_csv('./color_labels.csv')
states = pd.read_csv('./state_labels.csv')

train = pd.read_csv('./train/train.csv')
test = pd.read_csv('./test/test.csv', engine='python')
sub = pd.read_csv('./test/sample_submission.csv')

train['dataset_type'] = 'train'
test['dataset_type'] = 'test'
all_data = pd.concat([train, test])
sentiment_dict = {}
for filename in os.listdir('./train_sentiment/'):
    with open('./train_sentiment/' + filename, 'rb') as f:
        sentiment = json.load(f)
    pet_id = filename.split('.')[0]
    sentiment_dict[pet_id] = {}
    sentiment_dict[pet_id]['magnitude'] = sentiment['documentSentiment']['magnitude']
    sentiment_dict[pet_id]['score'] = sentiment['documentSentiment']['score']
    sentiment_dict[pet_id]['language'] = sentiment['language']

for filename in os.listdir('./test_sentiment/'):
    with open('./test_sentiment/' + filename, 'rb') as f:
        sentiment = json.load(f)
    pet_id = filename.split('.')[0]
    sentiment_dict[pet_id] = {}
    sentiment_dict[pet_id]['magnitude'] = sentiment['documentSentiment']['magnitude']
    sentiment_dict[pet_id]['score'] = sentiment['documentSentiment']['score']
    sentiment_dict[pet_id]['language'] = sentiment['language']

train['lang'] = train['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
train['magnitude'] = train['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
train['score'] = train['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

test['lang'] = test['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
test['magnitude'] = test['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
test['score'] = test['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

all_data['lang'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
all_data['magnitude'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
all_data['score'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

####### Basic model
cols_to_use = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'RescuerID', 'health', 'Free', 'score',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'No_name', 'Pure_breed', 'desc_length', 'desc_words',
               'averate_word_length', 'magnitude']

train = train[[col for col in cols_to_use if col in train.columns]]
test = test[[col for col in cols_to_use if col in test.columns]]

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

# time
indexer = {}
for col in cat_cols:
    # print(col)
    _, indexer[col] = pd.factorize(train[col].astype(str))

for col in cat_cols:
    # print(col)
    train[col] = indexer[col].get_indexer(train[col].astype(str))
    test[col] = indexer[col].get_indexer(test[col].astype(str))

y = train['AdoptionSpeed']
train = train.drop(['AdoptionSpeed'], axis=1)

# model
n_fold = 5
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=15)


def train_model(X=train, X_test=test, y=y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False,
                averaging='usual', make_oof=False):
    result_dict = {}
    if make_oof:
        oof = np.zeros((len(X), 5))
    prediction = np.zeros((len(X_test), 5))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        gc.collect()
        print('Fold', fold_n + 1, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
            valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols)

            model = lgb.train(params,
                              train_data,
                              num_boost_round=20000,
                              valid_sets=[train_data, valid_data],
                              verbose_eval=500,
                              early_stopping_rounds=200)

            del train_data, valid_data

            y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
            del X_valid
            gc.collect()
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                              verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)

        if model_type == 'lcv':
            model = LogisticRegressionCV(scoring='neg_log_loss', cv=3, multi_class='multinomial')
            model.fit(X_train, y_train)

            y_pred_valid = model.predict_proba(X_valid)
            y_pred = model.predict_proba(X_test)

        if model_type == 'cat':
            model = CatBoostClassifier(iterations=500, loss_function='MultiClass')
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[],
                      use_best_model=False, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict_proba(X_test)

        if make_oof:
            oof[valid_index] = y_pred_valid
        scores.append(kappa(y_valid, y_pred_valid.argmax(1)))
        print('Fold kappa:', kappa(y_valid, y_pred_valid.argmax(1)))
        print('')

        if averaging == 'usual':
            prediction += y_pred
        elif averaging == 'rank':
            prediction += pd.Series(y_pred).rank().values

        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importance()
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    if model_type == 'lgb':

        if plot_feature_importance:
            feature_importance["importance"] /= n_fold
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')

            result_dict['feature_importance'] = feature_importance

    result_dict['prediction'] = prediction
    if make_oof:
        result_dict['oof'] = oof

    return result_dict
##############LogisticRegressionCV##################
print('----------training LogisticRegressionCV-------------')
result_dict_lcv = train_model(model_type='lcv')
prediction_lcv= (result_dict_lcv['prediction']).argmax(1)
submission_lcv = pd.DataFrame({'PetID': sub.PetID, 'AdoptionSpeed': [int(i) for i in prediction_lcv]})
submission_lcv.head()
submission_lcv.to_csv('submission_lcv.csv', index=False)
print('----------train LogisticRegressionCV end-------------')



##############Catboost##################
print('----------training Catboost-------------')
result_dict_cat = train_model(model_type='cat', make_oof=True)
prediction_cat = (result_dict_cat['prediction']).argmax(1)
submission_cat = pd.DataFrame({'PetID': sub.PetID, 'AdoptionSpeed': [int(i) for i in prediction_cat]})
submission_cat.head()
submission_cat.to_csv('submission_cat.csv', index=False)
print('----------train Catboost end-------------')


#########lightgbm###############
print('----------training lightgbm-------------')
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
result_dict_lgb = train_model(X=train, X_test=test, y=y, params=params, model_type='lgb', plot_feature_importance=False, make_oof=True)
prediction_lgb = (result_dict_lgb['prediction']).argmax(1)
submission_lgb = pd.DataFrame({'PetID': sub.PetID, 'AdoptionSpeed': [int(i) for i in prediction_lgb]})
submission_lgb.head()
submission_lgb.to_csv('submission_lgb.csv', index=False)
print('----------train lightgbm end-------------')

##############xgboost##################
print('----------training xgboost-------------')
xgb_params = {'eta': 0.01, 'max_depth': 9, 'subsample': 0.9, 'colsample_bytree': 0.9,
          'objective': 'multi:softprob', 'eval_metric': 'merror', 'silent': True, 'nthread': 4, 'num_class': 5}
result_dict_xgb = train_model(params=xgb_params, model_type='xgb', make_oof=True)
prediction_xgb= (result_dict_xgb['prediction']).argmax(1)
submission_xgb = pd.DataFrame({'PetID': sub.PetID, 'AdoptionSpeed': [int(i) for i in prediction_xgb]})
submission_xgb.head()
submission_xgb.to_csv('submission_xgb.csv', index=False)
print('----------train xgboost end-------------')

##############optimized model###############
prediction = (result_dict_lcv['prediction'] + result_dict_cat['prediction'] + result_dict_lgb['prediction'] + result_dict_xgb['prediction']).argmax(1)
submission = pd.DataFrame({'PetID': sub.PetID, 'AdoptionSpeed': [int(i) for i in prediction]})
submission.head()
submission.to_csv('submission.csv', index=False)
