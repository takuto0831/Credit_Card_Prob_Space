'''
- 各種機械学習手法により class分けする.
0: モデルのメイン分 Model code (下記の共通部分)
1: 訓練データを用いた validation code 
2: テストデータを用いた prediction code
3: 訓練データを用いた parameter tuning code (optuna)

class 
- LightGBM 

'''

import numpy as np # linear algebra
import pandas as pd # data processing
import lightgbm as lgb
from xgboost import XGBClassifier
from functools import partial
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier, AdaBoostClassifier
import pydotplus 
from IPython.display import Image
from sklearn.externals.six import StringIO

class LightGBM:
    def __init__(self):
        # validation
        self.fold = KFold(n_splits=5,random_state=831,shuffle=True)
        # base hyper parameter
        self.base_param = {
            'objective': 'binary',
            'metric': 'binary_error'
            }
            
    def Model(self,train,trn_index,val_index,features,category_features,param):
        # data set
        trn_data = lgb.Dataset(train.iloc[trn_index][features], label=train.iloc[trn_index]['y'])
        val_data = lgb.Dataset(train.iloc[val_index][features], label=train.iloc[val_index]['y'])
        # model
        model = lgb.train(param, 
                          trn_data, 
                          categorical_feature = category_features,
                          num_boost_round= 20000, 
                          valid_sets = [trn_data, val_data],
                          verbose_eval= 100, 
                          early_stopping_rounds= 200)
        return model
    def validation(self,train,features,category_features,param):
        score = []
        feature_importance = pd.DataFrame() # importance data frame
        for i,(trn_index, val_index) in enumerate(self.fold.split(train)):
            print("fold n°{}".format(i+1))
            # model execute
            model = self.Model(train,trn_index,val_index,features,category_features,param)
            # model importance 
            fold_importance = pd.DataFrame({'feature': features, 
                                            'importance': model.feature_importance(),
                                            'fold': i + 1})
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
            # validation predict
            pred = model.predict(train.iloc[val_index][features], num_iteration=model.best_iteration)
            pred = np.round(pred).astype(int)
            # score
            score.append(f1_score(y_true = train.iloc[val_index]['y'], y_pred = pred))
        # transform dataframe
        ans = pd.DataFrame({"model":"Lightgbm Classifier","fold":range(1,i+2),"score":score} )
        return ans, feature_importance
    def prediction(self,train,test,features,category_features,param):
        # 不足しているパラメータを補完
        param.update(self.base_param)
        # 予測値を格納する
        pred = np.zeros(test.shape[0])
        for i,(trn_index, val_index) in enumerate(self.fold.split(train)):
            print("fold n°{}".format(i+1))
            # model execute
            model = Model(train,trn_index,val_index,features,category_features,param)
            # validation predict
            tmp = model.predict(test[features], num_iteration = model.best_iteration)
            pred += tmp
        # transpose binary value
        pred = np.round(pred/(i+1)).astype(int)
        return pred
    def tuning(self,train,features,category_features,trial):
        # score
        score = []
        # tuning parameters
        param = {
            'objective': 'binary',
            'metric': 'binary_error',
            'verbosity': -1, # 計算結果の表示有無
            'max_depth' : -1, # 決定木の深さ, デフォルト値
            'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
            'num_leaves': trial.suggest_int('num_leaves',5, 1000), # 決定木の複雑度
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0),
            # 'min_data_in_leaf: # 決定木ノードの最小データ数
        }
        # boosting type parameters  
        if param['boosting_type'] == 'dart':
            param['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
            param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
        if param['boosting_type'] == 'goss':
            param['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
            param['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - param['top_rate'])
        
        for i,(trn_index, val_index) in enumerate(self.fold.split(train)):
            print("fold n°{}".format(i+1))
            # model execute
            model = Model(train,trn_index,val_index,features,category_features,param)
            # validation predict
            pred = model.predict(train.iloc[val_index][features],num_iteration = model.best_iteration)
            pred = np.round(pred).astype(int)
            # score
            score.append(f1_score(y_true = train.iloc[val_index]['y'], y_pred = pred))
        return 1 - np.mean(score)
class DecisionTree:
    def __init__(self):
        # validation
        self.fold = KFold(n_splits=5,random_state=831,shuffle=True)
        # base parameters
        self.base_param = {
            'random_state': 831
            }
    def tuning(train,features,trial):
        # score
        score = []
        # tuning parameters
        params = {
            'random_state': 831,
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
        for i,(trn_index, val_index) in enumerate(self.fold.split(train)):
            print("fold n°{}".format(i+1))
            # data set
            trn_data = lgb.Dataset(train.iloc[trn_index][features], label=train.iloc[trn_index]['y'])
            val_data = lgb.Dataset(train.iloc[val_index][features], label=train.iloc[val_index]['y'])
            # model
            model = lgb.train(param,
                              trn_data, 
                              categorical_feature = category_features, # カテゴリ変数として処理
                              valid_sets = [trn_data, val_data],
                              num_boost_round=20000, 
                              verbose_eval=100, 
                              early_stopping_rounds=200)
            # validation predict
            pred = model.predict(train.iloc[val_index][features],num_iteration = model.best_iteration)
            pred = np.round(pred).astype(int)
            # score
            score.append(f1_score(y_true = train.iloc[val_index]['y'], y_pred = pred))
        return 1 - np.mean(score)
