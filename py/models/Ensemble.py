import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

def Ensembles(clf,train_stack,test_stack,train,target,folds=5):
    ## predict data box
    validation_pred = np.zeros(train_stack.shape[0])
    test_pred = np.zeros(test_stack.shape[0])
    ## k-stratified k-Fold
    folds = Validation(folds)
    # 外れ値を考慮して, データを分割する
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack,train['target_class'].values)):
    # for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack,train['outliers'].values)):
        print("fold n°{}".format(fold_+1))
        trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
        val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values
        # fitting arvitrary model for train data
        clf.fit(trn_data, trn_y);
        # predicting validation data and predicting data
        validation_pred[val_idx] = clf.predict(val_data)
        test_pred += clf.predict(test_stack) / folds.n_splits
    return validation_pred, test_pred
