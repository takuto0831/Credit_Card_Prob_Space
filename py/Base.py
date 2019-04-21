import numpy as np # linear algebra
import pandas as pd # data processing
import feather # fast reading data
from datetime import datetime
import pickle,requests
import matplotlib.pyplot as plt
import seaborn as sns
import time, os, sys
from contextlib import contextmanager
from sklearn.cluster import KMeans
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

class Process:
    def __init__(self):
        # home path
        self.home_path = os.path.expanduser("~") + '/Desktop/Credit_Comp'
    # read original data
    def read_data1(self):
        train = pd.read_csv(self.home_path + '/input/original/train_data.csv').drop('id',axis=1)
        test = pd.read_csv(self.home_path + '/input/original/test_data.csv').drop('ID',axis=1)
        target = train['y'] # 目的変数を抽出
        print("{} observations and {} features in train set.".format(train.shape[0],train.shape[1]))
        print("{} observations and {} features in test set.".format(test.shape[0],test.shape[1]))
        return train,test,target
    # read processed data and features name list
    def read_data2(self,features_name = None):
        # Loading Train and Test Data
        train = pd.read_csv(self.home_path + '/input/aggregated/train.csv')
        test = pd.read_csv(self.home_path + '/input/aggregated/test.csv')
        features = []
        # features list
        if features_name is not None: 
          features = feather.read_dataframe(self.home_path + 'input/features/' + features_name + ".feather")
        # check data frame
        print("{} observations and {} features in train set.".format(train.shape[0],train.shape[1]))
        print("{} observations and {} features in test set.".format(test.shape[0],test.shape[1]))
        print("{} observations and {} features in features set.".format(features.shape[0],features.shape[1]))
        # extract target
        target = train['y'] # 目的変数を抽出
        features = features["feature"].tolist() # features list
        return train, test, target, features
    # make submit file
    def submit(self,predict,tech):
        # make submit file
        submit_file = pd.read_csv(self.home_path + '/input/original/submit_file.csv')
        submit_file["Y"] = predict
        # save for output/(technic name + datetime + .csv)
        file_name = self.home_path + '/output/submit/' + tech + '_' + datetime.now().strftime("%Y%m%d") + ".csv"
        submit_file.to_csv(file_name, index=False)
    # open parameters file
    def open_parameter(self,file_name):
        f = open(self.home_path + '/input/parameters/' + file_name + '.txt', 'rb')
        list_ = pickle.load(f)
        return list_
    # make best feature list (selection save or not)
    def extract_best_features(self,importance_df,num,file_name = None):
        cols = (importance_df[["feature", "importance"]]  
                .groupby("feature")
                .mean()
                .sort_values(by="importance", ascending=False)
                .reset_index())
        # save or not
        if file_name is not None: 
            cols.to_csv(self.home_path + '/input/features/' + file_name + '.csv')
        return cols[:num]["feature"].tolist()
class Applicate:
    # 欠損値の確認
    def missing_value(self,df):
        pd.set_option('display.max_columns', df.shape[1])
        total = df.isnull().sum().sort_values(ascending =False)
        percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
        return pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()
    #### 要変更 ####
    def down_sampling(self,train,features,rate=1):
        # 正例(y=1)の数を保存
        positive_count = train['y'].sum()
        # 任意の比率を元に, down sampling
        sampler = RandomUnderSampler(ratio={0:positive_count*rate, 1:positive_count}, random_state=831)
        X_resampled, y_resampled = sampler.fit_resample(train[features], train['y'])
        # 結果の確認
        print('original data rate:', Counter(train['y']))
        print('down sampling data rate:', Counter(y_resampled))
        # numpy -> pandas
        data = pd.DataFrame(X_resampled, columns=features)
        data['y'] = y_resampled
        return data
# other
def line(text):
    line_notify_token = '07tI1nvYaAtGaLdsCaxKZxkboOU0OsvLregXqodN2ZV' #発行したコード
    line_notify_api = 'https://notify-api.line.me/api/notify'
    # 変数messageに文字列をいれて送信します トークン名の隣に文字が来てしまうので最初に改行
    payload = {'message': '\n' + text}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    line_notify = requests.post(line_notify_api, data=payload, headers=headers)
@contextmanager
def timer(title):
    start = time.time()
    yield
    end = time.time()
    line("{} - done in {:.0f}s".format(title, end-start))
    
  
