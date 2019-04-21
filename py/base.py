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
    # display boosting importance (selection save or not)    
    def display_importances(self,importance_df,title,file_name = None):
        cols = (importance_df[["feature", "importance"]]
                .groupby("feature")
                .mean()
                .sort_values(by="importance", ascending=False)[:500].index)
        best_features = importance_df.loc[importance_df.feature.isin(cols)]
        plt.figure(figsize=(14,80))
        sns.barplot(x="importance",y="feature",
                    data=best_features.sort_values(by="importance",ascending=False))
        plt.title(title + 'Features (avg over folds)')
        plt.tight_layout()
        # save or not
        if file_name is not None: 
            plt.savefig(self.home_path + '/output/image/' + file_name)
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
    def under_sampling(self,num,rate,train,features):
        # 値の比率を確認
        print('train data rate:', Counter(train['y']))
        # 前処理
        data = train.query("y == 0")[features].copy()
        data = data.replace([np.inf, -np.inf], np.nan) # inf 処理
        data.fillna((data.mean()), inplace=True) # nan 処理
        # kmeans クラスタリング
        kmeans = KMeans(n_clusters = num, random_state=831, n_jobs = -2).fit(data)
        # 群別の構成比を少数派の件数に乗じて群別の抽出件数を計算
        train['cluster'] = np.nan
        train.loc[train['target_class'] == 0,'cluster'] = kmeans.labels_
        count_sum = train.groupby('cluster').count().iloc[0:,0].as_matrix()
        ratio = ( (1-rate) * train["target_class"].sum() ) / ( count_sum.sum()*rate)
        samp_num = np.round(count_sum * ratio,0).astype(np.int32)
        # 群別にサンプリング処理を実施
        tmp = pd.DataFrame(index=[], columns=data.columns)
        for i in np.arange(num) :
            tmp_ = train[train['cluster']==i].sample(samp_num[i],replace=True)
            tmp = pd.concat([tmp,tmp_])
        # 外れ値データを結合
        tmp = pd.concat([tmp,train.query("target_class == 1")])
        del tmp['cluster']# クラスター列削除
        print("{} observations and {} features in train set.".format(tmp.shape[0],tmp.shape[1]))
        return tmp
        
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
