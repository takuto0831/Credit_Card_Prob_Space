'''
1: 遺伝的アルゴリズムによる特徴量生成


'''
import sys, os
# set path
sys.path.append(os.getcwd())

import numpy as np # linear algebra
import pandas as pd # data processing
import lightgbm as lgb
import feather
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier, AdaBoostClassifier
import Classifier, Base

# genetic algorithm
from deap import algorithms, base, creator, tools, gp
import operator, math, time
from tqdm import tqdm

# instance
Process = Base.Process()

class GeneticFetureMake: 
    def __init__(self):
        # home path
        self.home_path = os.path.expanduser("~") + '/Desktop/Credit_Comp'
        # validation setting
        self.fold = KFold(n_splits=4,random_state=831,shuffle=True)
    def Model(self,train,features,params):
        DecisionTree = Classifier.DecisionTree()
        return DecisionTree.validation(train,features,params) # decision tree algorithm
        
    def GeneticMake(self,train,test,features,params,iteration,feature_limit,gen_num=10):
        '''
        iteration: 反復回数, 特徴量作成の試行回数
        feature_limit: 許容する特徴量数の限界値
        gen_num: 進化する世代数
        '''
        # 分母が0の場合を考慮した, 除算関数
        def protectedDiv(left, right):
            eps = 1.0e-7
            tmp = np.zeros(len(left))
            tmp[np.abs(right) >= eps] = left[np.abs(right) >= eps] / right[np.abs(right) >= eps]
            tmp[np.abs(right) < eps] = 1.0
            return tmp
        
        # 初期値
        base_score = self.Model(train,features,params)
        print("validation mean score:", base_score['score'].mean())   
        # 適合度を最大化するような木構造を個体として定義
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        # setting
        prev_score = np.mean(base_score['score']) # base score
        exprs = [] # 生成した特徴量
        results = pd.DataFrame(columns=['n_features','best_score','val_score']) # 結果を格納する. (best_score == val_score ??)
        n_features = len(features) # 初期時点の特徴量数
        X_train = train[features] # 訓練データの特徴量
        X_test = test[features] # テストデータの特徴量
        y_train = train['y'] # 訓練データのターゲット変数
        # main
        for i in tqdm(range(iteration)):
            pset = gp.PrimitiveSet("MAIN", n_features)
            pset.addPrimitive(operator.add, 2)
            pset.addPrimitive(operator.sub, 2)
            pset.addPrimitive(operator.mul, 2)
            pset.addPrimitive(protectedDiv, 2)
            pset.addPrimitive(operator.neg, 1)
            pset.addPrimitive(np.cos, 1)
            pset.addPrimitive(np.sin, 1)
            pset.addPrimitive(np.tan, 1)
            # function
            def eval_genfeat(individual):
                func = toolbox.compile(expr=individual)
                # make new features
                features_train = [np.array(X_train)[:,j] for j in range(n_features)]
                new_feat_train = func(*features_train)
                # combine table and select features name
                train_tmp = pd.concat([X_train,pd.DataFrame(new_feat_train,columns=['tmp']),y_train],axis=1)
                features_tmp = train_tmp.drop("y",axis=1).columns.values
                tmp_score = self.Model(train_tmp,features_tmp,params)
                # print(np.mean(tmp_score['score']))
                return np.mean(tmp_score['score']),
            # 関数のデフォルト値の設定
            toolbox = base.Toolbox()
            toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr) 
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("compile", gp.compile, pset=pset)
        # 評価、選択、交叉、突然変異の設定
            # 選択はサイズ10のトーナメント方式、交叉は1点交叉、突然変異は深さ2のランダム構文木生成と定義
            toolbox.register("evaluate", eval_genfeat)
            toolbox.register("select", tools.selTournament, tournsize=10)
            toolbox.register("mate", gp.cxOnePoint)
            toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
            toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
            # 構文木の制約の設定
            # 交叉や突然変異で深さ5以上の木ができないようにする
            toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
            toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5)) 
        
            # 世代ごとの個体とベスト解を保持するクラスの生成
            pop = toolbox.population(n=300)
            hof = tools.HallOfFame(1)
        
            # 統計量の表示設定
            stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
            stats_size = tools.Statistics(len)
            mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
            mstats.register("avg", np.mean)
            mstats.register("std", np.std)
            mstats.register("min", np.min)
            mstats.register("max", np.max)
        
            # 進化の実行
            # 交叉確率50%、突然変異確率10%、?世代まで進化
            start_time = time.time()
            pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, gen_num, stats=mstats, halloffame=hof, verbose=True)
            end_time = time.time()
        
            # ベスト解とscoreの保持
            best_expr = hof[0]
            best_score = mstats.compile(pop)["fitness"]["max"]
        
            # 生成変数を学習、テストデータに追加し、ベストスコアを更新する
            if prev_score < best_score:
                # 生成変数の追加
                func = toolbox.compile(expr=best_expr)
                features_train = [np.array(X_train)[:,j] for j in range(n_features)]
                features_test = [np.array(X_test)[:,j] for j in range(n_features)]
                new_feat_train = func(*features_train)
                new_feat_test = func(*features_test)
                # データ更新
                X_train = pd.concat([X_train,pd.DataFrame(new_feat_train,columns=['NEW'+ str(i)])],axis=1)
                X_test = pd.concat([X_test,pd.DataFrame(new_feat_test,columns=['NEW'+ str(i)])],axis=1)
                new_features = X_train.columns.values
                # テストスコアの計算（プロット用）
                val_score = self.Model(pd.concat([X_train,y_train],axis=1),new_features,params)
                # test_pred = DecisionTree.prediction(train,test,features,params)
        
                # ベストスコアの更新と特徴量数の加算
                prev_score = best_score
                n_features += 1
                # 表示と出力用データの保持
                print("n_features: %i, best_score: %f, time: %f second"
                      % (n_features, best_score, end_time - start_time))
                # 結果の格納 ( スコアの記録と作成した特徴量)
                tmp = pd.Series( [n_features, best_score, np.mean(val_score['score'])], index=results.columns )
                results = results.append( tmp, ignore_index=True )
                exprs.append(best_expr)
                # save 
                Process.write_feather(pd.concat([X_train,y_train],axis=1),file_name = 'train_gen')
                Process.write_feather(X_test,file_name = 'test_gen')
                # 変数追加後の特徴量数が??を超えた場合break
                if n_features >= feature_limit:
                    break
        return pd.concat([X_train,y_train],axis=1), X_test, results, exprs
