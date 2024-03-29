# Credit_Card_Prob_Space

## Detail 

クレジットカード会社における情報を活用し, 支払い不履行になり得る顧客を予測する.
このコンペディションでは, default of credit card clientsと呼ばれるデータセットを使用する.

2005年4月-2005年9月までのクレジットカード顧客の支払い情報やユーザーのデモグラフィック情報を元に, おそらく次月もしくは未来のある時点での支払いの履行(0)or不履行(1)を予測する.

## reference

- [kaggle tutorial](https://www.kaggle.com/lucabasa/credit-card-default-a-very-pedagogical-notebook)
- [optuna](https://github.com/pfnet/optuna/blob/master/examples/lightgbm_simple.py)
- [遺伝的アルゴリズムによる特徴量生成](https://qiita.com/overlap/items/e7f1077ef8239f454602)

## Plan

とりあえずいろいろなことを, [exploratory_data_analysis](https://github.com/takuto0831/Credit_Card_Prob_Space/blob/master/jn/exploratory_data_analysis.ipynb)において実行し, 汎用的な書き方でスクリプト化する. 特徴量エンジニアリングは[feature_engineering](https://github.com/takuto0831/Credit_Card_Prob_Space/blob/master/jn/feature_engineering.ipynb)ページで試す

- 各種手法のクロスバリデーション
- 特徴量エンジニアリング
- 機械学習のための特徴量エンジニアリング(株式会社ホクソエム)を読む

# Directory

```
.
├── README.md
├── input
│   ├── About_data.xlsx
│   ├── original
│   │   ├── submit_file.csv
│   │   ├── test_data.csv
│   │   └── train_data.csv
│   └── parameters
│       ├── Tree_classifer_param.txt
│       └── lgb_classifer_param.txt
├── jn
│   ├── exploratory_data_analysis.ipynb
│   ├── feature_engineering.ipynb
│   └── make_parameters.ipynb
├── output
│   └── report
│       └── train_data_profile.html
└── py
    ├── Base.py
    ├── __init__.py
    ├── __pycache__
    │   └── base.cpython-36.pyc
    └── models
        ├── Classifier.py
        ├── Ensemble.py
        ├── FeatureEngineering.py
        ├── __init__.py
        └── __pycache__
            └── Classifier.cpython-36.pyc

```
