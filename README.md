# Credit_Card_Prob_Space

## Detail 

クレジットカード会社における情報を活用し, 支払い不履行になり得る顧客を予測する.
このコンペディションでは, default of credit card clientsと呼ばれるデータセットを使用する.

2005年4月-2005年9月までのクレジットカード顧客の支払い情報やユーザーのデモグラフィック情報を元に, おそらく次月もしくは未来のある時点での支払いの履行(0)or不履行(1)を予測する.

## 参考url

- [kaggle tutorial](https://www.kaggle.com/lucabasa/credit-card-default-a-very-pedagogical-notebook)

各種手法においてクロスバリデーションを適用し, 暫定的に使用するモデルを決定する. 特徴量エンジニアリング+パラメータチューン(optuna?)+遺伝的アルゴリズムによる特徴量探索.
