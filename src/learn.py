import sys, os
from pprint import pprint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from hyperopt import fmin, tpe
from scipy.stats import randint, uniform, expon
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.externals.joblib import Parallel, delayed, dump
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit, cross_val_score, train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def reg_model():
    model = Sequential()
    model.add(Dense(40, input_dim=15, activation='relu'))
    model.add(Dense(1))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def plot_variance(X, path):

    sns.set_style("whitegrid", {'grid.linestyle': ':'})
    # matplotlib のフォントを変更
    plt.rcParams['font.family'] = 'Times New Roman'
    f, ax = plt.subplots()
    variances = X.var(axis=0)
    n_features = X.shape[1]
    features = np.arange(n_features)
    ax.bar(features, variances, align='center', width=1.0)
    ax.set_xlim(0, n_features)
    ax.set_xlabel('Features')
    ax.set_ylabel('Unbiased variance')
    ax.set_yscale('log')
    f.set_tight_layout(True)
    # グラフを書込
    f.savefig(path)
    # グラフを消去
    plt.clf


def plot_feature_importance(clf, path):

    sns.set_style("whitegrid", {'grid.linestyle': ':'})
    # matplotlib のフォントを変更
    plt.rcParams['font.family'] = 'Times New Roman'
    f, ax = plt.subplots()
    feature_importances = clf.feature_importances_
    n_features = feature_importances.size
    features = np.arange(n_features)
    ax.bar(features, feature_importances, align='center', width=1.0)
    ax.set_xlim(0, n_features)
    ax.set_ylim(0, 0.02)
    ax.set_xlabel('Features')
    ax.set_ylabel('Feature importance')
    f.set_tight_layout(True)
    # グラフを書込
    f.savefig(path)
    # グラフを消去
    plt.clf


def main(
    X_path='./data/X.csv',
    Y_path='./data/Y.csv',
    variance_path='./pictures/variances.png',
    feature_importance_path='./pictures/feature_importances.png'
    ):

    n_splits = 4
    test_size = 0.2
    random_state = 0

    X = np.array(pd.read_csv(X_path))
    Y = np.array(pd.read_csv(Y_path)).reshape(-1,)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    n_samples, n_features = X_train.shape

    # 各特徴量の分散を描画
    plot_variance(X, variance_path)

    # # 学習したいもののみをリストに追加
    clf_names = ['svm', 'rf', 'xgb', 'krs']

    estimator = {
        'svm': SVR(),
        'rf': RandomForestRegressor(random_state=random_state, n_jobs=-1),
        'xgb': XGBRegressor(seed=random_state, nthread=-1),
        'krs': KerasRegressor(build_fn=reg_model, verbose=0)
    }

    param_distributions = {
        'svm': {
            'C': expon(scale=1),
            'gamma': expon(scale=1)
            },
        'rf': {'max_depth': randint(1, 100), 'max_features': randint(3, n_features), 'n_estimators': randint(10, 100)},
        'xgb': {
            'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': uniform(0.0, 10.0),
            'learning_rate': uniform(0.0, 1.0),
            'max_delta_step': randint(1, 5),
            'max_depth': randint(1, 20),
            'min_child_weight': randint(1, 100),
            'n_estimators': randint(1, 20),
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
            },
        'krs': {'epochs': randint(10, 100), 'batch_size': randint(1, n_samples/4)}
    }

    for clf_name in clf_names:
        # チューニング
        randomized_search = RandomizedSearchCV(
            estimator = estimator[clf_name],
            param_distributions = param_distributions[clf_name],
            scoring = 'neg_mean_squared_error',
            cv = ShuffleSplit(n_splits=n_splits, test_size=1.0/n_splits, random_state=random_state),
            random_state = random_state,
            n_iter = 300,
            verbose = 1,
            n_jobs = 1
        ).fit(X_train, Y_train)

        best_estimator = randomized_search.best_estimator_
        best_params = randomized_search.best_params_

        pprint(best_params)

        if clf_name == 'rf':
            # 各特徴量の重要度を描画
            plot_feature_importance(best_estimator, feature_importance_path)

        #予測器をバイナリデータとして保存
        if clf_name == 'krs':
            model = reg_model()
            model.fit(X_train, Y_train, verbose=0, **best_params)
            model.save('./estimators/' + clf_name + '.h5')
            Y_pred = model.predict(X_test, verbose=0)
        else:
            dump(best_estimator, './estimators/' + clf_name + '.pkl')
            Y_pred = best_estimator.predict(X_test)

        # 予測
        Y_pred_pd = pd.DataFrame(np.c_[Y_pred, Y_test])
        pprint(mean_squared_error(Y_test, Y_pred) ** 0.5)
        # 予測結果を保存
        Y_pred_pd.to_csv('./data/pred_' + clf_name + '.csv',index = False)

if __name__ == '__main__':

    dirs = pd.read_json('dirs.json', typ='series')
    if not os.path.isdir(dirs['log_dir']):
        os.makedirs(dirs['log_dir'])
    sys.stdout = open(dirs['log_dir'] + 'learn.txt', 'w')

    main()
