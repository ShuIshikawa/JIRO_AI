
import numpy as np
import pandas as pd
import sys
from sklearn.externals.joblib import load
from sklearn.metrics import mean_squared_error

def main():
    sys.stdout = open('./resources/log/predict.txt', 'w')

    X_test = pd.read_pickle('./resources/test/data.pkl')
    Y_test = pd.read_pickle('./resources/test/target.pkl')

    n_samples, n_features = X_test.shape

    clf_names = ['svm', 'rf', 'xgb']

    for clf_name in clf_names:
        print(clf_name)
        # 予測器を読込
        clf = load('resources/estimators/' + clf_name + '.pkl')
        # 予測
        Y_pred = clf.predict(X_test)
        Y_pred_pd = pd.DataFrame(np.c_[Y_pred, Y_test])
        print(mean_squared_error(Y_test, Y_pred))

        # 予測結果を保存
        Y_pred_pd.to_csv('./resources/prediction/prediction_' + clf_name + '.csv',index = False)

if __name__ == '__main__':
    main()
