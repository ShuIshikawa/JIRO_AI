
import numpy             as np
import pandas            as pd
from sklearn.externals.joblib import load

def main():

    X_test = np.array(pd.read_pickle('./resources/test/data.pkl'))
    Y_test = np.array(pd.read_pickle('./resources/test/target.pkl')).reshape(-1,)

    n_samples, n_features = X_test.shape

    clf_names = ['svm', 'rf', 'xgb']

    for clf_name in clf_names:
        # 予測器を読込
        clf = load('resources/estimators/' + clf_name + '.pkl')
        # 予測
        Y_pred = pd.DataFrame(np.c_[clf.predict(X_test), Y_test])
        print(clf.score(X_test, Y_test))

        # 予測結果を保存
        Y_pred.to_csv('./resources/prediction/prediction_' + clf_name + '.csv',index = False)

if __name__ == '__main__':
    main()
