import numpy as np
import pandas as pd
import sys
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

def reg_model():
    model = Sequential()
    model.add(Dense(50, input_dim=30, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def main():

    sys.stdout = open('./resources/log/nnlearn.txt', 'w')

    X = np.array(pd.read_csv('./resources/preprocessed_data/data.csv').values)[:,1:]
    Y = np.array(pd.read_csv('./resources/preprocessed_data/target.csv').values)[:,1:].reshape(-1,)

    estimator = KerasRegressor(build_fn=reg_model, epochs=20, batch_size=10, verbose=0)
    kfold = KFold(n_splits=5, random_state=0)

    results = cross_val_score(estimator, X, Y, scoring='neg_mean_squared_error', cv=kfold, n_jobs=-1)
    mse = -results.mean()
    print("KERAS REG RMSE : %.2f" % (mse ** 0.5))


if __name__ == '__main__':
    main()
