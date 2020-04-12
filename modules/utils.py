import os

import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit as sss

from tensorflow.keras.callbacks import EarlyStopping as es
from tensorflow.keras.utils import to_categorical

from .tuners import tuning


def load_data(name, portion=.999):
    X = np.load('data\\{}\\X.npy'.format(name))
    y = np.load('data\\{}\\y.npy'.format(name))
    for sel, dis in sss(1, train_size=portion).split(X, y):

        X = X[sel]
        y = y[sel]
        
    data = (X, y)
    return data


def train_test_split(data):
    X, y = data
    y = to_categorical(y)
    for tr_i, ts_i in sss(n_splits=1, train_size=0.1).split(X, y):

        X_tr, y_tr = X[tr_i], y[tr_i]
        X_tr = mms().fit_transform(X_tr)
        X_ts, y_ts = X[ts_i], y[ts_i]

    return X_tr, X_ts, y_tr, y_ts


def comparison_pipeline(data_sources, budget, cv):
    index = 0
    df = pd.DataFrame(
        columns=['tuner', 'source', 'score', 'n_conf', 'time']
    )
    for data_name, source in data_sources.items():

        data = source['data']
        min_delta = source['min_delta']

        X_tr, X_ts, y_tr, y_ts = train_test_split(
            data=data
        )

        tuners = tuning(
            X_tr=X_tr,
            y_tr=y_tr,
            data_name=data_name,
            min_delta=min_delta,
            cv=cv,
            budget=budget,
        )

        for tuner_name, tuner_dict in tuners.items():

            model = tuner_dict['best']
            time = tuner_dict['time']
            path = 'o\\{}_{}'.format(tuner_name, data_name)
            n_conf = len(os.listdir(path)) - 2
            splitter = sss(n_splits=10)
            print('Start testing for {} with {}'.format(data_name, tuner_name))
            for tr_i, ts_i in splitter.split(X_ts, y_ts):

                X_ts_tr, y_ts_tr = X_ts[tr_i], y_ts[tr_i]
                scaler = mms().fit(X_ts_tr)
                X_ts_tr = scaler.transform(X_ts_tr)
                X_ts_ts, y_ts_ts = X_ts[ts_i], y_ts[ts_i]
                X_ts_ts = scaler.transform(X_ts_ts)

                stopper = es(
                    patience=5,
                    min_delta=0.001,
                    restore_best_weights=True
                )
                model.fit(
                    X_ts_tr,
                    y_ts_tr,
                    verbose=0,
                    epochs=100,
                    batch_size=256,
                    validation_split=0.2,
                    callbacks=[stopper]
                )

                predictions = model.predict(X_ts_ts).argmax(axis=1)
                score = f1_score(
                    y_ts_ts.argmax(axis=1),
                    predictions,
                    average='macro'
                )

                df.loc[index] = [
                    tuner_name,
                    data_name,
                    score,
                    n_conf,
                    time
                ]
                index += 1

        print('End testing for {} with {}'.format(data_name, tuner_name))

    df.to_csv('results\\results.csv', index=False)
    return df