import time

from kerastuner.tuners import RandomSearch, BayesianOptimization, Hyperband
from tensorflow.keras.callbacks import EarlyStopping as es
from tensorflow.keras.callbacks import TerminateOnNaN as tnan

from .hypermodels import MultiLayerPerceptron as MLP


def generate_tuners(data_name, hypermodel, X, y, budget):
    rst = RandomSearch(
        hypermodel(
            X_shape=X.shape,
            y_shape=y.shape
        ),
        objective='val_accuracy',
        max_trials=budget,
        executions_per_trial=1,
        directory='o',
        project_name='rs_{}'.format(data_name)
    )

    gpt = BayesianOptimization(
        hypermodel(
            X_shape=X.shape,
            y_shape=y.shape
        ),
        objective='val_accuracy',
        max_trials=budget,
        executions_per_trial=1,
        directory='o',
        project_name='gp_{}'.format(data_name)
    )

    hbt = Hyperband(
        hypermodel(
            X_shape=X.shape,
            y_shape=y.shape
        ),
        objective='val_accuracy',
        max_epochs=budget,
        executions_per_trial=1,
        directory='o',
        project_name='hb_{}'.format(data_name)
    )

    tuners = {
        'rs': {
            'tuner': rst,
            'time': None,
            'best': None
        },
        'gp': {
            'tuner': gpt,
            'time': None,
            'best': None
        },
        'hb': {
            'tuner': hbt,
            'time': None,
            'best': None
        }
    }
    return tuners


def tuning(X_tr, y_tr, data_name, budget, min_delta=0.001):
    '''
    Tuning a model with varius tuners
    '''
    tuners = generate_tuners(
        hypermodel=MLP,
        X=X_tr,
        y=y_tr,
        data_name=data_name,
        budget=budget
    )

    for tuner_name, tuner_dict in tuners.items():

        print('Start tuning for {} with {}'.format(data_name, tuner_name))
        stopper = es(
            patience=5,
            min_delta=min_delta,
            restore_best_weights=True
        )
        terminator = tnan()

        tuner = tuner_dict['tuner']
        begin = time.time()
        tuner.search(
            X_tr,
            y_tr,
            epochs=100,
            verbose=0,
            validation_split=0.2,
            batch_size=256,
            callbacks=[stopper, terminator]
        )
        print('End tuning for {} with {}'.format(data_name, tuner_name))
        end = time.time()
        print(end - begin)
        tuner_dict['time'] = end - begin
        best_model = tuner.get_best_models()[0]
        best_model.save(f'results\\best_models\\{tuner_name}_{data_name}')
        tuner_dict['best'] = best_model

    return tuners
