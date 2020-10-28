from modules.utils import load_data, comparison_pipeline

DATA = {
    'mn_v': {
        'type': 'clas',
        'data': load_data('mn_v', portion=0.33),
        'min_delta': 0.0001,
        'n_classes': 10
    },
    'mn_b': {
        'type': 'clas',
        'data': load_data('mn_b', portion=0.33),
        'min_delta': 0.0001,
        'n_classes': 10
    },
    'mn_r': {
        'type': 'clas',
        'data': load_data('mn_r', portion=0.33),
        'min_delta': 0.0001,
        'n_classes': 10
    },
    'mn_rb': {
        'type': 'clas',
        'data': load_data('mn_rb', portion=0.33),
        'min_delta': 0.0001,
        'n_classes': 10
    }

}

BUDGET = 33
CV = 33

df = comparison_pipeline(
    data_sources=DATA,
    budget=BUDGET,
    cv=CV
)
