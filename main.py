from modules.utils import load_data, comparison_pipeline

import matplotlib.pyplot as plt
import seaborn as sns

DATA = {
    'mn_v': {
        'type': 'clas',
        'data': load_data('mn_v', portion=0.999),
        'min_delta': 0.0001,
        'n_classes': 10
    },
    'mn_b': {
        'type': 'clas',
        'data': load_data('mn_b', portion=0.999),
        'min_delta': 0.0001,
        'n_classes': 10
    },
    'mn_r': {
        'type': 'clas',
        'data': load_data('mn_r', portion=0.999),
        'min_delta': 0.0001,
        'n_classes': 10
    },
    'mn_rb': {
        'type': 'clas',
        'data': load_data('mn_rb', portion=0.999),
        'min_delta': 0.0001,
        'n_classes': 10
    }

}

BUDGET = 50
CV = 100

df = comparison_pipeline(
    data_sources=DATA,
    budget=BUDGET,
    cv=CV
)

sns.set(style='whitegrid', font_scale=1.5)
fig, axs = plt.subplots(3, 2, figsize=(10, 15))
sns.boxplot(
    x='tuner',
    y='score',
    data=df,
    ax=axs[0][0]
)
sns.boxplot(
    x='tuner',
    y='score',
    hue='source',
    data=df,
    ax=axs[0][1]
)
sns.barplot(
    x='tuner',
    y='time',
    data=df,
    ax=axs[1][0]
)
sns.barplot(
    x='tuner',
    y='time',
    hue='source',
    data=df,
    ax=axs[1][1]
)
sns.barplot(
    x='tuner',
    y='n_conf',
    data=df,
    ax=axs[2][0]
)
sns.barplot(
    x='tuner',
    y='n_conf',
    hue='source',
    data=df,
    ax=axs[2][1]
)
plt.tight_layout()
plt.savefig('results\\boxplot_results.png')
plt.show()
