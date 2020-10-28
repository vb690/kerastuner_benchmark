import numpy as np

import pymc3 as pm

import matplotlib.pyplot as plt


class GLMMPerformance:
    """
    """
    def __init__(self, df, tuners_column, contexts_column,
                 targets_column):
        """
        """
        self.df = df
        self.tuners_column = tuners_column
        self.contexts_column = contexts_column
        self.targets_column = targets_column

    def __define_data(self, target):
        """
        """
        data = self.df[self.df[self.targets_column] == target]

        unique_context = data[self.contexts_column].unique()
        unique_tuners = data[self.tuners_column].unique()

        contexts = data[self.contexts_column].map(
            {value: code for code, value in enumerate(unique_context)}
        ).values
        tuners = data[self.tuners_column].map(
            {value: code for code, value in enumerate(unique_tuners)}
        ).values

        outcomes = data['value'].values

        return unique_context,\
            unique_tuners, contexts, tuners, outcomes

    def __build(self, target):
        """
        """
        unique_context, unique_tuners, contexts, tuners,\
            outcomes = self.__define_data(target=target)

        coords = {
            'Tuners': unique_tuners,
            'Contexts': unique_context,
            'Outcomes': np.arange(outcomes.size)
        }
        with pm.Model(coords=coords) as model:
            contexts_idx = pm.Data(
                'contexts_idx',
                contexts,
                dims='Outcomes'
            )
            tuners_idx = pm.Data(
                'tuners_idx',
                tuners,
                dims='Outcomes'
            )

            hyper_mu = pm.TruncatedNormal(
                name='hyper_normal',
                mu=0.5,
                sd=0.01,
                lower=0,
                upper=1,
            )
            hyper_sigma = pm.Uniform(
                name='hyper_sigma',
                lower=0.01,
                upper=0.1
            )

            name = 'Context'
            intercept_dims = 'Contexts'
            varying_intercept = pm.TruncatedNormal(
                name=name,
                mu=hyper_mu,
                sd=hyper_sigma,
                lower=0,
                upper=1,
                dims=intercept_dims
            )

            tuner_slope = pm.TruncatedNormal(
                name='Tuner',
                mu=0.0,
                sd=0.1,
                dims='Tuners'
            )

            intercept = varying_intercept[contexts_idx]
            mu = pm.Deterministic(
                'mu',
                intercept
                + tuner_slope[tuners_idx],
            )
            sigma = pm.Uniform(
                name='sigma',
                lower=0.01,
                upper=0.1
            )

            out = pm.TruncatedNormal(
                name=f'observed_{target}',
                mu=mu,
                sd=sigma,
                observed=outcomes
            )
        plate = pm.model_graph.model_to_graphviz(
            model
        )
        setattr(self, 'model', model)
        setattr(self, 'plate', plate)

    def analyze(self, targets, figsize=(5, 5), include_time=False,
                **kwargs):
        """
        """
        for target in targets:

            self.__build(
                target=target
            )

            with self.model:

                traces = pm.sample(
                    **kwargs
                )

                print(target)
                summary = pm.summary(
                    traces,
                    var_names=['Tuner']
                )

                ax_1 = pm.traceplot(
                    traces,
                    figsize=figsize
                )

                var_names = ['Context']
                ax_2 = pm.plot_forest(
                    traces,
                    var_names=var_names,
                    combined=True,
                    ridgeplot_quantiles=[0.05, .25, .5, .75, 0.95],
                    figsize=figsize
                )
                ax_3 = pm.plot_forest(
                    traces,
                    var_names=['Tuner'],
                    combined=True,
                    ridgeplot_quantiles=[0.05, .25, .5, .75, 0.95],
                    figsize=figsize
                )

            print(summary)
            plt.tight_layout()
            plt.show()
