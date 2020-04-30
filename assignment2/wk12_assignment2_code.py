'''
Created on May 2, 2017
@author: jhkwakkel
'''
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from scipy.optimize import brentq
from multiprocessing import Process, freeze_support

# lake model is now in wk12_lake_model_solo.py to prevent parallel errors
from wk12_lake_model_solo import lake_model_actual


start=time.time()

def lake_model():
    lake_model_actual()


if __name__ == '__main__':
    freeze_support()
    lake_model_actual()



#####################################################################################################

# now connect the model with the workbench
from ema_workbench import Model, RealParameter, ScalarOutcome, Constant


model = Model('lakeproblem', function=lake_model_actual)
model.time_horizon = 100

# specify uncertainties
model.uncertainties = [RealParameter('b', 0.1, 0.45),
                       RealParameter('q', 2.0, 4.5),
                       RealParameter('mean', 0.01, 0.05),
                       RealParameter('stdev', 0.001, 0.005),
                       RealParameter('delta', 0.93, 0.99)]

# specify outcomes
model.outcomes = [ScalarOutcome('max_P'),
                  ScalarOutcome('utility'),
                  ScalarOutcome('inertia'),
                  ScalarOutcome('reliability')]

# set levers
#model.levers = [RealParameter('decisions', 0, 0.1)]
model.levers = [RealParameter(str(i), 0, 0.1) for i in
                     range(model.time_horizon)]

# model constants
model.constants = [Constant('alpha', 0.4),
                   Constant('nsamples', 100),
                   Constant('steps', 100)]

#####################################################################################################

# performing experiments
from ema_workbench import (MultiprocessingEvaluator, ema_logging, perform_experiments)
ema_logging.log_to_stderr(ema_logging.INFO)

if __name__ == '__main__':
    freeze_support()
    with MultiprocessingEvaluator(model, n_processes=7) as evaluator:
        results = evaluator.perform_experiments(scenarios=1000, policies=4)

#####################################################################################################

# process the results of the experiments
    experiments, outcomes = results
    print(experiments.shape)
    print(list(outcomes.keys()))
    stop = time.time()
    print(f"Runtime in minutes: { ((stop-start)/60) }")

    from ema_workbench.analysis import pairs_plotting

    fig, axes = pairs_plotting.pairs_scatter(experiments, outcomes, group_by='policy',
                                             legend=False)
    fig.set_size_inches(8, 8)
    plt.show()




##PLOTTEN LUKT NIET
    # from ema_workbench.analysis import plotting, plotting_util
    #
    #
    # for outcome in outcomes.keys():
    #     plotting.lines(experiments, outcomes, outcomes_to_show=outcome,
    #                density=plotting_util.Density.HIST)
    # plt.show()


# # Printing as done on workbench website example
#     policies = experiments['policy']
#     for i, policy in enumerate(np.unique(policies)):
#         experiments.loc[policies==policy, 'policy'] = str(i)
#
#     data = pd.DataFrame(outcomes)
#     data['policy'] = policies
#
#     sns.pairplot(data, hue='policy', vars=list(outcomes.keys()), diag_kind='auto', diag_kws=None)
#     plt.show()

#
# # generate some random policies by sampling over levers
# n_scenarios = 100
# n_policies = 1
#
# # with SequentialEvaluator(lake_problem) as evaluator:
# #     res = evaluator.perform_experiments(n_scenarios, n_policies,
# #                                             levers_sampling=MC)
#
# # from assignment1
# with SequentialEvaluator(model) as evaluator:
#     experiments, outcomes = evaluator.perform_experiments(n_scenarios)
#
#
# # from ema_workbench.analysis import plotting, plotting_util
# #
# #
# # for outcome in outcomes.keys():
# #     plotting.lines(experiments, outcomes, outcomes_to_show=,
# #                    density=plotting_util.Density.HIST)
# # plt.show()
#
# # from ema_workbench.analysis import pairs_plotting
# #
# # fig, axes = pairs_plotting.pairs_scatter(experiments, outcomes, group_by='policy',
# #                                          legend=False)
# # fig.set_size_inches(8,8)
# # plt.show()
#
# from ema_workbench import (MultiprocessingEvaluator, ema_logging,
#                            perform_experiments)
# ema_logging.log_to_stderr(ema_logging.INFO)
#
# with MultiprocessingEvaluator(model, n_processes=1) as evaluator:
#     experiments, outcomes = evaluator.perform_experiments(scenarios=100, policies=1)
