import numpy as np
import matplotlib.pyplot as plt

from ema_workbench import (Model, RealParameter, TimeSeriesOutcome, perform_experiments,
                           ema_logging)

from ema_workbench.connectors.netlogo import NetLogoModel
from ema_workbench.connectors.excel import ExcelModel
from ema_workbench.connectors.pysd_connector import PysdModel

from ema_workbench.em_framework.evaluators import LHS, SOBOL, MORRIS

from ema_workbench.analysis.plotting import lines, Density
from ema_workbench import Model, RealParameter, ScalarOutcome, Constant

ema_logging.log_to_stderr(level=ema_logging.INFO)


def predprey(prey_birth_rate=0.025, predation_rate=0.0015, predator_efficiency=0.002,
             predator_loss_rate=0.06, initial_prey=50, initial_predators=20, dt=0.25, final_time=365, reps=1):
    # Initial values
    predators, prey, sim_time = [np.zeros((reps, int(final_time / dt) + 1)) for _ in range(3)]

    for r in range(reps):
        predators[r, 0] = initial_predators
        prey[r, 0] = initial_prey

        # Calculate the time series
        for t in range(0, sim_time.shape[1] - 1):
            dx = (prey_birth_rate * prey[r, t]) - (predation_rate * prey[r, t] * predators[r, t])
            dy = (predator_efficiency * predators[r, t] * prey[r, t]) - (predator_loss_rate * predators[r, t])

            prey[r, t + 1] = max(prey[r, t] + dx * dt, 0)
            predators[r, t + 1] = max(predators[r, t] + dy * dt, 0)
            sim_time[r, t + 1] = (t + 1) * dt

    # Return outcomes
    return {'TIME': sim_time,
            'predators': predators,
            'prey': prey}


from ema_workbench import Model, RealParameter, TimeSeriesOutcome


model = Model('predprey', function=predprey)

# specify uncertainties
model.uncertainties = [RealParameter('prey_birth_rate', 0.015, 0.35),
                       RealParameter('predation_rate', 0.0005, 0.003),
                       RealParameter('predator_efficiency', 0.001, 0.004),
                       RealParameter('predator_loss_rate', 0.04, 0.08),
                       ]

# specify outcomes
model.outcomes = [TimeSeriesOutcome('prey'), # amount of prey
                  TimeSeriesOutcome('predators'), # amount of predators
                  ]

# model constants
model.constants = [Constant('dt', 0.25),
                   Constant('final_time', 365),
                   Constant('reps', 1)]

###
# Perform experiments
from ema_workbench import (SequentialEvaluator)


with SequentialEvaluator(model) as evaluator:
    results = perform_experiments(model, 100, reporting_interval=1,
                                  evaluator=evaluator)

experiments, outcomes = results
print(experiments.shape)
print(list(outcomes.keys()))

# How to plot this?
