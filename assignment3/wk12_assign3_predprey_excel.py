import matplotlib.pyplot as plt
import scipy as sc
import matplotlib
import numpy as np

from ema_workbench import (RealParameter, TimeSeriesOutcome, ema_logging,
                           perform_experiments)

from ema_workbench.connectors.excel import ExcelModel
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator
from ema_workbench.em_framework.evaluators import SequentialEvaluator


if __name__ == "__main__":
    ema_logging.log_to_stderr(level=ema_logging.INFO)

    model = ExcelModel("predatorPrey", wd=".",
                       model_file='PredPrey.xlsx')

    model.uncertainties = [RealParameter("B3", 0.015, 0.35), # prey_birth_rate
                           RealParameter("B4", 0.0005, 0.003), # predation_rate
                           RealParameter("B5", 0.001, 0.004), # predator_efficiency
                           RealParameter("B6", 0.04, 0.08), # predator_loss_rate
                           ]

    # specification of the outcomes
    model.outcomes = [TimeSeriesOutcome("B17:BDF17"),  # we can refer to a range in the normal way
                      TimeSeriesOutcome("B18:BDF18")]  # we can also use named range

    # name of the sheet
    model.default_sheet = "Sheet1"

    with SequentialEvaluator(model) as evaluator:
        results = perform_experiments(model, 25, reporting_interval=1,
                                      evaluator=evaluator)

    experiments, outcomes = results
    print(experiments.shape)
    print(list(outcomes.keys()))

    #How to plot this?