from ema_workbench import (RealParameter, TimeSeriesOutcome, ema_logging,
                           perform_experiments, MultiprocessingEvaluator)

from ema_workbench.connectors.vensim import VensimModel

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    model = VensimModel("PredPrey", wd='.',
                        model_file='PredPrey.vpmx') # CANNOT MAKE A VPM FILE, SO THIS CONNECTOR CANNOT BE USED?

    # outcomes
    model.outcomes = [TimeSeriesOutcome('predators'),
                      TimeSeriesOutcome('prey')]

    # Plain Parametric Uncertainties
    model.uncertainties = [RealParameter('prey birth rate', 0.015, 0.35),
                           RealParameter('predation rate', 0.0005, 0.003),
                           RealParameter('predator efficiency', 0.001, 0.004),
                           RealParameter('predator loss rate', 0.04, 0.08),
                           ]

    nr_experiments = 10
    with MultiprocessingEvaluator(model) as evaluator:
        results = perform_experiments(model, nr_experiments,
                                      evaluator=evaluator)

import matplotlib.pyplot as plt
from ema_workbench.analysis.plotting import lines

figure = lines(results, density=True) #show lines, and end state density
plt.show() #show figure