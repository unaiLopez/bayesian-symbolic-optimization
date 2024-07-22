import optuna

class MetricEarlyStoppingCallback:
    def __init__(self, stopping_criteria: float):
        self.stopping_criteria = stopping_criteria

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        try:
            if study.best_value <= self.stopping_criteria:
                study.stop()
        except ValueError:
            pass
        