import numpy as np

from typing import Union, List, Callable

class Metrics:
    def __init__(self, metric: Union[str, Callable[[Union[List[float], np.array], Union[List[float], np.array]], float]]):
        self.metric = metric

    def calculate(self, y: Union[List[float], np.array], y_preds: Union[List[float], np.array]) -> float:
        if isinstance(self.metric, str):
            if self.metric == "mae":
                return np.mean(np.abs(y - y_preds))
            elif self.metric == "mse":
                return np.mean((y - y_preds) ** 2)
            elif self.metric == "rmse":
                return np.mean(np.sqrt((y - y_preds) ** 2))
            else:
                raise NotImplementedError(f"Metric {self.metric} is not supported. Please try 'mae', 'mse' or 'rmse' instead...")
        elif isinstance(self.metric, Callable):
            try:
                return self.metric(y, y_preds)
            except Exception as e:
                raise RuntimeError(f"Custom metric should receive 2 arrays (predictions and ground truth) and shoul return a float value...")
        else:
            raise ValueError(f"Metric of type {type(self.metric)} is not supported. Try a 'str' of 'Callable' instead...")