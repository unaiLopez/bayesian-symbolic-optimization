import utils

import sympy as sp
import numpy as np
import pandas as pd

from typing import List
from metrics import Metrics
from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult

def optimize_constants(equation: sp.Symbol, X: pd.DataFrame, y: pd.Series, variables: List[sp.Symbol], constants: List[sp.Symbol], metric: Metrics) -> OptimizeResult:
    if isinstance(constants, sp.Symbol):
        constants = tuple([constants])
        variables = tuple(set(equation.free_symbols) - set(constants))
    else:
        variables = tuple(set(equation.free_symbols) - set(constants))

    if len(variables) > 0:
        equation_lambdified = sp.lambdify((*variables, *constants), equation, modules='numpy')
    else:
        equation_lambdified = sp.lambdify(constants, equation, modules='numpy')
    equation_vectorized = np.vectorize(equation_lambdified)

    def objective_function(params: np.array, X: pd.DataFrame, y: pd.Series) -> float:
        constant_values = params
        constant_values = [np.array([constant_value] * len(y)) for constant_value in constant_values]
        features_values = list()

        if len(variables) > 0:
            equation_variable_symbols = [str(var) for var in list(variables)]
            features = list(X.columns)
            if not utils.check_lists_equal(features, equation_variable_symbols):
                features_values = [X[feature].values for feature in features if feature in equation_variable_symbols]
            else:
                features_values = [X[feature].values for feature in X]
            y_pred = equation_vectorized(*features_values, *constant_values)
        else:
            y_pred = equation_vectorized(*constant_values)
        
        return metric.calculate(y, y_pred)
    
    initial_guess = np.ones(len(constants))

    return minimize(objective_function, initial_guess,  options={'maxiter': 20, 'disp': False}, args=(X, y.values))