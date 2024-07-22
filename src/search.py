import utils
import optuna

import numpy as np
import sympy as sp
import pandas as pd

from metrics import Metrics
from binary_tree import BinaryTree
from optimize_constants import optimize_constants
from callbacks import MetricEarlyStoppingCallback
from typing import Optional, Union, List, Callable


class BayesianSymbolicRegressor:
    def __init__(
            self,
            max_tree_depth: int,
            metric: Union[str, Callable[[Union[List[float], np.array], Union[List[float], np.array]], float]],
            n_iterations: int,
            timeout: Optional[int] = None,
            use_constants: Optional[bool] = True,
            stopping_criteria: Optional[float] = None
    ):
        self.max_tree_depth = max_tree_depth
        self.metric = Metrics(metric)
        self.n_iterations = n_iterations
        self.timeout = timeout
        self.stopping_criteria = stopping_criteria
        self.use_constants = use_constants
        self.binary_operators = ['+', '-', '*', '/']
        self.unary_operators = []#['sin', 'cos', 'tan', 'Abs', '**2', '**3', '**-2', '**-3', 'sqrt', 'log', 'exp', 'neg']
        self.operators = self.binary_operators + self.unary_operators

        self.equations = {
            "number": [],
            "raw_tree_equation": [],
            "equation": []
        }
        
        self.equation_search_history = set()
        self.best_equation = None
        self.trials_dataframe = None
        self.study = None

    def _check_if_equation_already_explored(self, equation: str):
        if equation in self.equation_search_history:
            raise optuna.exceptions.TrialPruned(f"Trial pruned due to repeated equation {equation}")
        else:
            self.equation_search_history.add(equation)

    def _create_objective(self, X: pd.DataFrame, y: pd.Series, max_tree_depth: int, unary_operators: List[str], binary_operators: List[str], variables: List[str]) -> Callable[[optuna.Trial], float]:
        def _objective(trial: optuna.Trial) -> float:
            max_depth = trial.suggest_int('max_depth', 0, max_tree_depth)
            bt = BinaryTree(trial, max_depth, variables, unary_operators, binary_operators)
            #self._check_if_equation_already_explored(bt.equation)
            self.equations["number"].append(trial.number)
            print(bt.equation)

            try:
                if "const" not in bt.equation:
                    self.equations["raw_tree_equation"].append(bt.equation)
                    sympy_equation = sp.sympify(bt.equation)
                    equation_free_symbols = [str(var) for var in list(sympy_equation.free_symbols)]
                    equation_lambdified = sp.lambdify(equation_free_symbols, sympy_equation, modules='numpy')
                    equation_vectorized = np.vectorize(equation_lambdified)

                    features = list(X.columns)
                    features_values = list()
                    if not utils.check_lists_equal(features, equation_free_symbols):
                        features_values = [X[feature].values for feature in features if feature in equation_free_symbols]
                    else:
                        features_values = [X[feature] for feature in features]

                    result = equation_vectorized(*features_values)
                    metric = self.metric.calculate(y.values, result)
                else:
                    equation, constants = utils.get_and_replace_consts(bt.equation)
                    self.equations["raw_tree_equation"].append(equation)
                    sympy_equation = sp.sympify(equation)

                    non_const_variables = variables.copy()
                    non_const_variables.remove("const")
                    constant_symbols = sp.symbols(" ".join(constants))
                    variable_symbols = sp.symbols(" ".join(non_const_variables))

                    result = optimize_constants(sympy_equation, X, y, variable_symbols, constant_symbols, self.metric)
                    optimized_constants = result.x
                    metric = result.fun
                    
                    if isinstance(constant_symbols, sp.Symbol):
                        optimized_const_dict = dict(zip([constant_symbols], optimized_constants))
                    else:
                        optimized_const_dict = dict(zip(list(constant_symbols), optimized_constants))
                    sympy_equation = sympy_equation.subs(optimized_const_dict)

                print(f"y = {sympy_equation}")          
                self.equations["equation"].append(sympy_equation)
                if isinstance(metric, float) and (metric != metric):
                    raise optuna.exceptions.TrialPruned("Trial pruned due to objective NaN")

                return metric
            
            except SyntaxError as se:
                raise optuna.exceptions.TrialPruned(f"Trial pruned due to {se}")
            except RuntimeError as re:
                raise optuna.exceptions.TrialPruned(f"Trial pruned due to {re}")
            except ValueError as ve:
                raise optuna.exceptions.TrialPruned(f"Trial pruned due to {ve}")
            except TypeError as te:
                raise optuna.exceptions.TrialPruned(f"Trial pruned due to {te}")
            except KeyError as ke:
                raise optuna.exceptions.TrialPruned(f"Trial pruned due to {ke}")
            except ZeroDivisionError as zde:
                raise optuna.exceptions.TrialPruned(f"Trial pruned due to {zde}")
            except OverflowError as oe:
                raise optuna.exceptions.TrialPruned(f"Trial pruned due to {oe}")
            except RuntimeWarning as rw:
                raise optuna.exceptions.TrialPruned(f"Trial pruned due to {rw}")
            finally:
                if len(self.equations["number"]) > len(self.equations["equation"]):
                    self.equations["equation"].append(None)
        return _objective
                
    def _order_and_save_trials_dataframe(self):
        self.trials_dataframe = self.trials_dataframe.sort_values(by='value', ascending=True)

    def _optimize(self, objective):
        if self.stopping_criteria is None and self.timeout is None:
             self.study.optimize(objective, n_trials=self.n_iterations, gc_after_trial=True)
        elif self.stopping_criteria is None and self.timeout is not None:
             self.study.optimize(objective, timeout=self.timeout, n_trials=self.n_iterations, gc_after_trial=True)
        elif self.stopping_criteria is not None and self.timeout is None:
            metric_early_stopping = MetricEarlyStoppingCallback(stopping_criteria=self.stopping_criteria)
            self.study.optimize(objective, n_trials=self.n_iterations, gc_after_trial=True, callbacks=[metric_early_stopping])
        else:
            metric_early_stopping = MetricEarlyStoppingCallback(stopping_criteria=self.stopping_criteria)
            self.study.optimize(objective, timeout=self.timeout, n_trials=self.n_iterations, gc_after_trial=True, callbacks=[metric_early_stopping])
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        variables = list(X.columns)
        if self.use_constants: variables += ["const"]

        self.study = optuna.create_study(direction='minimize')
        objective = self._create_objective(
            X,
            y,
            self.max_tree_depth,
            self.unary_operators,
            self.binary_operators,
            variables
        )
        self._optimize(objective)

        self.trials_dataframe = self.study.trials_dataframe()
        self.trials_dataframe = pd.merge(
            self.trials_dataframe,
            pd.DataFrame(self.equations),
            on=["number"],
            how="left"
        )
        self._order_and_save_trials_dataframe()
        self.best_equation = sp.sympify(self.trials_dataframe["equation"].values[0])

    def plot_optimization_history(self):
        fig = optuna.visualization.plot_optimization_history(self.study)
        fig.show()
        
    def predict(self, X: pd.DataFrame):
        lambdified_equation = sp.lambdify(tuple(X.columns), self.best_equation, modules='numpy')
        predictions = lambdified_equation(**X)
        if not isinstance(predictions, pd.Series):
            predictions = np.array([predictions] * X.shape[0])
        else:
            predictions = predictions.values
        return predictions