import io

import optuna
import graphviz

import numpy as np
import matplotlib.pyplot as plt

from node import Node
from PIL import Image
from typing import List

class BinaryTree:
    def __init__(self, trial: optuna.Trial, depth: int, variables: List[str], unary_operators: List[str], binary_operators: List[str]):
        self.trial = trial
        self.depth = depth
        self.variables = variables
        self.unary_operators = unary_operators
        self.binary_operators = binary_operators
        self.operators = unary_operators + binary_operators

        self.index = 0
        self.no_constant_yet = True
        self.root = self._build_bayesian_tree(trial, depth)
        self.equation = self.build_equation(self.root)

    def _build_bayesian_tree(self, trial: optuna.Trial, depth: int) -> Node:
        if depth == 0:
            value_id = f'values_depth_{self.depth - depth}_node_{self.index}'
            value = trial.suggest_categorical(value_id, self.variables)
            self.index += 1

            return Node(
                id=value_id,
                value=value
            )
        else:
            operator_id = f'any_operator_and_values_depth_{self.depth - depth}_node_{self.index}'
            operator = trial.suggest_categorical(operator_id, self.operators + list(self.variables))

            node = Node(
                id=operator_id,
                value=operator
            )
            self.index += 1

            if operator in self.unary_operators:
                node.right = self._build_bayesian_tree(trial, depth - 1)
            elif operator in self.binary_operators:
                node.left = self._build_bayesian_tree(trial, depth - 1)
                node.right = self._build_bayesian_tree(trial, depth - 1)
            
            return node

    def evaluate(self, node: Node, variables: dict) -> float:
        if not node:
            return 0
        else:
            if not node.left and not node.right:
                if node.value in variables:
                    return variables[node.value]
                else:
                    return float(node.value)
            
            if node.left and node.right:
                left_val = self.evaluate(node.left, variables)
                right_val = self.evaluate(node.right, variables)
                
                if node.value == '+':
                    return left_val + right_val
                elif node.value == '-':
                    return left_val - right_val
                elif node.value == '*':
                    return left_val * right_val
                elif node.value == '/':
                    try:
                        return left_val / right_val
                    except ZeroDivisionError:
                        raise optuna.exceptions.TrialPruned("Trial pruned due to division by zero.")
            
            elif node.right and not node.left:
                right_val = self.evaluate(node.right, variables)

                if node.value == 'sin':
                    return np.sin(right_val)
                elif node.value == 'cos':
                    return np.cos(right_val)
                elif node.value == 'tan':
                    return np.tan(right_val)
                elif node.value == 'neg':
                    return -right_val
                elif node.value == 'Abs':
                    return np.abs(right_val)
                elif '**-' in node.value:
                    try:
                        return np.power(right_val, float(node.value.split('**')[1]))
                    except ZeroDivisionError:
                        raise optuna.exceptions.TrialPruned("Trial pruned due to division by zero.")
                elif '**' in node.value and '-' not in node.value:
                    return np.power(right_val, float(node.value.split('**')[1]))
                elif node.value == 'sqrt':
                    if right_val > 0:
                        return np.sqrt(right_val) 
                    else:
                        raise optuna.exceptions.TrialPruned("Trial pruned due to passing negative numbers to square root, which would result in complex numbers.")
                elif node.value == 'log':
                    if right_val > 0:
                        return np.log(right_val)
                    else:
                        raise optuna.exceptions.TrialPruned("Trial pruned due to passing negative numbers to logarithm.")
                elif node.value == 'exp':
                    return np.exp(right_val)
            else:
                raise ValueError("Invalid node structure")

    def build_equation(self, node: Node) -> str:
        if not node:
            return ""
        if not node.left and not node.right:
            return str(node.value)
        elif node.right and not node.left:
            right_expr = self.build_equation(node.right)
            if "**" in node.value:
                return f"({right_expr}{node.value})"
            elif "neg" in node.value:
                return f"(-{right_expr})"
            else:
                return f"({node.value}({right_expr}))"
        elif node.left and node.right:
            left_expr = self.build_equation(node.left)
            right_expr = self.build_equation(node.right)
            return f"({left_expr}{node.value}{right_expr})"
        
    def visualize_binary_tree(self) -> None:
        graph = graphviz.Digraph()
        graph.node(name=str(self.root.id), label=str(self.root.value), style="filled", fillcolor="red")

        def add_nodes_edges(node):
            if node.left:
                if node.left.value not in list(self.variables) + ["const"]:
                    left_fill_color = "red"
                    shape = None
                else:
                    left_fill_color = "green"
                    shape = "rectangle"

                graph.node(name=str(node.left.id), label=str(node.left.value), shape=shape, style="filled", fillcolor=left_fill_color)
                graph.edge(str(node.id), str(node.left.id))
                add_nodes_edges(node.left)
            if node.right:
                if node.right.value not in list(self.variables) + ["const"]:
                    right_fill_color = "red"
                    shape = None
                else:
                    right_fill_color = "green"
                    shape = "rectangle"

                graph.node(name=str(node.right.id), label=str(node.right.value), shape=shape, style="filled", fillcolor=right_fill_color)
                graph.edge(str(node.id), str(node.right.id))
                add_nodes_edges(node.right)

        add_nodes_edges(self.root)

        #graph_path = 'equation_tree.png'
        #graph.render(graph_path, format='png', cleanup=True)
        graph_data = graph.pipe(format='png')
        img = Image.open(io.BytesIO(graph_data))
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()    