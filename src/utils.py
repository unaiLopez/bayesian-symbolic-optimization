
import re
import sympy as sp

from typing import Union, List

def check_lists_equal(list1: List[str], list2: List[str]) -> bool:
    for elem1 in list1:
        if elem1 not in list2:
            return False
    return True

def get_and_replace_consts(equation: str) -> Union[str, List[str]]:
    matches = re.finditer(r'\bconst\b', equation)
    replacements = {i: f'const{idx+1}' for idx, i in enumerate(match.start() for match in matches)}
    constants = [replacements[key] for key in replacements.keys()]
    new_equation = ""
    last_index = 0
    for index in sorted(replacements.keys()):
        new_equation += equation[last_index:index] + replacements[index]
        last_index = index + len('const')
    new_equation += equation[last_index:]

    return new_equation, constants