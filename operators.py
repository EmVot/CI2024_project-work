import numpy as np
import random as rnd

BINARY_OPERATORS = {
    '+' : np.add,
    '-' : np.subtract,
    '*' : np.multiply,
    '/' : lambda x, y: x / y if y != 0 else 1,
    #'^' : np.power
}

UNARY_OPERATORS = {
    "": lambda x: x,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "log": np.log,
    "arccos": np.arccos,
    "arcsin": np.arcsin,
    "arctan": np.arctan,
    "sqrt": np.sqrt,
    "cbrt": np.cbrt,
    "abs": np.abs,
    "^(-1)": np.reciprocal,
}