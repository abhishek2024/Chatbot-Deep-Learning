from itertools import chain
from typing import List, Iterable


def flatten_nested_list(l):
    """
    Flatten multiple nested list
    :param l: nested list
    :return: flattened list
    """
    for sublist in l:
        while isinstance(sublist, List):
            res = list(chain(sublist))
            return flatten_nested_list(res)
    return l
