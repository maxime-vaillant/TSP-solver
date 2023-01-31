import math
from typing import Tuple, List
import random

import numpy as np


def generate_nodes(grid_size: Tuple[int, int], n_node: int) -> List[Tuple[int, int]]:
    """
    this function generate random coordinate of node in a grid
    :param grid_size: size of the grid (width, height)
    :param n_node: number of node wanted
    :return: a list of node
    """
    x_max, y_max = grid_size
    assert (x_max + 1) * (y_max + 1) >= n_node, 'too much nodes for the grid capacity'

    nodes = []

    while len(nodes) < n_node:
        x = random.randint(0, x_max)
        y = random.randint(0, y_max)

        if (x, y) not in nodes:
            nodes.append((x, y))

    return nodes


def calculate_nodes_distance(node_a: Tuple[int, int], node_b: Tuple[int, int]) -> float:
    """
    :param node_a: (Xa, Ya)
    :param node_b: (Xb, Yb)
    :return: euclidian distance between A and B
    """
    x_a, y_a = node_a
    x_b, y_b = node_b

    return math.sqrt(((x_b - x_a) / 2) ** 2 + ((y_b - y_a) / 2) ** 2)


def generate_circuit_from_nodes(nodes: List[Tuple[int, int]]) -> np.array:
    """
    this function is computing the distance matrix of given nodes
    :param nodes: a list of node
    :return: distance matrix
    """
    n_nodes = len(nodes)
    circuit = []

    for i in range(n_nodes):
        arc_list = []
        node_a = nodes[i]
        for j in range(n_nodes):
            node_b = nodes[j]
            arc_list.append(calculate_nodes_distance(node_a=node_a, node_b=node_b))
        circuit.append(arc_list)

    return np.array(circuit)
