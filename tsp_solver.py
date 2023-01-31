import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pulp as pl
import time
from typing import Tuple, List, Any

__author__ = "Maxime Vaillant"

from helper import generate_nodes, generate_circuit_from_nodes


def initialize_tsp(circuit: np.array) -> Tuple[np.array, pl.LpProblem]:
    """
    this function is initializing the TSP problem by allocating variables from the distance matrix
    an arc is represented as a variable X_nodeA_nodeB and can be active or not (0 or 1)
    a node can only have two arcs actives (sum of X_nodeA_nobeB where nodeA or nodeB is nodeX = 2)
    :param circuit: distance matrix
    :return: names of variables and the solver's model
    """
    n_node = circuit.shape[0]

    assert circuit.shape == (n_node, n_node), 'not good shape'

    model = pl.LpProblem("TSP problem", pl.LpMinimize)

    # Create variables named like X_nodeA_nodeB
    variable_names = ['{}_{}'.format(i, j) for j in range(1, n_node + 1) for i in range(1, n_node + 1)]

    # Declare variables as integer equal to 0 or 1
    variables = pl.LpVariable.matrix("X", variable_names, cat="Integer", lowBound=0, upBound=1)

    allocation = np.array(variables).reshape(n_node, n_node).T

    obj_func = pl.lpSum(allocation[i, j] * circuit[i, j] for i, j in itertools.combinations(range(0, n_node), 2))

    model += obj_func

    for i in range(n_node):
        lp_sum = 0
        for j in range(n_node):
            if circuit[i, j] != 0:
                lp_sum += allocation[j, i] if i > j else allocation[i, j]

        # Each node have one arc in, one arc out
        model += pl.lpSum(lp_sum) == 2

    return allocation, model


def solve_problem(model: pl.LpProblem) -> Tuple[List[List[int]], Any]:
    """
    this function is solving the TSP problem with partial constraints
    :param model: solver's model
    :return: all the partial solutions, a partial solution is a list of nodes in a closed circuit
    """
    solver = pl.PULP_CBC_CMD()

    model.solve(solver)

    solution = []
    all_partial_solutions = []

    for var in model.variables():
        if var.value() == 1:
            solution.append(var)

    # Identify all Hamiltonian circuits in the solution
    while solution:
        nodes_in_partial_solution = []
        i, j = [int(x) for x in solution[0].name.split('_')[1:]]

        solution.pop(0)
        nodes_in_partial_solution.append(i)
        cur_node = j

        while cur_node not in nodes_in_partial_solution:
            for index, var in enumerate(solution):
                i, j = [int(x) for x in var.name.split('_')[1:]]

                if i == cur_node:
                    nodes_in_partial_solution.append(cur_node)
                    solution.pop(index)
                    cur_node = j
                    break
                if j == cur_node:
                    nodes_in_partial_solution.append(cur_node)
                    solution.pop(index)
                    cur_node = i
                    break

        all_partial_solutions.append(nodes_in_partial_solution)

    return all_partial_solutions, model


def create_new_constraint(nodes_in_partial_solution: List[int], allocation: np.array, circuit: np.array) -> pl.lpSum:
    """
    this function is creating a new constraint from partial solution
    :param nodes_in_partial_solution: nodes that are in the partial solution
    :param allocation: names of variables
    :param circuit: distance matrix
    :return: the new constraint to add to the model
    """
    new_constraint = []

    for node in nodes_in_partial_solution:
        for index, x in enumerate(circuit[node - 1]):
            if x != 0 and index + 1 not in nodes_in_partial_solution:
                if node - 1 > index:
                    new_constraint.append(allocation[index, node - 1])
                else:
                    new_constraint.append(allocation[node - 1, index])

    return pl.lpSum(np.array(new_constraint)) >= 2


def print_solution(model):
    """
    this function is printing current solution
    :param model: solver's model
    """
    print("SOLUTION")

    for v in model.variables():
        try:
            print(v.name, "=", v.value())
        except Exception:
            print("error couldnt find value")


def show_solution(nodes: List[Tuple[int, int]], model):
    """
    this function is plotting the solution
    :param nodes: list of node
    :param model: solver's model
    """
    x, y = zip(*nodes)

    plt.plot(x, y, 'o')

    nodes_in_partial_solution = []
    solution = []

    for var in model.variables():
        if var.value() == 1:
            solution.append(var)

    while solution:
        arc_x, arc_y = [], []
        i_1, j_1 = [int(x) for x in solution.pop(0).name.split('_')[1:]]

        arc_x.append(x[i_1 - 1])
        arc_y.append(y[i_1 - 1])
        cur_node = j_1

        # The objective of the loop is to localize a Hamiltonian circuit in the solution
        while cur_node not in nodes_in_partial_solution:
            for index, var in enumerate(solution):
                i, j = [int(x) for x in var.name.split('_')[1:]]

                if i == cur_node:
                    nodes_in_partial_solution.append(cur_node)
                    arc_x.append(x[cur_node - 1])
                    arc_y.append(y[cur_node - 1])
                    solution.pop(index)
                    cur_node = j
                    break
                if j == cur_node:
                    nodes_in_partial_solution.append(cur_node)
                    arc_x.append(x[cur_node - 1])
                    arc_y.append(y[cur_node - 1])
                    solution.pop(index)
                    cur_node = i
                    break
            if cur_node == i_1:
                break

        arc_x.append(x[i_1 - 1])
        arc_y.append(y[i_1 - 1])

        plt.plot(arc_x, arc_y)
    plt.show()


def solve_tsp(circuit: np.array, verbose: bool = False):
    """
    this function is solving the TSP problem from a given distance matrix
    :param circuit: distance matrix
    :param verbose: allow partial printing
    """
    n_node = circuit.shape[0]
    allocation, model = initialize_tsp(circuit)

    all_partial_solutions, model = solve_problem(model)

    i = 1

    print("Iteration {} : {} / {}".format(i, max([len(x) for x in all_partial_solutions]), n_node))

    # While solution contains more than 1 Hamiltonian circuit, then the TSP problem isn't solved
    while len(all_partial_solutions) > 1:
        for nodes_in_partial_solution in all_partial_solutions:
            model += create_new_constraint(nodes_in_partial_solution, allocation, circuit)

        all_partial_solutions, model = solve_problem(model)

        i += 1

        if verbose:
            show_solution(nodes, model)

        print("Iteration {}".format(i))

    print_solution(model)
    print("Total of iteration : {}".format(i))
    show_solution(nodes, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_node', default=50, type=int)
    parser.add_argument('-s', '--grid_size', nargs='+', default=[150, 150], type=int)
    parser.add_argument('-v', '--verbose', default=False, type=bool)
    parser.add_argument('-t', '--timer', default=False, type=bool)

    args = parser.parse_args()

    n_node = args.n_node
    grid_size = tuple(args.grid_size)
    verbose = args.verbose
    timer = args.timer if not verbose else False

    start = time.time() if timer else 0

    nodes = generate_nodes(grid_size=grid_size, n_node=n_node)
    circuit = generate_circuit_from_nodes(nodes=nodes)

    solve_tsp(circuit=circuit, verbose=verbose)

    if timer:
        end = time.time()
        print('Total execution time:', end - start, 'seconds')
