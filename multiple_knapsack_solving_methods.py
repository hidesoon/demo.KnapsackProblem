import json
import time

from algorithms.dynamic_programming import knapsack_dp
from algorithms.backtrack import knapsack_backtrack
from algorithms.greedy import knapsack_greedy
from algorithms.branch_and_bound import BranchAndBoundKnapsack
from algorithms.genetic_algorithm import KnapsackGenetic


def get_model_input(filename='given_problem.json') -> dict:
    with open(filename, 'r') as f:
        given_problem = json.load(f)

    capacity = given_problem.get('capacity')
    weights = []
    values = []
    for item in given_problem.get('items'):
        weights.append(item.get('weight'))
        values.append(item.get('value'))

    print(f"pack capacity: {capacity}")
    print(f"items: {given_problem.get('items')}")

    return weights, values, capacity, given_problem.get('items')


def run(filename='given_problem.json', methods=['dp', 'backtrace', 'greedy', 'branch_and_bound', 'GA']):
    print("set problem-----------------------------")
    weights, values, capacity, items = get_model_input(filename)
    items_selected_idx, max_value = knapsack_backtrack(
        weights, values, capacity)
    print("solving-----------------------------")
    for method in methods:
        print(f"with {method} method-----------------------------")
        tic = time.time()
        match method:
            case 'dp':
                items_selected_idx, max_value = knapsack_dp(
                    weights, values, capacity)
                solutions = [(items_selected_idx, max_value)]
            case 'backtrace':
                items_selected_idx, max_value = knapsack_backtrack(
                    weights, values, capacity)
                solutions = [(items_selected_idx, max_value)]
            case 'greedy':
                items_selected_idx, max_value = knapsack_greedy(
                    weights, values, capacity)
                solutions = [(items_selected_idx, max_value)]
            case 'branch_and_bound':
                kbb = BranchAndBoundKnapsack()
                items_selected_idx, max_value = kbb.knapsack_branch_and_bound(
                    weights, values, capacity)
                solutions = [(items_selected_idx, max_value)]
            case 'GA':
                kga = KnapsackGenetic()
                solutions = kga.solve(weights, values, capacity)
            case _:
                print('method not supported')
        toc = time.time()
        # elapsed time in ns
        elapsed_time = (toc - tic) * 1e9
        print(f"elapsed time: {elapsed_time:0.4f} ns")
        for solution in solutions:
            items_selected_idx = solution[0]
            max_value = solution[1]
            items_selected_idx.sort()
            items_selected = [items[i] for i in items_selected_idx]
            print(f"selected items: {items_selected}")
            print(f"max value: {max_value}")
    print("completed-----------------------------")


if __name__ == '__main__':
    for i in range(10):
        print(f"run {i}+++++++++++++++++++++++++++++++")
        run(filename="data\given_problem.json")
