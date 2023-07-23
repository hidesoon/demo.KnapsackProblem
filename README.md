# Knapsack Problem Solutions

Welcome to the `demo.KnapsackProblem` repository! This repository is dedicated to providing various algorithmic solutions to the classic optimization problem known as the Knapsack Problem.

## Overview

The Knapsack Problem is a fundamental combinatorial optimization problem where the goal is to select a number of items with given weights and values to fit into a knapsack of limited capacity. The objective is to maximize the total value of the items in the knapsack without exceeding its weight capacity.

## Solutions Provided

This repository offers multiple algorithmic approaches to solve the Knapsack Problem:

1. **Backtracking**: A recursive approach that explores all possible combinations to find the optimal solution. Check out the solution [here](./algorithms/backtrack.py).
   
2. **Branch and Bound**: An optimization technique that prunes the search space to find the optimal solution faster. Check out the solution [here](./algorithms/branch_and_bound.py).

3. **Dynamic Programming**: A bottom-up approach that breaks the problem into smaller subproblems and uses their solutions to build the solution for the main problem. Check out the solution [here](./algorithms/dynamic_programming.py).

4. **Genetic Algorithm**: An evolutionary algorithm inspired by the process of natural selection. Check out the solution [here](./algorithms/genetic_algorithm.py).

5. **Greedy Algorithm**: A heuristic method that makes locally optimal choices at each step. Check out the solution [here](./algorithms/greedy.py).

## Sample Data

A sample problem instance for the knapsack problem is provided in the [given_problem.json](./data/given_problem.json) file. You can use this data to test the provided algorithms.

## Running the Solutions

To see a demonstration of all the methods in action, run the `multiple_knapsack_solving_methods.py` script:

```bash
python multiple_knapsack_solving_methods.py
```
## Complexity Analysis of Knapsack Problem Algorithms

### 1. Backtrack Algorithm:
- **Time Complexity**: The worst-case time complexity is $O(2^n)$, where $n$ is the number of items. This arises due to the exploration of all possible combinations of items.
- **Space Complexity**: $O(n)$ due to the recursive call stack.

### 2. Branch and Bound Algorithm:
- **Time Complexity**: Worst-case time complexity is $O(2^n)$. However, due to pruning, the actual number of nodes explored can be significantly less.
- **Space Complexity**: Worst-case space complexity is $O(2^n)$ due to the maximum number of nodes stored in the queue. However, pruning can reduce the actual space usage.

### 3. Dynamic Programming Algorithm:
- **Time Complexity**: $O(n \times W)$, where $n$ is the number of items and $W$ is the maximum weight capacity of the knapsack.
- **Space Complexity**: $O(n \times W)$ due to the 2D table used for storing solutions to subproblems.

### 4. Genetic Algorithm:
- **Time Complexity**: $O(n \times \text{POP SIZE} \times \text{MAX GEN})$, where $n$ is the number of items, $\text{POP SIZE}$ is the population size, and $\text{MAX GEN}$ is the maximum number of generations.
- **Space Complexity**: $O(\text{POP SIZE})$ due to the storage of the population.

### 5. Greedy Algorithm:
- **Time Complexity**: $O(n \log n)$ due to the sorting step based on the value-to-weight ratio.
- **Space Complexity**: $O(n)$ for storing the value-to-weight ratios and the sorted list of items.
