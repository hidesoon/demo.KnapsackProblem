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

### 1. Backtrack Algorithm
- **Time Complexity**: $O(2^n)$
  - The algorithm explores all possible combinations of items, leading to an exponential number of recursive calls.
- **Space Complexity**: \(O(n)\)
  - Due to the recursive call stack.

### 2. Branch and Bound Algorithm
- **Time Complexity**: \(O(2^n)\) 
  - In the worst case, the algorithm might need to explore all possible combinations of items. However, pruning often reduces the actual search space.
- **Space Complexity**: \(O(2^n)\) 
  - Determined by the maximum number of nodes stored in the queue.

### 3. Dynamic Programming Algorithm
- **Time Complexity**: \(O(n \times W)\) 
  - The algorithm fills up a table of size \(n \times W\), where \(W\) is the maximum weight capacity.
- **Space Complexity**: \(O(n \times W)\) 
  - Due to the 2D table used for storing solutions.

### 4. Genetic Algorithm
- **Time Complexity**: \(O(n \times \text{POP_SIZE} \times \text{MAX_GEN})\)
  - The algorithm evaluates the fitness of the entire population over multiple generations.
- **Space Complexity**: \(O(\text{POP_SIZE})\)
  - Determined by the size of the population.

### 5. Greedy Algorithm
- **Time Complexity**: \(O(n \log n)\)
  - Dominated by the sorting step based on the value-to-weight ratio.
- **Space Complexity**: \(O(n)\)
  - For storing the value-to-weight ratios and the sorted list of items.

## Contributing

Feel free to contribute to this repository by submitting pull requests or raising issues. All contributions are welcome!

## License

This project is open-source and available under the MIT License.
