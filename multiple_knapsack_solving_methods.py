import random
from queue import Queue
import json
import time


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


def knapsack_dp(weights, values, W):

    n = len(weights)
    dp = [[0 for x in range(W + 1)] for y in range(n + 1)]

    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1]
                               [w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]

    # 回溯，找出选择的物品
    w = W
    items_selected = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            items_selected.append(i-1)  # 第i-1个物品被选中
            w -= weights[i-1]

    return items_selected, dp[n][W]


def knapsack_backtrack(weights, values, W):
    n = len(weights)
    best_value = 0
    best_combination = []

    def backtrack(i, current_weight, current_value, current_combination):
        nonlocal best_value, best_combination
        if i == n or current_weight > W:
            if current_value > best_value:
                best_value = current_value
                best_combination = current_combination.copy()
            return
        # 不选第i个物品
        backtrack(i+1, current_weight, current_value, current_combination)
        # 选第i个物品
        if current_weight + weights[i] <= W:
            current_combination.append(i)
            backtrack(i+1, current_weight +
                      weights[i], current_value + values[i], current_combination)
            current_combination.pop()

    backtrack(0, 0, 0, [])
    return best_combination, best_value


def knapsack_greedy(weights, values, W):
    n = len(weights)
    # 按价值/重量比率排序
    ratios = [(i, values[i] / weights[i]) for i in range(n)]
    ratios.sort(key=lambda x: x[1], reverse=True)

    total_weight = 0
    total_value = 0
    items_selected = []

    for i, _ in ratios:
        if total_weight + weights[i] <= W:
            total_weight += weights[i]
            total_value += values[i]
            items_selected.append(i)

    return items_selected, total_value


class Node:
    def __init__(self, level, weight, profit, bound):
        self.level = level
        self.weight = weight
        self.profit = profit
        self.bound = bound


def bound(node, n, W, wt, val):
    if node.weight >= W:
        return 0
    profit_bound = node.profit
    j = node.level + 1
    total_wt = node.weight
    while j < n and total_wt + wt[j] <= W:
        total_wt += wt[j]
        profit_bound += val[j]
        j += 1
    if j < n:
        profit_bound += (W - total_wt) * (val[j] / wt[j])
    return profit_bound


def bound(u, n, W, weights, values):
    # 如果当前重量已经超过背包的容量，则此路径不可行，返回0
    if u["weight"] >= W:
        return 0

    # 设定当前考虑物品的索引为下一个
    j = u["level"] + 1
    # 从当前价值和重量开始，计算可能的上界
    bound_value = u["value"]
    total_weight = u["weight"]

    # 循环遍历后续物品，尽量填满背包，直到没有更多的物品或背包已满
    while j < n and total_weight + weights[j] <= W:
        total_weight += weights[j]
        bound_value += values[j]
        j += 1

    # 如果还有物品，并且背包仍有空间，用下一个物品的单位价值来估算上界
    if j < n:
        bound_value += (W - total_weight) * values[j] / weights[j]

    return bound_value


def knapsack_branch_and_bound(weights, values, W):
    n = len(weights)

    # 生成物品的索引列表并根据每个物品的价值-重量比进行排序
    items = list(range(n))
    items.sort(key=lambda x: values[x] / weights[x], reverse=True)

    # 初始化队列
    Q = Queue()
    # u 是当前正在考虑的节点，v 是其父节点
    u = {"level": -1, "value": 0, "weight": 0, "items": []}
    v = {"level": -1, "value": 0, "weight": 0, "items": []}
    Q.put(v)

    # 初始的最大价值为0
    max_value = 0
    best_combination = []

    # 使用 BFS 遍历每个可能的物品组合
    while not Q.empty():
        v = Q.get()

        # 如果所有的物品都已被考虑，则跳过此迭代
        if v["level"] == n - 1:
            continue

        # 下一个要考虑的物品的索引
        item_idx = items[v["level"] + 1]

        # 先考虑包含此物品的情况
        u = v.copy()
        u["level"] += 1
        u["weight"] += weights[item_idx]
        u["value"] += values[item_idx]
        u["items"] = v["items"] + [item_idx]

        # 如果此组合的重量在允许范围内且价值大于当前最大价值，则更新最大价值
        if u["weight"] <= W and u["value"] > max_value:
            max_value = u["value"]
            best_combination = u["items"]

        # 如果此路径的上界大于当前最大价值，则将其加入队列以进一步探索
        if bound(u, n, W, weights, values) > max_value:
            Q.put(u)

        # 再考虑不包含此物品的情况
        u = v.copy()
        u["level"] += 1

        # 如果此路径的上界大于当前最大价值，则将其加入队列以进一步探索
        if bound(u, n, W, weights, values) > max_value:
            Q.put(u)

    # 返回最佳组合和其对应的价值
    return sorted(best_combination), max_value


class KnapsackGenetic:
    def __init__(self, pop_size=100, mutation_rate=0.25, crossover_rate=0.7, max_gen=200):
        # 遗传算法的主要参数
        self.POP_SIZE = pop_size       # 种群大小
        self.MUTATION_RATE = mutation_rate  # 染色体突变率
        self.CROSSOVER_RATE = crossover_rate  # 交叉率
        self.MAX_GEN = max_gen        # 最大代数
        self.GENES = [0, 1]           # 基因集，代表物品是否被选中

    class Individual:
        def __init__(self, chromosome, weights, values, W):
            self.chromosome = chromosome  # 个体的染色体，代表一个解
            self.fitness = self.calculate_fitness(
                weights, values, W)  # 个体的适应度值

        # 交叉两个染色体产生新的染色体
        def mate(self, partner):
            child_chromosome = []
            for gp, cp in zip(self.chromosome, partner.chromosome):
                prob = random.random()  # 随机生成[0,1)之间的数
                # 根据概率决定继承哪一个父代的基因
                if prob < 0.45:
                    child_chromosome.append(gp)
                elif prob < 0.9:
                    child_chromosome.append(cp)
                else:
                    child_chromosome.append(random.choice([0, 1]))
            return child_chromosome

        # 计算个体的适应度值
        def calculate_fitness(self, weights, values, W):
            weight, value = 0, 0
            for i, gene in enumerate(self.chromosome):
                weight += weights[i] * gene  # 总重量
                value += values[i] * gene    # 总价值
            if weight > W:
                return -1  # 若超过背包容量则返回-1，表示此解不可行
            return value

    def solve(self, weights, values, W):
        # 生成随机染色体
        def random_chromosome():
            return [random.choice(self.GENES) for _ in range(len(weights))]

        # 进化种群以产生下一代
        def evolve_population(population):
            new_population = []
            # 选择适应度最高的10%的个体直接进入下一代
            top_10_percent = int(0.1 * self.POP_SIZE)
            population.sort(key=lambda x: x.fitness, reverse=True)
            new_population.extend(population[:top_10_percent])

            while len(new_population) < self.POP_SIZE:
                # 从前50%的个体中选择父代
                parent1 = random.choice(population[:50])
                parent2 = random.choice(population[:50])

                # 根据交叉率决定是否进行交叉
                if random.random() < self.CROSSOVER_RATE:
                    child_chromosome = parent1.mate(parent2)
                    child = self.Individual(
                        child_chromosome, weights, values, W)

                    # 根据突变率决定是否进行突变
                    if random.random() < self.MUTATION_RATE:
                        mutate_position = random.randint(0, len(weights) - 1)
                        child.chromosome[mutate_position] = 1 - \
                            child.chromosome[mutate_position]
                        child.fitness = child.calculate_fitness(
                            weights, values, W)

                    new_population.append(child)

            return new_population

        # 初始化种群
        population = [self.Individual(
            random_chromosome(), weights, values, W) for _ in range(self.POP_SIZE)]
        for _ in range(self.MAX_GEN):
            population = evolve_population(population)

        # 提取最佳解
        best_fitness = max([ind.fitness for ind in population])
        best_solutions = [
            ind for ind in population if ind.fitness == best_fitness]

        # 根据选择的物品的索引排序并删除重复的解
        unique_solutions = set()
        for sol in best_solutions:
            items_selected = tuple(
                sorted([i for i, gene in enumerate(sol.chromosome) if gene == 1]))
            unique_solutions.add(items_selected)

        return [(list(sol), best_fitness) for sol in unique_solutions]


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
                items_selected_idx, max_value = knapsack_branch_and_bound(
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
        run()
