from queue import Queue


class BranchAndBoundKnapsack:

    """
    这是使用分支界限法求解0-1背包问题的实现。分支界限法是一个在搜索空间树中使用广度优先搜索（BFS）的方法
    在分支界限法中，每一次决策都生成两个子问题：一个是包含当前物品，另一个是不包含。
    每一个子问题都对应着一个节点。
    使用bound函数来计算一个节点的上界，即通过选择剩余的物品来达到的最大可能价值。
    使用队列Q来保存还未被探索的节点。
    在循环中，不断从Q中取出节点并根据该节点生成子节点。
    用当前的最大价值max_value与新生成的节点的上界对比，如果上界大于max_value，则将该节点加入Q。
    最后，返回具有最大价值的物品组合。
    这种方法通过计算每一个节点的上界来减少搜索空间，从而加速求解过程。但是，由于广度优先搜索可能会生成大量的节点，所以它可能在物品数目很大的情况下变得不够高效。
    """

    # 定义节点类来保存每一个考虑物品的状态
    class Node:
        def __init__(self, level, weight, profit, bound):
            self.level = level
            self.weight = weight
            self.profit = profit
            self.bound = bound

    # 计算上界函数
    @staticmethod
    def bound(u, n, W, weights, values):
        # 如果当前重量已经超过背包的容量，则此路径不可行，返回0
        if u["weight"] >= W:
            return 0

        j = u["level"] + 1
        bound_value = u["value"]
        total_weight = u["weight"]

        while j < n and total_weight + weights[j] <= W:
            total_weight += weights[j]
            bound_value += values[j]
            j += 1

        if j < n:
            bound_value += (W - total_weight) * values[j] / weights[j]

        return bound_value

    # 主函数
    @staticmethod
    def knapsack_branch_and_bound(weights, values, W):
        n = len(weights)
        items = list(range(n))
        items.sort(key=lambda x: values[x] / weights[x], reverse=True)
        Q = Queue()
        u = {"level": -1, "value": 0, "weight": 0, "items": []}
        v = {"level": -1, "value": 0, "weight": 0, "items": []}
        Q.put(v)
        max_value = 0
        best_combination = []

        while not Q.empty():
            v = Q.get()
            if v["level"] == n - 1:
                continue

            item_idx = items[v["level"] + 1]
            u = v.copy()
            u["level"] += 1
            u["weight"] += weights[item_idx]
            u["value"] += values[item_idx]
            u["items"] = v["items"] + [item_idx]

            if u["weight"] <= W and u["value"] > max_value:
                max_value = u["value"]
                best_combination = u["items"]

            if BranchAndBoundKnapsack.bound(u, n, W, weights, values) > max_value:
                Q.put(u)

            u = v.copy()
            u["level"] += 1

            if BranchAndBoundKnapsack.bound(u, n, W, weights, values) > max_value:
                Q.put(u)

        return sorted(best_combination), max_value
