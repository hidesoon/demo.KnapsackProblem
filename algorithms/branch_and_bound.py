from queue import Queue  # 导入队列模块，用于存放待处理的节点

# 定义分支和界限背包类
class BranchAndBoundKnapsack:
    # 定义一个内部类Node，用于表示每一个决策节点
    class Node:
        # 初始化函数，用于设置节点的层级、重量、利润和上界
        def __init__(self, level, weight, profit, bound):
            self.level = level  # 节点的层级
            self.weight = weight  # 节点的重量
            self.profit = profit  # 节点的利润
            self.bound = bound  # 节点的上界

    # 静态方法，用于计算节点的上界
    @staticmethod
    def bound(u, n, W, weights, values):
        # 如果节点的重量超出背包容量，上界为0
        if u["weight"] >= W:
            return 0

        # 初始化j和总重量以及上界值
        j = u["level"] + 1
        bound_value = u["value"]
        total_weight = u["weight"]

        # 如果物品可以被完全添加到背包中，则累加其价值和重量
        while j < n and total_weight + weights[j] <= W:
            total_weight += weights[j]
            bound_value += values[j]
            j += 1

        # 如果物品不能被完全添加到背包中，用它的单位价值来估算可以添加的部分的价值
        if j < n:
            bound_value += (W - total_weight) * values[j] / weights[j]

        return bound_value  # 返回计算得到的上界值

    # 主函数，解决0-1背包问题
    @staticmethod
    def knapsack_branch_and_bound(weights, values, W):
        n = len(weights)  # 获取物品数量
        # 对物品按单位价值排序
        items = list(range(n))
        items.sort(key=lambda x: values[x] / weights[x], reverse=True)
        Q = Queue()  # 初始化队列Q
        # 初始化两个节点u和v
        u = {"level": -1, "value": 0, "weight": 0, "items": []}
        v = {"level": -1, "value": 0, "weight": 0, "items": []}
        Q.put(v)  # 将v节点放入队列中
        max_value = 0  # 记录当前的最大价值
        best_combination = []  # 记录当前最佳组合

        # 当队列不为空时，继续处理节点
        while not Q.empty():
            v = Q.get()  # 从队列中取出一个节点
            # 如果v是叶子节点，则继续下一个循环
            if v["level"] == n - 1:
                continue

            # 获取下一个物品的索引
            item_idx = items[v["level"] + 1]
            u = v.copy()  # 深拷贝v到u
            u["level"] += 1
            u["weight"] += weights[item_idx]
            u["value"] += values[item_idx]
            u["items"] = v["items"] + [item_idx]  # 将当前物品添加到u的物品列表中

            # 如果u的重量在限制内且价值大于当前最大价值，则更新最大价值和最佳组合
            if u["weight"] <= W and u["value"] > max_value:
                max_value = u["value"]
                best_combination = u["items"]

            # 如果u的上界大于当前最大价值，则将u放入队列
            if BranchAndBoundKnapsack.bound(u, n, W, weights, values) > max_value:
                Q.put(u)

            # 对于不选择当前物品的情况，只更新u的层级
            u = v.copy()
            u["level"] += 1

            # 如果u的上界大于当前最大价值，则将u放入队列
            if BranchAndBoundKnapsack.bound(u, n, W, weights, values) > max_value:
                Q.put(u)

        # 返回最佳组合和最大价值
        return sorted(best_combination), max_value