def knapsack_greedy(weights, values, W):
    """
    贪婪算法是一种在每一步都选择局部最优解的方法，期望通过这种方式获得全局最优解。在这个背包问题的版本中，局部最优解是选择当前价值最高且重量最轻的物品。
    为了实现这种选择，我们首先计算每个物品的价值/重量比率，并按这个比率降序排序。
    然后，从价值最高且重量最轻的物品开始，尝试加入背包，直到背包装满或者所有物品都被考虑过。
    最后，返回已选物品的索引和总价值。
    需要注意的是，虽然这种贪婪方法在某些情况下可以获得最优解，但它并不总是能得到0-1背包问题的最优解。
    """
    n = len(weights)  # 获取物品的数量

    # 计算每个物品的价值到重量的比率
    # 结果是一个元组列表，其中每个元组的第一个元素是物品的索引，第二个元素是该物品的价值/重量比率
    ratios = [(i, values[i] / weights[i]) for i in range(n)]
    
    # 按价值/重量比率降序排序
    # 这样，价值最高且重量最轻的物品将首先被考虑
    ratios.sort(key=lambda x: x[1], reverse=True)

    total_weight = 0  # 记录已选物品的总重量
    total_value = 0   # 记录已选物品的总价值
    items_selected = []  # 记录已选物品的索引

    # 从价值最高且重量最轻的物品开始，尝试加入背包
    for i, _ in ratios:
        # 如果加入这个物品后总重量不超过背包的容量
        if total_weight + weights[i] <= W:
            total_weight += weights[i]  # 更新总重量
            total_value += values[i]   # 更新总价值
            items_selected.append(i)  # 记录这个物品的索引

    # 返回已选物品的索引和总价值
    return items_selected, total_value