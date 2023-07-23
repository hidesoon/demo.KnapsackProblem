def knapsack_backtrack(weights, values, W):
    """
    - 回溯法是一种通过枚举所有可能的组合来寻找问题解的方法。在这个函数中，对于每一个物品，都有两种可能：选或者不选。因此，所有可能的组合就是所有物品的选或不选的所有可能。
    - 对于每一种组合，都计算其总重量和总价值。如果总重量超过背包的容量，或者已经考虑了所有的物品，就结束这个组合的考虑。否则，对于当前的物品，有两种可能：选或者不选，因此有两次递归调用。
    - 每次更新最大价值时，都需要记录下产生这个最大价值的物品组合。
    - 最后，返回最大价值和对应的物品组合。
    """
    n = len(weights)  # 获取物品的数量
    best_value = 0  # 用于保存最大价值
    best_combination = []  # 用于保存产生最大价值的物品组合

    # 定义递归函数，用于执行回溯操作
    def backtrack(i, current_weight, current_value, current_combination):
        nonlocal best_value, best_combination  # 让函数内部可以修改外部的变量
        # 递归的终止条件：已经考虑了所有物品，或者当前总重量已经超过背包的容量
        if i == n or current_weight > W:
            # 如果当前价值大于已知的最大价值，则更新最大价值和对应的物品组合
            if current_value > best_value:
                best_value = current_value
                best_combination = current_combination.copy()  # 注意这里需要拷贝列表，而不是直接赋值
            return
        # 递归的第一部分：不选第i个物品，然后考虑下一个物品
        backtrack(i+1, current_weight, current_value, current_combination)
        # 递归的第二部分：选第i个物品，然后考虑下一个物品
        # 注意这里需要检查选第i个物品之后总重量是否超过背包的容量
        if current_weight + weights[i] <= W:
            current_combination.append(i)  # 将第i个物品添加到当前组合中
            backtrack(i+1, current_weight + weights[i], current_value + values[i], current_combination)
            current_combination.pop()  # 将第i个物品从当前组合中移除，以便于回溯

    # 从第0个物品开始执行回溯
    backtrack(0, 0, 0, [])
    # 返回最大价值和对应的物品组合
    return best_combination, best_value