def knapsack_dp(weights, values, W):
    """
    该算法使用一个二维数组dp来存储解决方案，其中dp[i][w]代表在给定的i个物品和总重量为w的限制下可以得到的最大价值。
    通过比较不包含当前物品的最大价值和包含当前物品的最大价值（即当前物品的价值加上剩余重量的最大价值）来决定是否选择当前物品。
    使用回溯法从动态规划表中提取出被选择的物品。
    """
    # 获取物品数量
    n = len(weights)

    # 初始化动态规划表格，dp[i][w]表示前i个物品在总重量为w时能得到的最大价值
    dp = [[0 for x in range(W + 1)] for y in range(n + 1)]

    # 开始填充表格
    for i in range(n + 1):  # 对于所有物品
        for w in range(W + 1):  # 对于所有重量
            # 如果考虑的物品数量为0或考虑的总重量为0，价值为0
            if i == 0 or w == 0:
                dp[i][w] = 0
            # 如果第i个物品的重量小于等于考虑的总重量w
            elif weights[i-1] <= w:
                # 选择该物品并考虑剩余的重量，或者不选择该物品
                # 两者中的最大价值会被存储
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            # 如果第i个物品的重量大于考虑的总重量w，不能选择该物品
            else:
                dp[i][w] = dp[i-1][w]

    # 从dp表中回溯，找出被选择的物品
    w = W  # 起始总重量为W
    items_selected = []  # 存储被选中的物品的索引
    for i in range(n, 0, -1):  # 从最后一个物品开始考虑
        # 如果在包含第i个物品和不包含第i个物品的情况下，价值是不同的，
        # 那么第i个物品必定被选中
        if dp[i][w] != dp[i-1][w]:
            items_selected.append(i-1)  # 将该物品加入被选中物品列表
            w -= weights[i-1]  # 减去该物品的重量，并继续回溯

    # 返回被选择的物品的索引和最大价值
    return items_selected, dp[n][W]