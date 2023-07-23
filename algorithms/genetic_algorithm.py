
import random

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