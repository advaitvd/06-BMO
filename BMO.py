import numpy as np

class BMO:
    def __init__(self, cost_fn, n_vars, population_size=100, max_generations=100, ub=1, lb=0):
        self.population_size = population_size
        self.n_vars = n_vars
        self.ub = ub
        self.lb = lb
        self.cost_fn = cost_fn
        self.society, self.fitness = self.initialize_population()
        self.max_generations = max_generations

    def initialize_population(self):
        population = self.lb + (self.ub - self.lb)*np.random.random(size = (self.population_size, self.n_vars))
        fitness = np.zeros((self.population_size,))
        for i in range(self.population_size):
            fitness[i] = self.cost_fn(population[i,:])
        
        return population, fitness



if __name__ == '__main__':
    def func(x):
        return np.sum(x**2)
    test = BMO(cost_fn=func,n_vars=5)

    print(test.society,test.fitness)