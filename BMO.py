import numpy as np
import matplotlib.pyplot as plt

class BMO:
    def __init__(self, cost_fn, n_vars, population_size=100, max_generations=100, ub=0.99, lb=0, verbose = False, guess = 'random'):
        self.population_size = population_size
        self.n_vars = n_vars
        self.ub = ub
        self.lb = lb
        self.cost_fn = cost_fn
        self.society, self.fitness = self.initialize_population(guess)
        self.max_generations = max_generations
        self.mcf = 0.9
        self.verbose = verbose

        self.mcfp0 = 0.1
        self.mcfpinf = 0.9
        
        self.mu = 1e-3
        
        self.w0 = 2
        self.winf = 0.25

        # species proportions
        self.monogamous_ = 0.5
        self.polygynous_ = 0.3
        self.promiscuous_ = 0.1
        self.polyandrous_ = 0.05
        self.parthenogenetic_ = 0.05

        self.best_ = None
        self.best_fitness_ = None

        self.log = np.zeros((max_generations,))

    def initialize_population(self,guess):
        population = None
        if guess == 'random':
            population = self.lb + (self.ub - self.lb)*np.random.random(size = (self.population_size, self.n_vars))
        else:
            population = guess * np.ones(shape = (self.population_size, self.n_vars))

        fitness = np.zeros((self.population_size,))
        for i in range(self.population_size):
            fitness[i] = self.cost_fn(population[i,:])
        
        return population, fitness
    
    def plot_graph(self):
        plt.grid()
        plt.plot(self.log)
        plt.show()
    
    def run(self):
        for step in range(1,1+self.max_generations):
            # sort birds
            order = self.fitness.argsort()
            self.society = self.society[order,:]
            self.fitness = self.fitness[order]

            # disturbance procedure
            if step*10>self.max_generations:
                self.society[0][self.society[0]<=0.015] = 0
                self.fitness[0] = self.cost_fn(self.society[0,:])

                order = self.fitness.argsort()
                self.society = self.society[order,:]
                self.fitness = self.fitness[order]

            self.best_, self.best_fitness_ = (self.society[0,:], self.fitness[0])
            if self.verbose:
                print("{0} : {1}".format(step, self.best_fitness_))
            
            self.log[step-1] = self.best_fitness_
            
            # males: monogamous, polygynous and promiscuous
            # females: parthenogenetic and polyandrous
            n1 = int(self.population_size * self.monogamous_)
            n2 = int(self.population_size * self.polygynous_)
            n4 = int(self.population_size * self.polyandrous_)
            n5 = int(self.population_size * self.parthenogenetic_)
            polyandrous, parthenogetic, monogamous, polygynous, promiscuous = np.split(self.society, [n4,n4+n5, n4+n5+n1, n4+n5+n1+n2])
            polyandrous_fit, parthenogetic_fit, monogamous_fit, polygynous_fit, promiscuous_fit = np.split(self.fitness, [n4,n4+n5, n4+n5+n1, n4+n5+n1+n2])

            
            # remove the worst birds and generate promiscuous birds based on the chaotic sequence
            # promiscuous = self.lb + (self.ub - self.lb)*np.random.random(size = promiscuous.shape)

            # compute objective function of the promiscuous birds
            for i in range(len(promiscuous_fit)):
                promiscuous_fit[i] = self.cost_fn(promiscuous[i,:])

            w = self.w0 + (self.winf - self.w0) * step / self.max_generations

            # produce broods for monogamous birds
            for i in range(len(monogamous)):
                xb  = monogamous[i,:] + w * np.random.random(self.n_vars) * (polyandrous[np.random.randint(0,len(polyandrous)),:] - monogamous[i,:])
                if np.random.random() > self.mcf:
                    xb[np.random.randint(0,self.n_vars)] = self.lb - np.random.random() *(self.lb - self.ub)
                
                xb[xb>self.ub] = self.ub
                xb[xb<self.lb] = self.lb

                cost_val = self.cost_fn(xb)
                if cost_val < monogamous_fit[i]:
                    monogamous_fit[i] = cost_val
                    monogamous[i,:] = xb

            # produce broods for polygynous birds
            for i in range(len(polygynous)):
                x = polygynous[i,:]
                sumation_term = np.random.random(self.n_vars) * (polyandrous[np.random.randint(0,len(polyandrous)),:]-x)
                for i in range (2):
                    sumation_term += np.random.random(self.n_vars) * (polyandrous[np.random.randint(0,len(polyandrous)),:]-x)

                xb  = polygynous[i,:] + w * sumation_term
                xb[xb>self.ub] = self.ub
                xb[xb<self.lb] = self.lb
                if np.random.random() > self.mcf:
                    xb[np.random.randint(0,self.n_vars)] = self.lb - np.random.random() *(self.lb - self.ub)
                
                cost_val = self.cost_fn(xb)
                if cost_val < polygynous_fit[i]:
                    polygynous_fit[i] = cost_val
                    polygynous[i,:] = xb
                
            # produce broods for polyandrous birds
            for i in range(len(polyandrous)):
                x = polyandrous[i,:]
                sumation_term = np.random.random(self.n_vars) * (monogamous[np.random.randint(0,len(monogamous)),:]-x)
                for i in range (2):
                    sumation_term += np.random.random(self.n_vars) * (monogamous[np.random.randint(0,len(monogamous)),:]-x)

                xb  = polyandrous[i,:] + w * sumation_term
                xb[xb>self.ub] = self.ub
                xb[xb<self.lb] = self.lb
                if np.random.random() > self.mcf:
                    xb[np.random.randint(0,self.n_vars)] = self.lb - np.random.random() *(self.lb - self.ub)

                cost_val = self.cost_fn(xb)
                if cost_val < polyandrous_fit[i]:
                    polyandrous_fit[i] = cost_val
                    polyandrous[i,:] = xb

            # produce broods for promiscuous birds
            for i in range(len(promiscuous)):
                x = promiscuous[i,:]
                xb  = promiscuous[i,:] + w * np.random.random(self.n_vars) * (polyandrous[np.random.randint(0,len(polyandrous)),:]-x)
                xb[xb>self.ub] = self.ub
                xb[xb<self.lb] = self.lb
                if np.random.random() > self.mcf:
                    xb[np.random.randint(0,self.n_vars)] = self.lb - np.random.random() *(self.lb - self.ub)

                cost_val = self.cost_fn(xb)
                if cost_val < promiscuous_fit[i]:
                    promiscuous_fit[i] = cost_val
                    promiscuous[i,:] = xb

            # produce broods for parthenogenetic birds
            mcfp = self.mcfp0 + (self.mcfpinf - self.mcfp0) * step / self.max_generations
            for i in range(len(parthenogetic)):
                if np.random.random() > mcfp :
                    xb = parthenogetic[i,:]+ self.mu * (np.random.random(self.n_vars)-0.5) * parthenogetic[i,:]
                    xb[xb>self.ub] = self.ub
                    xb[xb<self.lb] = self.lb
                    cost_val = self.cost_fn(xb)
                    if cost_val < parthenogetic_fit[i]:
                        parthenogetic_fit[i] = cost_val
                        parthenogetic[i,:] = xb

            # replacement stage
            # update parameters

            self.society = np.concatenate((polyandrous, parthenogetic, monogamous, polygynous, promiscuous),axis = 0)
            self.fitness = np.concatenate((polyandrous_fit, parthenogetic_fit, monogamous_fit, polygynous_fit, promiscuous_fit))



if __name__ == '__main__':
    def func(x):
        return np.sum(np.abs(x+0.5)**2)
    test = BMO(cost_fn=func,n_vars=10,max_generations=100, ub = 10, lb = -10)
    test.run()
    test.plot_graph()

    print(test.best_fitness_,test.best_)