import random
import copy
import numpy as np
import os

class LIBWO:

    # Initialization function
    # Sets the objective function for the optimization problem
    # and initializes key BWO parameters such as population size,
    # number of iterations, dimension, and variable bounds.
    def __init__(self, func, bwo_param, constraint_ueq=None):
        self.func = func  # Model function
        # Initialize parameters
        self.pop = bwo_param['pop']
        self.MaxIter = bwo_param['MaxIter']
        self.dim = bwo_param['dim']
        self.lb = bwo_param['lb']
        self.ub = bwo_param['ub']
        self.now_iter_x_best = 0
        self.now_iter_y_best = 0
        self.pre_iter_x_best = 0
        self.pre_iter_y_best = 0
        self.li_Score_best = 0
        self.li_Position_best = 0

    # Initialize the positions of individuals in the population
    # Randomly generate initial solutions within the bounds
    # for each dimension of each individual.
    def initial(self):
        X = np.zeros([self.pop, self.dim])
        for i in range(self.pop):
            for j in range(self.dim):
                X[i, j] = random.random() * (self.ub[j] - self.lb[j]) + self.lb[j]  # Each dimension is a random float
        return X, self.lb, self.ub

    # Boundary check function
    # Ensures that all individuals stay within the specified bounds
    # after any position updates.
    def BorderCheck(self, X):
        for i in range(self.pop):
            for j in range(self.dim):
                if X[i, j] > self.ub[j]:
                    X[i, j] = self.ub[j]
                elif X[i, j] < self.lb[j]:
                    X[i, j] = self.lb[j]
        return X

    # Fitness evaluation function
    # Applies the objective function to each individual
    # and returns the corresponding fitness values.
    def CaculateFitness(self, X, fun):
        pop = X.shape[0]
        fitness = np.zeros([pop, 1])
        for i in range(pop):
            fitness[i] = self.func(X[i, :])  # Apply function to each row of X
        return fitness

    # Fitness sorting function
    # Sorts the fitness values in ascending order
    # and returns the sorted values and corresponding indices.
    def SortFitness(self, Fit):
        fitness = np.sort(Fit, axis=0)
        index = np.argsort(Fit, axis=0)
        return fitness, index

    # Position sorting function
    # Reorders the population positions based on the sorted fitness indices.
    def SortPosition(self, X, index):
        Xnew = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Xnew[i, :] = X[index[i], :]
        return Xnew

    # Pheromone calculation function
    # Calculates the pheromone intensity for each individual
    # based on the difference between its fitness and the best/worst fitness values.
    def getPheromone(self, fit, minfit, maxfit, pop):
        out = np.zeros([pop])
        if minfit != maxfit:
            for i in range(pop):
                out[i] = (maxfit - fit[i]) / (maxfit - minfit)
        return out

    # Binary 0/1 generator
    # Simulates probabilistic events (e.g., crossover or mutation)
    # by returning either 0 or 1 with equal probability.
    def getBinary(self):
        value = 0
        if np.random.random() < 0.5:
            value = 0
        else:
            value = 1
        return value

     # Lagrange Interpolation-enhanced Black Widow Optimization Algorithm (LIBWO)
    #
    # This function enhances the original BWO algorithm by introducing a Lagrange interpolation mechanism
    # to further improve search capability and convergence performance in high-dimensional optimization problems.
    #
    # The main steps of the algorithm are as follows:
    # 1. Initialize the population and individual positions, calculate fitness, and record the initial global best;
    # 2. In each iteration:
    #    - Based on probability P, choose one of two spider movement update strategies;
    #    - If the individual's pheromone level is low, trigger a "mating" mechanism to generate a new position;
    #    - Perform boundary check on the new position and recalculate its fitness;
    #    - If the new position is better, replace the old one;
    # 3. At the end of each iteration:
    #    - Use the best positions and fitness values from the current and previous generations
    #      to perform Lagrange interpolation and predict a potentially better solution;
    #    - If the interpolated position yields a better fitness, update it as the new global best;
    #    - Update pheromone levels and record the best fitness of the current generation;
    #    - Update the previous generation's best values for use in the next interpolation step.
    #
    # Returns:
    # - GbestScore: the best fitness value found by the algorithm
    # - GbestPosition: the solution vector corresponding to the best fitness
    # - Curve: the best fitness value at each generation (can be used for plotting)
    def LIBWO(self):
        global r2
        X, self.lb, self.ub = self.initial()  # Initialize population
        fitness = self.CaculateFitness(X, self.func)  # Calculate fitness value
        indexBest = np.argmin(fitness)
        indexWorst = np.argmax(fitness)
        GbestScore = copy.copy(fitness[indexBest])
        GbestPositon = np.zeros([1, self.dim])
        GbestPositon[0, :] = copy.copy(X[indexBest, :])

        self.pre_iter_x_best = copy.copy(fitness[indexBest])  # Set initial value of previous iteration's best position
        self.pre_iter_y_best = copy.copy(X[indexBest, :])  # Set initial value of previous iteration's best fitness

        Curve = np.zeros([self.MaxIter, 1])
        pheromone = self.getPheromone(fitness, fitness[indexBest], fitness[indexWorst], self.pop)  # Compute pheromone
        Xnew = copy.deepcopy(X)
        fitNew = copy.deepcopy(fitness)
        for t in range(self.MaxIter):
            beta = -1 + 2 * np.random.random()  # -1 < beta < 1
            m = 0.4 + 0.5 * np.random.random()  # 0.4 < m < 0.9
            for i in range(self.pop):
                P = np.random.random()
                r1 = int(self.pop * np.random.random())
                if P >= 0.3:  # Spider movement update
                    Xnew[i, :] = GbestPositon - np.cos(2 * np.pi * beta) * X[i, :]
                else:
                    Xnew[i, :] = GbestPositon - m * X[r1, :]
                if pheromone[i] <= 0.3:  # Replace black widow position
                    band = 1
                    while band:
                        r1 = int(self.pop * np.random.random())
                        r2 = int(self.pop * np.random.random())
                        if r1 != r2:
                            band = 0
                    Xnew[i, :] = GbestPositon + (X[r1, :] - (-1) ** self.getBinary() * X[r2, :]) / 2
                for j in range(self.dim):
                    if Xnew[i, j] > self.ub[j]:
                        Xnew[i, j] = self.ub[j]
                    if Xnew[i, j] < self.lb[j]:
                        Xnew[i, j] = self.lb[j]
                fitNew[i] = self.func(Xnew[i, :])
                if fitNew[i] < fitness[i]:
                    X[i, :] = copy.copy(Xnew[i, :])
                    fitness[i] = copy.copy(fitNew[i])
            indexBest = np.argmin(fitness)
            indexWorst = np.argmax(fitness)

            # At the end of each iteration, perform Lagrange interpolation to predict a potentially better solution.
            # The steps are as follows:
            # 1. Record the best position (now_iter_x_best) and best fitness (now_iter_y_best) of the current iteration;
            # 2. Calculate the numerator and denominator of the Lagrange interpolation;
            #    The numerator is calculated using the squared differences of the current and previous best positions and their corresponding fitness values;
            # 3. To avoid division by zero, a small constant (1e-6) is used for smoothing in the denominator;
            # 4. Calculate the interpolated best position (li_Position_best);
            # 5. If the interpolated position does not have the correct dimension (not 20), adjust it to the correct dimension
            #    and fill it with the current best position;
            # 6. Calculate the fitness of the interpolated position (li_Score_best);
            # 7. If the fitness of the interpolated position is better than the current best, update the best position and fitness;
            # 8. If the current best fitness is better, update the global best fitness.

            self.now_iter_x_best = copy.copy(X[indexBest, :])
            self.now_iter_y_best = copy.copy(fitness[indexBest])

            numerator = (np.square(self.now_iter_x_best) - np.square(GbestPositon)) * self.pre_iter_y_best + (
                    np.square(GbestPositon) - np.square(self.pre_iter_x_best)) * self.now_iter_y_best + (
                                np.square(self.pre_iter_x_best) - np.square(self.now_iter_x_best)) * GbestScore

            denominator = 2 * ((self.now_iter_x_best - GbestPositon) * self.pre_iter_y_best + (
                    GbestPositon - self.pre_iter_x_best) * self.now_iter_y_best + (
                                       self.pre_iter_x_best - self.now_iter_x_best) * GbestScore)

            denominator = np.where(denominator == 0, 1e-6, denominator)
            
            self.li_Position_best = numerator / denominator

            if len(self.li_Position_best) != 20:
                # If the dimension is incorrect, adjust `li_Position_best` to have a size of 20
                self.li_Position_best = np.zeros(20)
                self.li_Position_best[:len(fitness[indexBest])] = fitness[indexBest]

            self.li_Score_best = self.func(self.li_Position_best)

            if self.li_Score_best < self.now_iter_y_best:
                fitness[indexBest] = self.li_Score_best
                X[indexBest, :] = self.li_Position_best

            if fitness[indexBest] <= GbestScore:  # Update global best
                GbestScore = copy.copy(fitness[indexBest])  # Best fitness
                GbestPositon[0, :] = copy.copy(X[indexBest, :])  # Best position
            pheromone = self.getPheromone(fitness, fitness[indexBest], fitness[indexWorst], self.pop)
            Curve[t] = GbestScore  # Store best fitness in each iteration

            self.pre_iter_x_best = self.now_iter_x_best
            self.pre_iter_y_best = self.now_iter_y_best

        return GbestScore, GbestPositon, Curve
