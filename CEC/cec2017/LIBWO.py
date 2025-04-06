import random
import copy
import numpy as np
import os

class LIBWO:

    # Calculate fitness value --- model_param

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

    # Initialize the spider population and step size

    def initial(self):
        X = np.zeros([self.pop, self.dim])
        for i in range(self.pop):
            for j in range(self.dim):
                X[i, j] = random.random() * (self.ub[j] - self.lb[j]) + self.lb[j]  # Random float value in each dimension
        return X, self.lb, self.ub

    # Boundary check function
    def BorderCheck(self, x):
        return np.clip(x, self.lb, self.ub)

    # Fitness calculation function
    def CaculateFitness(self, X, fun):
        pop = X.shape[0]
        fitness = np.zeros([pop, 1])
        for i in range(pop):
            fitness[i] = self.func(X[i, :])  # Apply function to the i-th row of array X
        return fitness

    # Sort fitness
    def SortFitness(self, Fit):
        fitness = np.sort(Fit, axis=0)
        index = np.argsort(Fit, axis=0)
        return fitness, index

    # Sort position based on fitness
    def SortPosition(self, X, index):
        Xnew = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Xnew[i, :] = X[index[i], :]
        return Xnew

    # Pheromone calculation
    def getPheromone(self, fit, minfit, maxfit, pop):
        out = np.zeros([pop])
        if minfit != maxfit:
            for i in range(pop):
                out[i] = (maxfit - fit[i]) / (maxfit - minfit)
        return out

    # Binary generator (0/1)
    def getBinary(self):
        value = 0
        if np.random.random() < 0.5:
            value = 0
        else:
            value = 1
        return value

    # Black Widow Optimization
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

            # Store the best position and fitness of the current iteration
            self.now_iter_x_best = copy.copy(X[indexBest, :])  # Best position in this iteration
            self.now_iter_y_best = copy.copy(fitness[indexBest])  # Best fitness in this iteration

            # Perform quadratic interpolation to predict the next best position based on previous and current best positions and their fitness values
            # The numerator is calculated by considering the squared differences of the positions and the corresponding fitness values from different iterations
            numerator = (np.square(self.now_iter_x_best) - np.square(GbestPositon)) * self.pre_iter_y_best + (
                    np.square(GbestPositon) - np.square(self.pre_iter_x_best)) * self.now_iter_y_best + (
                                np.square(self.pre_iter_x_best) - np.square(self.now_iter_x_best)) * GbestScore

            # The denominator is calculated based on the differences in positions and their corresponding fitness values
            denominator = 2 * ((self.now_iter_x_best - GbestPositon) * self.pre_iter_y_best + (
                    GbestPositon - self.pre_iter_x_best) * self.now_iter_y_best + (
                                       self.pre_iter_x_best - self.now_iter_x_best) * GbestScore)

            # To avoid division by zero, replace any zero in the denominator with a small epsilon value (1e-6)
            denominator = np.where(denominator == 0, 1e-6, denominator)

            # Final result: Use the quadratic interpolation to calculate the predicted position (`li_Position_best`) and the corresponding fitness value (`li_Score_best`)
            self.li_Position_best = numerator / denominator

            # Check if the predicted position vector (`li_Position_best`) has the correct dimension
            if len(self.li_Position_best) != 20:
                # If the dimension is incorrect, adjust `li_Position_best` to have a size of 20
                self.li_Position_best = np.zeros(20)
                self.li_Position_best[:len(fitness[indexBest])] = fitness[indexBest]

            # Calculate the fitness score for the predicted position
            self.li_Score_best = self.func(self.li_Position_best)

            # If the predicted fitness score is better than the current best fitness, update the position and fitness
            if self.li_Score_best < self.now_iter_y_best:
                fitness[indexBest] = self.li_Score_best
                X[indexBest, :] = self.li_Position_best

            if fitness[indexBest] <= GbestScore:  # Update global best
                GbestScore = copy.copy(fitness[indexBest])  # Best fitness
                GbestPositon[0, :] = copy.copy(X[indexBest, :])  # Best position
            pheromone = self.getPheromone(fitness, fitness[indexBest], fitness[indexWorst], self.pop)  # Update pheromone
            Curve[t] = GbestScore  # Store best fitness in each iteration

            # Update previous iteration's best values and positions for next interpolation
            self.pre_iter_x_best = self.now_iter_x_best
            self.pre_iter_y_best = self.now_iter_y_best

        return GbestScore, GbestPositon, Curve
