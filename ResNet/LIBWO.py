import random
import copy
import numpy as np
import model as md
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def fit_fun(param, X):  # Fitness function, here it is model training
    # Set GPU to use as needed
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    # Get model parameters
    train_data = param['data']
    train_label = param['label']
    model = md.resnet18_model()
    # Pass the learning_rate parameter to be optimized
    res_model = model.model_create(X[-1])
    history = res_model.fit(train_data, train_label, epochs=1, batch_size=32, validation_split=0.2)
    # Get the minimum loss value, optimize when the loss is the smallest
    val_loss = min(history.history['val_loss'])
    return val_loss


class LIBWO:

    # Calculate fitness value --- model_param

    def __init__(self, model_param, bwo_param, constraint_ueq=None):
        self.model_param = model_param  # Model parameters
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

    # '''Initialize spider population''' Initialize the population step size

    def initial(self):
        X = np.zeros([self.pop, self.dim])
        for i in range(self.pop):
            for j in range(self.dim):
                X[i, j] = random.random() * (self.ub[j] - self.lb[j]) + self.lb[j]  # Each dimension value is a random floating-point number

        return X, self.lb, self.ub

    '''Boundary check function'''

    def BorderCheck(self, x):
        return np.clip(x, self.lb, self.ub)

    '''Calculate fitness function'''

    def CaculateFitness(self, X, fun):
        pop = X.shape[0]
        fitness = np.zeros([pop, 1])
        for i in range(pop):
            fitness[i] = fun(self.model_param, X[i, :])  # Apply function fun to the i-th row of array X
        return fitness

    '''Fitness sorting'''

    def SortFitness(self, Fit):
        fitness = np.sort(Fit, axis=0)
        index = np.argsort(Fit, axis=0)
        return fitness, index

    '''Sort positions based on fitness'''

    def SortPosition(self, X, index):
        Xnew = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Xnew[i, :] = X[index[i], :]
        return Xnew

    '''Pheromone calculation'''

    def getPheromone(self, fit, minfit, maxfit, pop):
        out = np.zeros([pop])
        if minfit != maxfit:
            for i in range(pop):
                out[i] = (maxfit - fit[i]) / (maxfit - minfit)
        return out

    '''0/1 generator'''

    def getBinary(self):
        value = 0
        if np.random.random() < 0.5:
            value = 0
        else:
            value = 1
        return value

    # Black widow

    def LIBWO(self):
        global r2
        X, self.lb, self.ub = self.initial()  # Initialize population
        # fitness = self.CaculateFitness(X, fit_fun(self.model_param, X))  # Calculate fitness values
        fitness = self.CaculateFitness(X, fit_fun)  # Calculate fitness values
        indexBest = np.argmin(fitness)
        indexWorst = np.argmax(fitness)
        GbestScore = copy.copy(fitness[indexBest])
        GbestPositon = np.zeros([1, self.dim])
        GbestPositon[0, :] = copy.copy(X[indexBest, :])

        self.pre_iter_x_best = copy.copy(fitness[indexBest])  # Set initial value for the best position of the previous iteration
        self.pre_iter_y_best = copy.copy(X[indexBest, :])  # Set initial value for the best fitness of the previous iteration

        Curve = np.zeros([self.MaxIter, 1])
        pheromone = self.getPheromone(fitness, fitness[indexBest], fitness[indexWorst], self.pop)  # Calculate pheromone
        Xnew = copy.deepcopy(X)
        fitNew = copy.deepcopy(fitness)
        for t in range(self.MaxIter):
            beta = -1 + 2 * np.random.random()  # -1 < beta < 1
            m = 0.4 + 0.5 * np.random.random()  # 0.4 < m < 0.9
            for i in range(self.pop):
                P = np.random.random()
                r1 = int(self.pop * np.random.random())
                if P >= 0.3:  # Spider movement position update
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
                fitNew[i] = fit_fun(self.model_param, Xnew[i, :])  #
                if fitNew[i] < fitness[i]:
                    X[i, :] = copy.copy(Xnew[i, :])
                    fitness[i] = copy.copy(fitNew[i])
            indexBest = np.argmin(fitness)
            indexWorst = np.argmax(fitness)

            self.now_iter_x_best = copy.copy(X[indexBest, :])  # Best position of this iteration
            self.now_iter_y_best = copy.copy(fitness[indexBest])  # Best fitness of this iteration

            numerator = (np.square(self.now_iter_x_best) - np.square(GbestPositon)) * self.pre_iter_y_best + (
                    np.square(GbestPositon) - np.square(self.pre_iter_x_best)) * self.now_iter_y_best + (
                                np.square(self.pre_iter_x_best) - np.square(self.now_iter_x_best)) * GbestScore
            # Calculate the numerator
            denominator = 2 * ((self.now_iter_x_best - GbestPositon) * self.pre_iter_y_best + (
                    GbestPositon - self.pre_iter_x_best) * self.now_iter_y_best + (
                                       self.pre_iter_x_best - self.now_iter_x_best) * GbestScore)
            # Handle zero elements in the denominator
            denominator = np.where(denominator == 0, 1e-6, denominator)

            # Calculate the final result, Lagrange interpolation predicted position and its corresponding fitness value
            self.li_Position_best = numerator / denominator
            self.li_Position_best = copy.copy(fitness[indexBest])
            self.li_Score_best = fit_fun(self.model_param, self.li_Position_best)

            if self.li_Score_best < self.now_iter_y_best:
                fitness[indexBest] = self.li_Score_best
                X[indexBest, :] = self.li_Position_best

            if fitness[indexBest] <= GbestScore:  # Update global best
                GbestScore = copy.copy(fitness[indexBest])  # Best fitness
                GbestPositon[0, :] = copy.copy(X[indexBest, :])  # Best position
            pheromone = self.getPheromone(fitness, fitness[indexBest], fitness[indexWorst], self.pop)  # Calculate pheromone
            Curve[t] = GbestScore  # Current best fitness, Curve saves the best fitness from each iteration

            # Update the best values and positions from the previous iteration to the best values and positions from this iteration
            self.pre_iter_x_best = self.now_iter_x_best
            self.pre_iter_y_best = self.now_iter_y_best

        return GbestScore, GbestPositon
