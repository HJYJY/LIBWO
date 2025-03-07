import random
import copy
import numpy as np
from ResNet import model as md
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def fit_fun(param, X):  # 适应函数,此处为模型训练
    # 设置GPU按需使用
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    # 获取模型参数
    train_data = param['data']
    train_label = param['label']
    model = md.resnet18_model()
    # 传入待优化参数learning_rate
    res_model = model.model_create(X[-1])
    history = res_model.fit(train_data, train_label, epochs=5, batch_size=128, validation_split=0.2)
    # 获取最小的loss值,优化为loss最小时learning_rate
    val_loss = min(history.history['val_loss'])
    return val_loss


class BWO:

    # 计算适度值 --- model_param

    def __init__(self, model_param, bwo_param, constraint_ueq=None):
        self.model_param = model_param  # 模型参数
        # 初始化参数
        self.pop = bwo_param['pop']
        self.MaxIter = bwo_param['MaxIter']
        self.dim = bwo_param['dim']
        self.lb = bwo_param['lb']
        self.ub = bwo_param['ub']

    # '''初始化蜘蛛种群'''  初始化种群 步长

    def initial(self):
        X = np.zeros([self.pop, self.dim])
        for i in range(self.pop):
            for j in range(self.dim):
                X[i, j] = random.random() * (self.ub[j] - self.lb[j]) + self.lb[j]  # 每个维度的值是一个随机的浮点数

        return X, self.lb, self.ub

    '''边界检查函数'''

    def BorderCheck(self, X):
        for i in range(self.pop):
            for j in range(self.dim):
                if X[i, j] > self.ub[j]:
                    X[i, j] = self.ub[j]
                elif X[i, j] < self.lb[j]:
                    X[i, j] = self.lb[j]
        return X

    '''计算适应度函数'''

    def CaculateFitness(self, X, fun):
        pop = X.shape[0]
        fitness = np.zeros([pop, 1])
        for i in range(pop):
            fitness[i] = fun(self.model_param,X[i, :])         # 函数fun应用于数组X的第i行
        return fitness

    '''适应度排序'''

    def SortFitness(self, Fit):
        fitness = np.sort(Fit, axis=0)
        index = np.argsort(Fit, axis=0)
        return fitness, index

    '''根据适应度对位置进行排序'''

    def SortPosition(self, X, index):
        Xnew = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Xnew[i, :] = X[index[i], :]
        return Xnew

    '''信息素计算'''

    def getPheromone(self, fit, minfit, maxfit, pop):
        out = np.zeros([pop])
        if minfit != maxfit:
            for i in range(pop):
                out[i] = (maxfit - fit[i]) / (maxfit - minfit)
        return out

    '''0/1 生成器'''

    def getBinary(self):
        value = 0
        if np.random.random() < 0.5:
            value = 0
        else:
            value = 1
        return value

    # 黑寡妇

    def BWO(self):
        global r2
        X, self.lb, self.ub = self.initial()  # 初始化种群
        fitness = self.CaculateFitness(X, fit_fun)  # 计算适应度值
        indexBest = np.argmin(fitness)
        indexWorst = np.argmax(fitness)
        GbestScore = copy.copy(fitness[indexBest])
        GbestPositon = np.zeros([1, self.dim])
        GbestPositon[0, :] = copy.copy(X[indexBest, :])
        Curve = np.zeros([self.MaxIter, 1])
        pheromone = self.getPheromone(fitness, fitness[indexBest], fitness[indexWorst], self.pop)  # 计算信息素
        Xnew = copy.deepcopy(X)
        fitNew = copy.deepcopy(fitness)
        for t in range(self.MaxIter):
            beta = -1 + 2 * np.random.random()  # -1<beta<1
            m = 0.4 + 0.5 * np.random.random()  # 0.4<m<0.9
            for i in range(self.pop):
                P = np.random.random()
                r1 = int(self.pop * np.random.random())
                if P >= 0.3:  # 蜘蛛运动位置更新
                    Xnew[i, :] = GbestPositon - np.cos(2 * np.pi * beta) * X[i, :]
                else:
                    Xnew[i, :] = GbestPositon - m * X[r1, :]
                if pheromone[i] <= 0.3:  # 替换黑寡妇位置
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
                fitNew[i] = fit_fun(self.model_param, Xnew[i, :])               #
                if fitNew[i] < fitness[i]:
                    X[i, :] = copy.copy(Xnew[i, :])
                    fitness[i] = copy.copy(fitNew[i])
            indexBest = np.argmin(fitness)
            indexWorst = np.argmax(fitness)
            if fitness[indexBest] <= GbestScore:  # 更新全局最优
                GbestScore = copy.copy(fitness[indexBest])  # 最佳适应度
                GbestPositon[0, :] = copy.copy(X[indexBest, :])  # 最佳位置
            pheromone = self.getPheromone(fitness, fitness[indexBest], fitness[indexWorst], self.pop)  # 计算信息素
            Curve[t] = GbestScore  # 当前的最佳适应度  Curve保存每一次循环的最佳适应度

        return GbestScore, GbestPositon
