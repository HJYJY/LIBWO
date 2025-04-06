# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import opfunu
from optimizers.LIBWO import LIBWO
from optimizers.BWO import BWO
from optimizers.DOA import DOA
from optimizers.ALA import ALA
from optimizers.SDO import SDO
from optimizers.CBSO import CBSO
from optimizers import PSEQADE
from optimizers.COVIDOA import COVIDOA
from optimizers.SASS import SASS
from optimizers.LSHADE_cnEpSin import LSHADE_cnEpSin
from optimizers.AGSK import AGSK


"""You can modify the parameters here to obtain different versions of the results."""
test_version = 'v1'
fun_name = 'F1'
year = '2022'

'''
CEC objective function
'''
def cec_fun(X):
    # Calls a specific CEC benchmark function and evaluates the fitness value of input X
    funcs = opfunu.get_functions_by_classname(func_num)
    func = funcs[0](ndim=dim)
    F = func.evaluate(X)
    return F

def get_details(func_num, dim):
    # Retrieves the lower and upper bounds of the search space for a given CEC benchmark function
    funcs = opfunu.get_functions_by_classname(func_num)
    func = funcs[0](ndim=dim)
    lb = func.lb.tolist()
    ub = func.ub.tolist()
    return lb, ub

'''
CEC function selection format: Function name + year
Examples:
cec2005: F1-F25, allowed dim = 10, 30, 50
cec2008: F1-F7, allowed 2 <= dim <= 1000
cec2010: F1-F20, allowed 100 <= dim <= 1000
cec2013: F1-F28, allowed dim = 2, 5, 10, ..., 100
cec2014: F1-F30, allowed dim = 10, 20, 30, 50, 100
cec2015: F1-F15, allowed dim = 10, 30
cec2017: F1-F29, allowed dim = 2, 10, 20, 30, 50, 100
cec2019: F1-F10, allowed dim: F1=9, F2=16, F3=18, others=10
cec2020: F1-F10, allowed dim = 2, 5, 10, 15, 20, 30, 50, 100
cec2021: F1-F10, allowed dim = 2, 10, 20
cec2022: F1-F12, allowed dim = 2, 10, 20
'''

"""
Function selection
"""
func_num = fun_name + year
objf = cec_fun
dim = 20  # Dimension (must match the selected function requirements)
lb, ub = get_details(func_num, dim)  # lb -> lower bound, ub -> upper bound

# Parameters
PopSize = 30  # Population size
Maxiter = 10  # Maximum number of iterations
initial_pos = (ub[0] - lb[0]) * np.random.random([PopSize, dim]) + lb  # Initial population

"""LIBWO"""
# Initialize and run the LIBWO (Lagrange Interpolation Black Widow Optimization) algorithm
libwo_param = {
    "dim": dim,
    "pop": PopSize,
    "MaxIter": Maxiter,
    "lb": lb,
    "ub": ub,
    "initial_pos": initial_pos,
}
LIBWO = LIBWO(objf, libwo_param)
LIBWO_best_f, LIBWO_best_x, LIBWO_convergence = LIBWO.LIBWO()
print(f"LIBWO Best Solution: {LIBWO_best_x},\nLIBWO Best Fitness: {LIBWO_best_f}")

"""BWO"""
# Initialize and run the BWO (Black Widow Optimization) algorithm
bwo_param = {
    "dim": dim,
    "pop": PopSize,
    "MaxIter": Maxiter,
    "lb": lb,
    "ub": ub,
    "initial_pos": initial_pos,
}
BWO = BWO(objf, bwo_param)
BWO_best_f, BWO_best_x, BWO_convergence = BWO.BWO()
print(f"BWO Best Solution: {BWO_best_x},\nBWO Best Fitness: {BWO_best_f}")

"""DOA"""
# Initialize and run the DOA (Dragonfly Optimization Algorithm)
doa_param = {
    "dim": dim,
    "pop": PopSize,
    "MaxIter": Maxiter,
    "lb": lb,
    "ub": ub,
    "initial_pos": initial_pos
}
DOA = DOA(objf, doa_param)
DOA_best_f, DOA_best_x, DOA_convergence = DOA.DOA()
print(f"DOA Best Solution: {DOA_best_x},\nDOA Best Fitness: {DOA_best_f}")

"""ALA"""
# Initialize and run the ALA (Artificial Life Algorithm)
ala_param = {
    "n_dim": dim,
    "size_pop": PopSize,
    "max_iter": Maxiter,
    "lb": lb,
    "ub": ub,
    "initial_pos": initial_pos
}
ALA = ALA(objf, ala_param)
ala_it, ala_x = ALA.run()
print(f"ALA Best Solution: {ala_x},\nALA Best Fitness: {ala_it}")

"""SDO"""
# Initialize and run the SDO (Spider Monkey Optimization or similar) algorithm
sdo_param = {
    "dim": dim,
    "pop": PopSize,
    "MaxIter": Maxiter,
    "lb": lb,
    "ub": ub,
    "initial_pos": initial_pos
}
SDO = SDO(objf, sdo_param)
SDO_best_f, SDO_best_x, SDO_convergence = SDO.SDO()
print(f"SDO Best Solution: {SDO_best_x},\nSDO Best Fitness: {SDO_best_f}")

"""CBSO"""
# Initialize and run the CBSO (Crow-Based Spider Optimization or similar) algorithm
cbso_param = {
    "dim": dim,
    "pop": PopSize,
    "MaxIter": Maxiter,
    "lb": lb,
    "ub": ub,
    "initial_pos": initial_pos
}
CBSO = CBSO(objf, cbso_param)
CBSO_best_f, CBSO_best_x, CBSO_convergence = CBSO.CBSO()
print(f"CBSO Best Solution: {CBSO_best_x},\nCBSO Best Fitness: {CBSO_best_f}")

"""PSEQADE"""
# Initialize and run the PSEQADE (a variant of Differential Evolution - DE)
optimal_results = PSEQADE.PSEQADE(objf, lb, ub, dim, PopSize, Maxiter, initial_pos)
PSEQADE_best_x = optimal_results.bestIndividual
PSEQADE_best_f = optimal_results.best
PSEQADE_convergence = optimal_results.convergence
print(f"PSEQADE Best Solution: {PSEQADE_best_x},\nPSEQADE Best Fitness: {PSEQADE_best_f}")

"""COVIDOA"""
# Initialize and run the COVIDOA (COVID-19 Optimization Algorithm)
covidoa_param = {
    "dim": dim,
    "pop": PopSize,
    "MaxIter": Maxiter,
    "lb": lb,
    "ub": ub,
    "initial_pos": initial_pos,
}
COVIDOA = COVIDOA(objf, covidoa_param)
COVIDOA_best_f, COVIDOA_best_x, COVIDOA_convergence = COVIDOA.COVIDOA()
print(f"COVIDOA Best Solution: {COVIDOA_best_x},\nCOVIDOA Best Fitness: {COVIDOA_best_f}")

"""SASS"""
# Initialize and run the SASS (Social Adaptive Swarm Search or similar) algorithm
sass_param = {
    "dim": dim,
    "pop": PopSize,
    "MaxIter": Maxiter,
    "lb": lb,
    "ub": ub,
    "initial_pos": initial_pos
}
SASS = SASS(objf, sass_param)
SASS_best_f, SASS_best_x, SASS_convergence = SASS.SASS()
print(f"SASS Best Solution: {SASS_best_x},\nSASS Best Fitness: {SASS_best_f}")

"""LSHADE_cnEpSin"""
# Initialize and run the LSHADE_cnEpSin (L-SHADE with constraint handling and ε-sin penalty)
LSHADE_cnEpSin_param = {
    "dim": dim,
    "pop": PopSize,
    "MaxIter": Maxiter,
    "lb": lb,
    "ub": ub,
    "initial_pos": initial_pos
}
LSHADE_cnEpSin = LSHADE_cnEpSin(objf, LSHADE_cnEpSin_param)
LSHADE_cnEpSin_best_f, LSHADE_cnEpSin_best_x, LSHADE_cnEpSin_convergence = LSHADE_cnEpSin.LSHADE_cnEpSin()
print(f"LSHADE_cnEpSin Best Solution: {LSHADE_cnEpSin_best_x},\nLSHADE_cnEpSin Best Fitness: {LSHADE_cnEpSin_best_f}")

"""AGSK"""
# Initialize and run the AGSK (possibly an adaptive genetic-swarm hybrid) algorithm
agsk_param = {
    "dim": dim,
    "pop": PopSize,
    "MaxIter": Maxiter,
    "lb": lb,
    "ub": ub,
    "initial_pos": initial_pos,
}
AGSK = AGSK(objf, agsk_param)
AGSK_best_f, AGSK_best_x, AGSK_convergence = AGSK.AGSK()
print(f"AGSK Best Solution: {AGSK_best_x},\nAGSK Best Fitness: {AGSK_best_f}")


'''
Plot convergence curves
'''
plt.figure()
plt.semilogy(LIBWO_convergence, color='blue', linewidth=2, label='LIBWO')
plt.plot(BWO_convergence, color='red', linewidth=2, label='BWO')
plt.plot(DOA_best_f, color='cyan', linewidth=2, label='DOA')
plt.plot(ala_it, color='gold', linewidth=2, label='ALA')
plt.plot(SDO_convergence, color='darkgreen', linewidth=2, label='SDO')
plt.plot(CBSO_convergence, color='brown', linewidth=2, label='CBSO')
plt.plot(PSEQADE_convergence, color='magenta', linewidth=2, label='PSEQADE')
plt.plot(COVIDOA_convergence, color='purple', linewidth=2, label='COVIDOA')
plt.plot(SASS_convergence, color='darkblue', linewidth=2, label='SASS')
plt.plot(LSHADE_cnEpSin_convergence, color='green', linewidth=2, label='LSHADE_cnEpSin')
plt.plot(AGSK_convergence, color='orange', linewidth=2, label='AGSK')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.grid()
plt.title('Convergence curve: ' + 'cec' + year + '-' + fun_name + ', Dim=' + str(dim))
plt.legend()
f = plt.gcf()
if not os.path.exists('result/' + year + '/' + fun_name):
    os.makedirs('result/' + year + '/' + fun_name)
f.savefig('result/' + year + '/' + fun_name + '/' + test_version + '.tif')
plt.show()
f.clear()

'''
Log best/mean/std results
'''
libwo_res = f"LIBWO Best: {LIBWO_best_f}, Mean: {np.mean(LIBWO_convergence)}, Std: {pd.Series(LIBWO_convergence.ravel()).std()}"
bwo_res = f"BWO Best: {BWO_best_f}, Mean: {np.mean(BWO_convergence)}, Std: {pd.Series(BWO_convergence.ravel()).std()}"
doa_res = f"DOA Best: {min(DOA_best_f)}, Mean: {np.mean(DOA_convergence)}, Std: {pd.Series(DOA_convergence.ravel()).std()}"
ala_res = f"ALA Best: {min(ala_it)}, Mean: {np.mean(ala_it)}, Std: {pd.Series(ala_it).std()}"
sdo_res = f"SDO Best: {SDO_best_f}, Mean: {np.mean(SDO_convergence)}, Std: {pd.Series(SDO_convergence).std()}"
cbso_res = f"CBSO Best: {CBSO_best_f}, Mean: {np.mean(CBSO_convergence)}, Std: {pd.Series(CBSO_convergence).std()}"
pseqade_res = f"PSEQADE Best: {PSEQADE_best_f}, Mean: {np.mean(PSEQADE_convergence)}, Std: {pd.Series(PSEQADE_convergence).std()}"
covidoa_res = f"COVIDOA Best: {COVIDOA_best_f}, Mean: {np.mean(COVIDOA_convergence)}, Std: {pd.Series(COVIDOA_convergence).std()}"
sass_res = f"SASS Best: {SASS_best_f}, Mean: {np.mean(SASS_convergence)}, Std: {pd.Series(SASS_convergence).std()}"
lshade_res = f"LSHADE_cnEpSin Best: {LSHADE_cnEpSin_best_f}, Mean: {np.mean(LSHADE_cnEpSin_convergence)}, Std: {pd.Series(LSHADE_cnEpSin_convergence).std()}"
agsk_res = f"AGSK Best: {AGSK_best_f}, Mean: {np.mean(AGSK_convergence)}, Std: {pd.Series(AGSK_convergence).std()}"

print(libwo_res)
print(bwo_res)
print(doa_res)
print(ala_res)
print(sdo_res)
print(cbso_res)
print(pseqade_res)
print(covidoa_res)
print(sass_res)
print(lshade_res)
print(agsk_res)

'''
Save results to file
'''
file = open('result/' + year + '/' + fun_name + '/' + test_version + '.txt', "a")
file.write(libwo_res)
file.write('\n' + bwo_res)
file.write('\n' + doa_res)
file.write('\n' + ala_res)
file.write('\n' + sdo_res)
file.write('\n' + cbso_res)
file.write('\n' + pseqade_res)
file.write('\n' + covidoa_res)
file.write('\n' + sass_res)
file.write('\n' + lshade_res)
file.write('\n' + agsk_res)
file.write('\n' + 'LIBWO: ' + str(LIBWO_convergence))
file.write('\n' + 'BWO: ' + str(BWO_convergence))
file.write('\n' + 'DOA: ' + str(DOA_best_f))
file.write('\n' + 'ALA: ' + str(ala_it))
file.write('\n' + 'SDO: ' + str(SDO_best_f))
file.write('\n' + 'CBSO: ' + str(CBSO_best_f))
file.write('\n' + 'PSEQADE: ' + str(PSEQADE_best_f))
file.write('\n' + 'COVIDOA: ' + str(COVIDOA_best_f))
file.write('\n' + 'SASS: ' + str(SASS_best_f))
file.write('\n' + 'LSHADE_cnEpSin: ' + str(LSHADE_cnEpSin_best_f))
file.write('\n' + 'AGSK: ' + str(AGSK_best_f))
file.close()
