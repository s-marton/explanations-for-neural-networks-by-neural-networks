# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
import pandas as pd
import scipy as sc
import itertools

import time

from mpmath import *
from sympy import *

from scipy.optimize import minimize

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from pysymbolic_adjusted.models.special_functions import MeijerG
from pysymbolic_adjusted.utilities.performance import compute_Rsquared

#from sympy.printing.theanocode import theano_function
import sympy as sym
from sympy.utilities.autowrap import ufuncify

from gplearn.genetic import SymbolicRegressor

import types

from interruptingcow import timeout

from copy import deepcopy

import traceback


def load_hyperparameter_config():

    hyperparameter_space = {
                        'hyper_1': (np.array([1.0,1.0]), [1,0,0,1]),
                        'hyper_2': (np.array([0.0,0.0,1.0,0.0,1.0]), [1,2,2,2]),
                        'hyper_3': (np.array([2.0,2.0,2.0,1.0,1.0]), [0,1,3,1]),
                        'hyper_4': (np.array([0.0,0.0,20.0]), [1,0,0,2])
                        }

    return hyperparameter_space                    


def Optimize(Loss, theta_0):
    opt       = minimize(Loss, theta_0, method='CG', options={'xtol': 1e-8, 'disp': True})
    Loss_     = opt.fun
    theta_opt = opt.x
    
    return theta_opt, Loss_ 


def symbolic_modeling(f, G_order, theta_0, npoints, xrange, n_vars=1, data=None, printing=False):
        
    if data is not None:
        X = data
    elif n_vars == 1:
        X  = np.linspace(xrange[0], xrange[1], npoints).reshape((-1,1))
    else:
        X  = np.random.uniform(low=xrange[0], high=xrange[1], size=(npoints, n_vars))
        
        
    def Loss(theta):
        
        G     = MeijerG(theta=theta, order=G_order)
        if type(f) is types.FunctionType:
            loss_ = np.mean((f(X) - G.evaluate(X))**2)
        else:
            loss_ = np.mean((f.predict(X) - G.evaluate(X))**2)
            
        if printing:
            print("Expression:", G.expression())
            print("Loss:", loss_)
        
        return loss_
    
    theta_opt, Loss_ = Optimize(Loss, theta_0)
    symbolic_model   = MeijerG(theta=theta_opt, order=G_order)

    return symbolic_model, Loss_ 

def get_symbolic_model(f, npoints, xrange, n_vars=1, data=None, printing=False):

    hyperparameter_space = load_hyperparameter_config() 
    loss_threshold       = 10e-5

    symbol_exprs         = []
    losses_              = [] 

    for k in range(len(hyperparameter_space)):

        symbolic_model, Loss_ = symbolic_modeling(f, hyperparameter_space['hyper_'+str(k+1)][1], 
                                                  hyperparameter_space['hyper_'+str(k+1)][0], npoints, xrange, n_vars, printing=printing)

        symbol_exprs.append(symbolic_model)
        losses_.append(Loss_)

        if losses_[-1] <= loss_threshold:
            break 

    best_model = np.argmin(np.array(losses_))

    X          = np.linspace(xrange[0], xrange[1], npoints).reshape((-1,1))
    
    if type(f) is types.FunctionType:
        Y_true     = f(X).reshape((-1,1))
    else:
        Y_true     = f.predict(X).reshape((-1,1))
    Y_est      = symbol_exprs[best_model].evaluate(X).reshape((-1,1))

    R2_perf    = compute_Rsquared(Y_true, Y_est)
    
    return symbol_exprs[best_model], R2_perf    




    try:
        with timeout(60*max_optimization_minutes, exception=RuntimeError):
            symbolic_reg, r2_score, time_required   = symbolic_regressor(model, metamodeling_hyperparams['dataset_size'], [x_min, x_max], sample_sparsity, n_vars=config['n'], printing=printing)
    except (RuntimeError, MemoryError) as e:
        print(e)
        if return_error:
            return np.nan, None, np.nan
        else:
            return None, np.nan         

def symbolic_regressor(f, npoints, xrange, sparsity, n_vars=1, data=None, printing=True, max_optimization_minutes=60):

    start = time.time()
    
    
    if data is not None:
        X = data
    elif n_vars == 1:
        X  = np.linspace(xrange[0], xrange[1], npoints).reshape((-1,1))
    else:
        X  = np.random.uniform(low=xrange[0], high=xrange[1], size=(npoints, n_vars))

    #print(f)
    #print('X[0]', X[0])
    #print('X.shape', X.shape)
    
    if type(f) is types.FunctionType:
        y  = f(X)
    else:
        #print(f.summary())
        y  = f.predict(X)

    generations = 100
    metric='mean absolute error'
    early_stopping = 10
    epsilon = 0#0.001
    
    #sparsity= 10
    
    est_gp = SymbolicRegressor(population_size=5000,
                               generations=1,#1000, 
                               #stopping_criteria=0.01,
                               tournament_size=100,#
                               init_depth=(int(np.round(np.log2(sparsity))), int(np.round(np.log2(sparsity)))),#
                               #p_crossover=0.7, 
                               #p_subtree_mutation=0.1,
                               #p_hoist_mutation=0.05, 
                               #p_point_mutation=0.1,
                               #max_samples=0.9, 
                               verbose=1 if printing else 0,
                               #parsimony_coefficient=1e-5,#'auto',#1e-5#1e-10,#0,#0.0001,#0.001,#0,#0.01, 
                               random_state=0,
                               metric=metric,#
                               #low_memory=True,
                               function_set=['add', 'sub', 'mul'],
                              )
    current_generation = 0
    best_fitness = np.inf
    early_stopping_counter = 0
    
    try:
        with timeout(60*max_optimization_minutes, exception=RuntimeError):    
            for generation in range(generations):
                est_gp_old = deepcopy(est_gp)
                est_gp.fit(X, y)
                current_generation += 1
                est_gp.set_params(generations=current_generation+1, warm_start=True)
                best_fitness_generation = est_gp.run_details_['best_fitness'][-1]
                #print(est_gp.run_details_)
                if printing:
                    print('best_fitness_generation', best_fitness_generation)
                    print('best_fitness', best_fitness)
                if best_fitness_generation < best_fitness-epsilon:
                    early_stopping_counter = 0
                    best_fitness = best_fitness_generation
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stopping:
                    break
            
    except (RuntimeError, MemoryError) as e:
        print(e)
        print(traceback.print_exc())
        est_gp = est_gp_old
        #SymbolicRegressor(population_size=1000, generations=20, tournament_size=20, stopping_criteria=0.0, const_range=(-1.0, 1.0), init_depth=(2, 6), init_method='half and half', function_set=('add', 'sub', 'mul', 'div'), metric='mean absolute error', parsimony_coefficient=0.001, p_crossover=0.9, p_subtree_mutation=0.01, p_hoist_mutation=0.01, p_point_mutation=0.01, p_point_replace=0.05, max_samples=1.0, feature_names=None, warm_start=False, low_memory=False, n_jobs=1, verbose=0, random_state=None)    
    
    
    sym_expr = str(est_gp._program)

    end = time.time()
    time_required = end - start
    
    
    converter = {
        'sub': lambda x, y : x - y,
        'div': lambda x, y : x/y,
        'mul': lambda x, y : x*y,
        'add': lambda x, y : x + y,
        'neg': lambda x    : -x,
        'pow': lambda x, y : x**y
    }

    #x, X0   = symbols('x X0')
    sym_reg = simplify(sympify(sym_expr, locals=converter))
    #print('str(sym_reg)', str(sym_reg))
    #sym_reg = sym_reg.subs(X0,x)

    Y_true  = y.reshape((-1,1))
    #print('SUBS str(sym_reg)', str(sym_reg))
    
    
    function_vars = [sym.symbols('X' + str(i)) for i in range(n_vars)]
    #lambda_function = lambdify([function_vars], sym_reg, modules=["scipy", "numpy"])
    #if len(function_vars) >= 1:
    #    Y_est = [lambda_function(data_point) for data_point in X]
    #else:
    #Y_est = [lambda_function() for i in range(X.shape[0])]  
    try:
        with timeout(60*max_optimization_minutes, exception=RuntimeError):
            function_values = []
            for data_point in X:
                function_value = sym_reg.evalf(subs={var: data_point[index] for index, var in enumerate(list(function_vars))})
                try:
                    function_value = float(function_value)
                except TypeError as te:
                    function_value = np.inf
                function_values.append(function_value)
            Y_est = np.nan_to_num(function_values).ravel()

            #Y_est   = np.array([sympify(str(sym_reg)).subs(x,X[k]) for k in range(len(X))]).reshape((-1,1))
            if printing:
                print(sym_reg)

            R2_perf = compute_Rsquared(Y_true, Y_est)

    except (RuntimeError, MemoryError) as e:
        print(e)
        print(traceback.print_exc())
        return None, np.nan, np.nan
    
    del est_gp, est_gp_old
    
    return sym_reg, R2_perf, time_required

