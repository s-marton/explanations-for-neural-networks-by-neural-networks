

# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
import pandas as pd
import scipy as sc
from scipy.special import digamma, gamma
import itertools
import copy

from mpmath import *
from sympy import *
#from sympy.printing.theanocode import theano_function
from sympy.utilities.autowrap import ufuncify

from pysymbolic_adjusted.models.special_functions import *

from tqdm import tqdm, trange, tqdm_notebook, tnrange

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sympy import Integral, Symbol
from sympy.abc import x, y

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

from IPython import get_ipython


def is_ipython():
    
    try:
        
        __IPYTHON__
        
        return True
    
    except NameError:
        
        return False


def basis(a, b, c, x, hyper_order=[1, 2, 2, 2], approximation_order=3):
        
    epsilon = 0.001
    
    
    #print('a, b, c', a, b, c)
    
    func_   = MeijerG(theta=[a, a, a, b, c], order=hyper_order, approximation_order=approximation_order)
    
    #print('END MeijerG')
    
    return func_.evaluate(x + epsilon)

def basis_expression(a, b, c, hyper_order=[1, 2, 2, 2], approximation_order=3):
        
    func_ = MeijerG(theta=[a, a, a, b, c], order=hyper_order, approximation_order=approximation_order)
    
    return func_
    

def basis_grad(a, b, c, x, hyper_order=[1, 2, 2, 2]):
    
    if c <= 0: #if c <= 0, log or div cant be calculated ans thus return nan 
        grad_a = np.empty(x.shape)
        grad_a[:] = np.nan
        grad_b = np.empty(x.shape)
        grad_b[:] = np.nan
        grad_c = np.empty(x.shape)
        grad_c[:] = np.nan        
        return grad_a, grad_b, grad_c
    
    #print('abc', a, b, c)
    
    K1     = sc.special.digamma(a - b + 1)
    K2     = sc.special.digamma(a - b + 2)
    K3     = sc.special.digamma(a - b + 3)
    K4     = sc.special.digamma(a - b + 4)
    
    #print('K', K1, K2, K3, K4)
    
    G1     = sc.special.gamma(a - b + 1)
    G2     = sc.special.gamma(a - b + 2)
    G3     = sc.special.gamma(a - b + 3)
    G4     = sc.special.gamma(a - b + 4)
    
    #print('G', G1, G2, G3, G4)
        
    nema1  = 6 * ((c * x)**3) * (K4 - np.log(c * x))
    nema2  = 2 * ((c * x)**2) * (-K3 + np.log(c * x))
    nema3  = (c * x) * (K2 - np.log(c * x))
    nema4  = -1 * (K1 - np.log(c * x))
    
    #print('nema', nema1[:5], nema2[:5], nema3[:5], nema4[:5])
    
    nemb1  = -1 * 6 * ((c * x)**3) * K4 
    nemb2  = 2 * ((c * x)**2) * K3 
    nemb3  = -1 * (c * x) * K2 
    nemb4  = K1 

    #print('nemb', nemb1[:5], nemb2[:5], nemb3[:5], nemb4)
    
    nemc1  = -1 * (c**2) * (x**3) * (6 * a + 18)
    nemc2  = (c * (x**2)) * (4 + 2 * a) 
    nemc3  = -1 * x * (1 + a)
    nemc4  = a / c
    
    #print('nemc', nemc1[:5], nemc2[:5], nemc3[:5], nemc4)
    
    grad_a = ((c * x) ** a) * (nema1/G4 + nema2/G3 + nema3/G2 + nema4/G1) 
    grad_b = ((c * x) ** a) * (nemb1/G4 + nemb2/G3 + nemb3/G2 + nemb4/G1) 
    grad_c = ((c * x) ** a) * (nemc1/G4 + nemc2/G3 + nemc3/G2 + nemc4/G1) 

    #print('grad', grad_a[:5], grad_b[:5], grad_c[:5])
    
    return grad_a, grad_b, grad_c



def tune_single_dim(lr, n_iter, x, y, verbosity=False, approximation_order=3, max_param_value=100):
    
    
    epsilon   = 0.001
    x         = x + epsilon
    
    a         = 2
    b         = 1
    c         = 1
    
    batch_size  = np.min((x.shape[0], 500)) 
        
    for u in range(n_iter):
        
        batch_index = np.random.choice(list(range(x.shape[0])), size=batch_size)
        
        #print('function calls')
                
        new_grads   = basis_grad(a, b, c, x[batch_index])
        func_true   = basis(a, b, c, x[batch_index], approximation_order=approximation_order)
        
        loss        =  np.mean((func_true - y[batch_index])**2)
        
        if verbosity:
        
            print("Iteration: %d \t--- Loss: %.3f" % (u, loss))
        
        #print('grads', new_grads)
        #print('func', func_true)
        
        grads_a   = np.mean(2 * new_grads[0] * (func_true - y[batch_index]))
        grads_b   = np.mean(2 * new_grads[1] * (func_true - y[batch_index]))
        grads_c   = np.mean(2 * new_grads[2] * (func_true - y[batch_index]))
                
            
        a_new = a - lr * grads_a
        b_new = b - lr * grads_b
        c_new = c - lr * grads_c
        if np.isnan([a_new, b_new, c_new]).any() or np.isinf([a_new, b_new, c_new]).any() or (np.abs(np.array([c_new, b_new, c_new])) > max_param_value).any() or c_new <= 0:
            break
        a = a_new
        b = b_new
        c = c_new          

                
        #grads_a   = np.nan_to_num(np.mean(2 * new_grads[0] * (func_true - y[batch_index])).astype(np.float32))
        #grads_b   = np.nan_to_num(np.mean(2 * new_grads[1] * (func_true - y[batch_index])).astype(np.float32))
        #grads_c   = np.nan_to_num(np.mean(2 * new_grads[2] * (func_true - y[batch_index])).astype(np.float32))
        
        
        #(grads_a, grads_b, grads_c) = Normalizer().fit_transform([np.nan_to_num([grads_a, grads_b, grads_c])])[0]
        
        #print('tune_single_dim abc', a, b, c)
    if verbosity:
        print('return abc', a, b, c)
    return a, b, c 


def compose_features(params, X, approximation_order=3):
    
    #print(params)
    
    X_out = [basis(a=float(params[k, 0]), b=float(params[k, 1]), c=float(params[k, 2]), 
                   x=X[:, k], hyper_order=[1, 2, 2, 2], approximation_order=approximation_order) for k in range(X.shape[1])] 
    
    return np.array(X_out).T
    

class symbolic_metamodel:
    
    def __init__(self, model, X, mode="classification", approximation_order=3, force_polynomial=False, verbosity=False):
                
        self.verbosity = verbosity
        
        self.feature_expander = PolynomialFeatures(2, include_bias=False, interaction_only=True)
        self.X                = X
        self.X_new            = self.feature_expander.fit_transform(X) 
        self.X_names          = self.feature_expander.get_feature_names()
        
        #print('self.X.shape', self.X.shape)
        #print('self.X_new.shape', self.X_new.shape)
        
        self.max_param_value = 100
        
        self.approximation_order = approximation_order
        self.force_polynomial = force_polynomial
                        
        if mode == "classification": 
        
            self.Y                = model.predict_proba(self.X)[:, 1]
            self.Y_r              = np.log(self.Y/(1 - self.Y))
            
        else:
            
            self.Y_r              = model.predict(self.X)
        
        
        self.num_basis        = self.X_new.shape[1]
        
        #print(self.num_basis)
        
        self.params_per_basis = 3
        self.total_params     = self.num_basis * self.params_per_basis + 1
        
        a_init                = 1.393628702223735 
        b_init                = 1.020550117939659
        c_init                = 1.491820813243337
        
        self.params           = np.tile(np.array([a_init, b_init, c_init]), [self.num_basis, 1])
        
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            
            self.tqdm_mode = tqdm_notebook
            
        else:
            
            self.tqdm_mode = tqdm
            
    
    def set_equation(self, reset_init_model=False):
         
        #print(self.params, self.X_new[:10], self.approximation_order)
            
        self.X_init           = compose_features(self.params, self.X_new, approximation_order=self.approximation_order)
        
        #print('self.X_init', self.X_init[:10])
        
        if reset_init_model:
            
            self.init_model   = Ridge(alpha=.1, fit_intercept=False, normalize=True) #LinearRegression
            
            self.init_model.fit(self.X_init, self.Y_r)
    
    def get_gradients(self, Y_true, Y_metamodel, batch_index=None):
        
        #print('FUNC: get_gradients', Y_true, Y_metamodel, batch_index)
        
        param_grads = self.params * 0
        epsilon     = 0.001 
        
        #print('self.params.shape[0]', self.params.shape[0])
        
        for k in range(self.params.shape[0]):
            
            a                 = float(self.params[k, 0])
            b                 = float(self.params[k, 1])
            c                 = float(self.params[k, 2])
            
            #print('abc', a, b, c)
            #print('self.X_new[:, k]', self.X_new[:, k])
            
            if batch_index is None:
                grads_vals    = basis_grad(a, b, c, self.X_new[:, k] +  epsilon)
            else:
                grads_vals    = basis_grad(a, b, c, self.X_new[batch_index, k] +  epsilon)
            
            
            
            param_grads[k, :] = np.array(self.loss_grads(Y_true, Y_metamodel, grads_vals))
            
            #print('param_grads[k, :]', param_grads[k, :])
        
        return param_grads
        
    
    def loss(self, Y_true, Y_metamodel):
        
        return np.mean((Y_true - Y_metamodel)**2)
    
    def loss_grads(self, Y_true, Y_metamodel, param_grads_x):
        
        loss_grad_a = np.mean(2 * (Y_true - Y_metamodel) * param_grads_x[0])
        loss_grad_b = np.mean(2 * (Y_true - Y_metamodel) * param_grads_x[1])
        loss_grad_c = np.mean(2 * (Y_true - Y_metamodel) * param_grads_x[2])
        
        return loss_grad_a, loss_grad_b, loss_grad_c 
    
    def loss_grad_coeff(self, Y_true, Y_metamodel, param_grads_x):
        
        loss_grad_ = np.mean(2 * (Y_true - Y_metamodel) * param_grads_x)
        
        return loss_grad_
        
    
    def fit(self, num_iter=10, batch_size=100, learning_rate=.01):
        if self.verbosity:
            print("---- Tuning the basis functions ----")
        
        for u in self.tqdm_mode(range(self.X.shape[1])):
            
            self.params[u, :] = tune_single_dim(lr=0.1, n_iter=500, x=self.X_new[:, u], y=self.Y_r, approximation_order=self.approximation_order, max_param_value=self.max_param_value)
            
        self.set_equation(reset_init_model=True)

        self.metamodel_loss = []
        if self.verbosity:
            print("----  Optimizing the metamodel  ----")
        
        #print(num_iter) 
        
        #print('self.params', self.params)
        
        for i in self.tqdm_mode(range(num_iter)):
                        
            batch_index = np.random.choice(list(range(self.X_new.shape[0])), size=batch_size)
            
            #print('batch_index', batch_index[:10])
                        
            if np.isnan(self.X_init[batch_index, :]).any() or np.isinf(self.X_init[batch_index, :]).any():
                if self.verbosity:
                    print('\n\nBREAK X_init')
                    print('self.X_init[batch_index, :]', self.X_init[batch_index, :])

                    print('self.params', self.params)
                    print('self.init_model.coef_', self.init_model.coef_)

                    print('self.exact_expression', self.exact_expression)
                    print('self.approx_expression', self.approx_expression)
                
                if i == 0:
                    self.set_equation()  
                    self.exact_expression, self.approx_expression = self.symbolic_expression()
                
                break                
                
            curr_func   = self.init_model.predict(np.nan_to_num(self.X_init[batch_index, :]))
            
            #print('curr_func', curr_func[:10])
                        
            #print('self.loss(self.Y_r[batch_index], curr_func)', self.loss(self.Y_r[batch_index], curr_func))
            
            #print('self.Y_r[batch_index]', self.Y_r[batch_index])
            
            if np.isnan(curr_func).any() or np.isinf(curr_func).any():
                if self.verbosity:
                    print('\n\nBREAK curr_func')
                    print('curr_func', curr_func)

                    print('self.params', self.params)
                    print('self.init_model.coef_', self.init_model.coef_)

                    print('self.exact_expression', self.exact_expression)
                    print('self.approx_expression', self.approx_expression)
                
                if i == 0:
                    self.set_equation()  
                    self.exact_expression, self.approx_expression = self.symbolic_expression()
                
                break
            
            #param_grads  = np.nan_to_num(self.get_gradients(self.Y_r[batch_index], curr_func, batch_index).astype(np.float32))
            param_grads  = self.get_gradients(self.Y_r[batch_index], curr_func, batch_index)
            
            params = self.params - learning_rate * param_grads #np.nan_to_num(self.params - learning_rate * param_grads, nan=0.001)

            
            
            
            
            #print('param_grads', param_grads[:10])
            #print('self.params', self.params[:10])
            
            #coef_grads            = np.nan_to_num(np.array([self.loss_grad_coeff(self.Y_r[batch_index], curr_func, self.X_init[batch_index, k]) for k in range(self.X_init.shape[1])]).astype(np.float32)) 
            
            coef_grads            = [self.loss_grad_coeff(self.Y_r[batch_index], curr_func, self.X_init[batch_index, k]) for k in range(self.X_init.shape[1])]
            
            coef_ = self.init_model.coef_ - learning_rate * np.array(coef_grads)

             
            #print('coef_grads', coef_grads[:10])
            #print('self.init_model.coef_', self.init_model.coef_[:10])
            
                        
            #print('self.init_model.coef_', self.init_model.coef_[:10])  
            
            if np.isnan(params).any() or (np.abs(np.array(params)) > self.max_param_value).any() or (params[:, 2] <= 0).any() or np.isnan(coef_).any() or np.abs((np.array(coef_) > self.max_param_value)).any():
                #self.set_equation()  
                #self.exact_expression, self.approx_expression = self.symbolic_expression()
                if self.verbosity:
                    print('\n\nBREAK Params or Coef')
                    print('curr_func', curr_func)

                    print('param_grads', param_grads)
                    print('params', params)
                    print('self.params', self.params)
                    print('coef_grads', coef_grads)
                    print('coef_', coef_)
                    print('self.init_model.coef_', self.init_model.coef_)

                    print('self.exact_expression', self.exact_expression)
                    print('self.approx_expression', self.approx_expression)
            
                if i == 0:
                    self.set_equation()  
                    self.exact_expression, self.approx_expression = self.symbolic_expression()
                
                break
            else:
                self.metamodel_loss.append(self.loss(self.Y_r[batch_index], curr_func))
                self.params  = params
                self.init_model.coef_ = coef_
                
                self.set_equation()  
                self.exact_expression, self.approx_expression = self.symbolic_expression()            
                if self.verbosity:
                    print('self.exact_expression', self.exact_expression)
                    print('self.approx_expression', self.approx_expression)
            
    def evaluate(self, X):
        
        X_modified  = self.feature_expander.fit_transform(X)
        X_modified_ = compose_features(self.params, X_modified, approximation_order=self.approximation_order)
        Y_pred_r    = self.init_model.predict(X_modified_)
        
        if self.force_polynomial:
            return Y_pred_r

        Y_pred      = 1 / (1 + np.exp(-1 * Y_pred_r))
        
        return Y_pred 
    
    def symbolic_expression(self):
    
        dims_ = []

        for u in range(self.num_basis):

            new_symb = self.X_names[u].split(" ")

            if len(new_symb) > 1:
    
                S1 = Symbol(new_symb[0].replace("x", "X"), real=True)
                S2 = Symbol(new_symb[1].replace("x", "X"), real=True)
        
                dims_.append(S1 * S2)
    
            else:
        
                S1 = Symbol(new_symb[0].replace("x", "X"), real=True)
    
                dims_.append(S1)
        
        self.dim_symbols = dims_
        
        sym_exact   = 0
        sym_approx  = 0
        x           = symbols('x')

        #print(self.num_basis)
        #print(self.init_model.coef_)
        #print('self.init_model.coef_.shape', self.init_model.coef_.shape)
        
        #if self.init_model.coef_.shape == (1,1):
        #    self.init_model.coef_ = self.init_model.coef_.reshape(1,)
        if len(self.init_model.coef_.shape) >= 2 and self.init_model.coef_.shape[0] == 1:
            self.init_model.coef_ = self.init_model.coef_.reshape(-1,)                
                
        #print('self.init_model.coef_.shape', self.init_model.coef_.shape)
                
        for v in range(self.num_basis):
    
            f_curr      = basis_expression(a=float(self.params[v,0]), 
                                           b=float(self.params[v,1]), 
                                           c=float(self.params[v,2]), 
                                           approximation_order=self.approximation_order)
        
            #print(v)
            #print('self.init_model.coef_', self.init_model.coef_)
            #print(self.init_model.coef_[v])
        
            #print(sympify(str(self.init_model.coef_[v] * re(f_curr.expression()))))
            #print(sympify(str(self.init_model.coef_[v] * re(f_curr.approx_expression()))))
            sym_exact  += sympify(str(self.init_model.coef_[v] * re(f_curr.expression()))).subs(x, dims_[v])
            
            sym_approx += sympify(str(self.init_model.coef_[v] * re(f_curr.approx_expression()))).subs(x, dims_[v])    
        
        if self.force_polynomial:
            return sym_exact, sym_approx
            
        return 1/(1 + exp(-1*sym_exact)), 1/(1 + exp(-1*sym_approx))   
    
    
    def get_gradient_expression(self):
        
        diff_dims  = self.dim_symbols[:self.X.shape[1]]
        gradients_ = [diff(self.approx_expression, diff_dims[k]) for k in range(len(diff_dims))]

        diff_dims  = [str(diff_dims[k]) for k in range(len(diff_dims))]
        evaluator  = [lambdify(diff_dims, gradients_[k], modules=['math']) for k in range(len(gradients_))]
    
        return gradients_, diff_dims, evaluator
    

    def _gradient(self, gradient_expressions, diff_dims, evaluator, x_in):
    
        Dict_syms  = dict.fromkeys(diff_dims)

        for u in range(len(diff_dims)):

            Dict_syms[diff_dims[u]] = x_in[u]
         
        grad_out  = [np.abs(evaluator[k](**Dict_syms)) for k in range(len(evaluator))]
    
        
        return np.array(grad_out)    
    
    
    def get_instancewise_scores(self, X_in):
    
        gr_exp, diff_dims, evaluator = self.get_gradient_expression()
    
        gards_ = [self._gradient(gr_exp, diff_dims, evaluator, X_in[k, :]) for k in range(X_in.shape[0])]
    
        return gards_
    
        
