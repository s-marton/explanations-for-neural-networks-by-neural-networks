#######################################################################################################################################################
#######################################################################Imports#########################################################################
#######################################################################################################################################################

#from itertools import product       # forms cartesian products
from tqdm import tqdm_notebook as tqdm
#import pickle
import numpy as np
from numpy import linspace
import pandas as pd
import scipy as sp

from functools import reduce
from more_itertools import random_product
import operator

import math

from joblib import Parallel, delayed
from collections.abc import Iterable
#from scipy.integrate import quad
import matplotlib.pyplot as plt 


#from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score, mean_absolute_error, r2_score
from similaritymeasures import frechet_dist, area_between_two_curves, dtw
import time

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from IPython.display import display, Math, Latex, clear_output

import os
import shutil
import pickle

import traceback
    
#udf import
from utilities.LambdaNet import *
from utilities.metrics import *
#from utilities.utility_functions import *

from scipy.optimize import minimize
from scipy import optimize
import sympy as sym
from sympy import Symbol, sympify, lambdify, abc, SympifyError

# Function Generation 0 1 import
from sympy.sets.sets import Union
from sympy import Number
import math

from numba import jit, njit
import itertools 

from interruptingcow import timeout

import time

from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures


#######################################################################################################################################################
#############################################################Setting relevant parameters from current config###########################################
#######################################################################################################################################################

def initialize_utility_functions_config_from_curent_notebook(config):
       
    try:
        globals().update(config['data'])
    except KeyError:
        print(KeyError)
        
    try:
        globals().update(config['lambda_net'])
    except KeyError:
        print(KeyError)
        
    try:
        globals().update(config['i_net'])
    except KeyError:
        print(KeyError)
        
    try:
        globals().update(config['evaluation'])
    except KeyError:
        print(KeyError)
        
    try:
        globals().update(config['computation'])
    except KeyError:
        print(KeyError)
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if int(tf.__version__[0]) >= 2:
        tf.random.set_seed(RANDOM_SEED)
    else:
        tf.set_random_seed(RANDOM_SEED)
        
    global list_of_monomial_identifiers
    from utilities.utility_functions import flatten, rec_gen, gen_monomial_identifier_list

    list_of_monomial_identifiers_extended = []

    if laurent:
        variable_sets = [list(flatten([[_d for _d in range(d+1)], [-_d for _d in range(1, neg_d+1)]])) for _ in range(n)]
        list_of_monomial_identifiers_extended = rec_gen(variable_sets)    

        if len(list_of_monomial_identifiers_extended) < 500:
            print(list_of_monomial_identifiers_extended)     

        list_of_monomial_identifiers = []
        for monomial_identifier in tqdm(list_of_monomial_identifiers_extended):
            if np.sum(monomial_identifier) <= d:
                if monomial_vars == None or len(list(filter(lambda x: x != 0, monomial_identifier))) <= monomial_vars:
                    list_of_monomial_identifiers.append(monomial_identifier)        
    else:
        variable_list = ['x'+ str(i) for i in range(n)]
        list_of_monomial_identifiers = gen_monomial_identifier_list(variable_list, d, n)

                                    
#######################################################################################################################################################
#############################################################General Utility Functions#################################################################
#######################################################################################################################################################


def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(Number)})

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


def rec_gen(x): 
    if len(x) == 1:                                                                 
        return [[item] for item in x[0]]                                           
    appended = []                                                                  
    for s_el in x[0]:                                                              
        for next_s in rec_gen(x[1:]):                                              
            appended.append([s_el] + next_s)                                       
    return appended                                                                


def gen_monomial_identifier_list(variable_list, degree, number_of_variables):
    
    def get_polynomial(vars, power): 

        if "c" in vars: 
            raise Exception("\"c\" cannot be a variable") 

        vars.append("c") # add dummy variable 

        # compute all combinations of variables 
        terms = [] 
        for x in itertools.combinations_with_replacement(vars, power): 
            terms.append(x) 

        # get rid of "c" terms 
        terms = list(map(list, terms))
        for i in range(len(terms)): 
            while "c" in terms[i]: 
                terms[i].remove("c") 

        return terms    
    
    
    terms = get_polynomial(variable_list, degree) 

    monomial_identifier_list = []
    for term in terms:
        monomial = [0 for i in range(number_of_variables)]
        for value in term:
            index = int(value[1:])
            monomial[index] = monomial[index] + 1
        monomial_identifier_list.append(monomial)    
        
    return monomial_identifier_list
    
def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def chunks(lst, chunksize):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunksize):
        yield lst[i:i + chunksize]

def prod(iterable):
    return reduce(operator.mul, iterable, 1)
        

def return_float_tensor_representation(some_representation, dtype=tf.float32):
    if tf.is_tensor(some_representation):
        some_representation = tf.dtypes.cast(some_representation, dtype) 
    else:
        some_representation = tf.convert_to_tensor(some_representation)
        some_representation = tf.dtypes.cast(some_representation, dtype) 
        
    if not tf.is_tensor(some_representation):
        raise SystemExit('Given variable is no instance of ' + str(dtype) + ':' + str(some_representation))
     
    return some_representation


def return_numpy_representation(some_representation):
    if isinstance(some_representation, pd.DataFrame):
        some_representation = some_representation.values
        some_representation = np.float32(some_representation)
        
    if isinstance(some_representation, list):
        some_representation = np.array(some_representation, dtype=np.float32)
        
    if isinstance(some_representation, np.ndarray):
        #print(some_representation)
        #print(type(some_representation))
        #print(some_representation.dtype)
        #print(some_representation[0])
        #print(some_representation[0].dtype)
        
        
        some_representation = np.float32(some_representation)
    else:
        raise SystemExit('Given variable is no instance of ' + str(np.ndarray) + ':' + str(some_representation))
    
    return some_representation

def mergeDict(dict1, dict2):
    #Merge dictionaries and keep values of common keys in list
    newDict = {**dict1, **dict2}
    for key, value in newDict.items():
        if key in dict1 and key in dict2:
            if isinstance(dict1[key], list) and isinstance(value, list):
                newDict[key] = dict1[key]
                newDict[key].extend(value)
            elif isinstance(dict1[key], list) and not isinstance(value, list):
                newDict[key] = dict1[key]
                newDict[key].extend([value])
            elif not isinstance(dict1[key], list) and isinstance(value, list):
                newDict[key] = [dict1[key]]
                newDict[key].extend(value)
            else:
                newDict[key] = [dict1[key], value]
    return newDict

def return_callbacks_from_string(callback_string_list):
    callbacks = [] if len(callback_string_list) > 0 else None
    #if 'plot_losses_callback' in callback_string_list:
        #callbacks.append(PlotLossesCallback())
    if 'reduce_lr_loss' in callback_string_list:
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=min(50, epochs//10), verbose=0, min_delta=0, mode='min') #epsilon
        callbacks.append(reduce_lr_loss)
    if 'early_stopping' in callback_string_list:
        try:
            patience = early_stopping_patience if early_stopping_patience is not None else min(50, epochs//10)
        except:
            patience = min(50, epochs//10)
        
        earlyStopping = EarlyStopping(monitor='val_loss', patience=patience, min_delta=0, verbose=0, mode='min', restore_best_weights=True)
        callbacks.append(earlyStopping)        
    #if not multi_epoch_analysis and samples_list == None: 
        #callbacks.append(TQDMNotebookCallback())        
    return callbacks

def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def shape_flat_network_parameters(flat_network_parameters, target_network_parameters):
    
    #from utilities.utility_functions import flatten_list
    
    #def recursive_len(item):
    #    if type(item) == list:
    #        return sum(recursive_len(subitem) for subitem in item)
    #    else:
    #        return 1      
        
    shaped_network_parameters =[]
    start = 0  
    
    for parameters in target_network_parameters:
        target_shape = parameters.shape
        size = np.prod(target_shape)#recursive_len(el)#len(list(flatten_list(el)))
        shaped_parameters = np.reshape(flat_network_parameters[start:start+size], target_shape)
        shaped_network_parameters.append(shaped_parameters)
        start += size

    return shaped_network_parameters


def shaped_network_parameters_to_array(shaped_network_parameters):
    network_parameter_list = []
    for layer_weights, biases in pairwise(shaped_network_parameters):    #clf.get_weights()
        for neuron in layer_weights:
            for weight in neuron:
                network_parameter_list.append(weight)
        for bias in biases:
            network_parameter_list.append(bias)
                
    return np.array(network_parameter_list)

#################################################################################################################################################################################### Normalization #################################################################################### ################################################################################################################################################################################################################

def get_order_sum(arrays):
    arrays = np.array(arrays)
    values = [np.sum(arrays[0])]
    order = [0]
    for i in range(1, len(arrays)):
        value = np.sum(arrays[i])
        pos = 0
        while pos<len(values) and value>=values[pos]:
            if value == values[pos]:
                print("!!!!!!!!!!!!!!!!KOLLISION!!!!!!!!!!!!!!!!!!")
                print(value)
                print(arrays[i])
                print(arrays[order[pos]])
            pos += 1
        values.insert(pos, value)
        order.insert(pos, i)
    return order

## source for sort_array: https://www.geeksforgeeks.org/permute-the-elements-of-an-array-following-given-order/

def sort_array(arr, order):
    length = len(order)
    #ordered_arr = np.zeros(length)
    ordered_arr = [None] * length
    for i in range(length):
        ordered_arr[i] = arr[order[i]]
    arr=ordered_arr
    return arr    

def normal_neural_net(model_arr):
    for i in range(len(lambda_network_layers)):
        index = 2*(i)
        dense_arr = np.transpose(model_arr[index])
        order = get_order_sum(dense_arr)
        for j in range(len(model_arr[index])):
            model_arr[index][j] = sort_array(model_arr[index][j], order)
        model_arr[index+1] = np.array(sort_array(model_arr[index+1], order))
        model_arr[index+2] = np.array(sort_array(model_arr[index+2], order))
    return model_arr


            
            
def print_polynomial_from_coefficients(coefficient_array, force_complete_poly_representation=False, round_digits=None):
    return display(get_sympy_string_from_coefficients(coefficient_array, force_complete_poly_representation=force_complete_poly_representation, round_digits=round_digits))


def get_polynomial_string_from_coefficients(coefficients, force_complete_poly_representation=False, round_digits=None):

    global list_of_monomial_identifiers
    global interpretation_net_output_monomials
        
    string = ''
    
    try: #catch if this is lambda-net training
        interpretation_net_output_monomials == None
    except NameError:
        interpretation_net_output_monomials = None
        
    if interpretation_net_output_monomials == None or force_complete_poly_representation:
        for identifier, coefficient in zip(list_of_monomial_identifiers, coefficients):
            if round_digits != None:
                string += str(np.round(coefficient, round_digits))
            else:
                string += str(coefficient)
            for index, variable_identifier in enumerate(identifier):  
                if int(variable_identifier) == 1:
                    #string += '*'
                    string += 'abcdefghijklmnopqrstuvwxyz'[index]
                elif int(variable_identifier) > 1:
                    #string += '*'
                    string += 'abcdefghijklmnopqrstuvwxyz'[index] + '^' + str(variable_identifier)
            string += ' + '
    else:
        # Convert output array to monomial identifier index and corresponding coefficient
        assert coefficient_array.shape[0] == interpretation_net_output_shape or coefficient_array.shape[0] == interpretation_net_output_shape + 1 + len(list_of_monomial_identifiers) 
        
        if coefficient_array.shape[0] == interpretation_net_output_shape:
            coefficients = coefficient_array[:interpretation_net_output_monomials]
            index_array = coefficient_array[interpretation_net_output_monomials:]


            assert index_array.shape[0] == interpretation_net_output_monomials*sparsity or index_array.shape[0] == interpretation_net_output_monomials*(d+1)*n
            index_list = np.split(index_array, interpretation_net_output_monomials)

            assert len(index_list) == coefficients.shape[0] == interpretation_net_output_monomials
            indices = np.argmax(index_list, axis=1)    
        else:
            coefficients = coefficient_array[:interpretation_net_output_monomials+1]
            index_array = coefficient_array[interpretation_net_output_monomials+1:]


            assert index_array.shape[0] == (interpretation_net_output_monomials+1)*sparsity
            index_list = np.split(index_array, interpretation_net_output_monomials+1)

            assert len(index_list) == coefficients.shape[0] == interpretation_net_output_monomials+1
            indices = np.argmax(index_list, axis=1)   
            
        for monomial_index, monomial_coefficient in zip(indices, coefficients):
            if round_digits != None:
                string += str(np.round(monomial_coefficient, round_digits))
            else:
                string += str(monomial_coefficient)
            #REPLACE NAN            
            for i, monomial_exponent in enumerate(list_of_monomial_identifiers[monomial_index]):
                if int(monomial_exponent) == 1:
                    #string += '*'
                    string += 'abcdefghijklmnopqrstuvwxyz'[i]
                elif int(monomial_exponent) > 1:
                    #string += '*'
                    string += 'abcdefghijklmnopqrstuvwxyz'[i] + '^' + str(monomial_exponent)                  
            string += ' + '   
            
    return string[:-3]

def get_sympy_string_from_coefficients(coefficient_array, force_complete_poly_representation=False, round_digits=None):
    
    global list_of_monomial_identifiers
    global interpretation_net_output_monomials
    
    variable_alphabet =  "abcdefghijklmnopqrstuvwxyz"
    
    variable_list = []
    for i in range(n):
        variable_list.append(sym.symbols(variable_alphabet[i]))    
    
    try: #catch if this is lambda-net training
        interpretation_net_output_monomials == None
    except NameError:
        interpretation_net_output_monomials = None
    
    if interpretation_net_output_monomials == None or force_complete_poly_representation:   
        f = 0
        for monomial_identifier, monomial_coefficient in zip(list_of_monomial_identifiers, coefficient_array):
            if round_digits != None:
                subfunction = np.round(monomial_coefficient, round_digits)
            else:
                subfunction = monomial_coefficient        
            for i, monomial_exponent in enumerate(monomial_identifier):
                subfunction *= variable_list[i]**monomial_exponent
            f += subfunction
    else:
        f = 0
        
        # Convert output array to monomial identifier index and corresponding coefficient
        assert coefficient_array.shape[0] == interpretation_net_output_shape or coefficient_array.shape[0] == interpretation_net_output_shape + 1 + len(list_of_monomial_identifiers) 
        
        if coefficient_array.shape[0] == interpretation_net_output_shape:
            coefficients = coefficient_array[:interpretation_net_output_monomials]
            index_array = coefficient_array[interpretation_net_output_monomials:]

            assert index_array.shape[0] == interpretation_net_output_monomials*sparsity or index_array.shape[0] == interpretation_net_output_monomials*(d+1)*n
            index_list = np.split(index_array, interpretation_net_output_monomials)

            assert len(index_list) == coefficients.shape[0] == interpretation_net_output_monomials
            indices = np.argmax(index_list, axis=1)
        else:
            coefficients = coefficient_array[:interpretation_net_output_monomials+1]
            index_array = coefficient_array[interpretation_net_output_monomials+1:]

            assert index_array.shape[0] == (interpretation_net_output_monomials+1)*sparsity
            index_list = np.split(index_array, interpretation_net_output_monomials+1)

            assert len(index_list) == coefficients.shape[0] == interpretation_net_output_monomials+1
            indices = np.argmax(index_list, axis=1)
        
        
        for monomial_index, monomial_coefficient in zip(indices, coefficients):
            if round_digits != None:
                subfunction = np.round(monomial_coefficient, round_digits)
            else:
                subfunction = monomial_coefficient
                #REPLACE NAN
            for i, monomial_exponent in enumerate(list_of_monomial_identifiers[monomial_index]):
                subfunction *= variable_list[i]**monomial_exponent
            f += subfunction    
    
    return f


def plot_polynomial_from_coefficients(coefficient_array, force_complete_poly_representation=False):
    
    sympy_function_string = get_sympy_string_from_coefficients(coefficient_array, force_complete_poly_representation=False)
    
    variable_alphabet =  "abcdefghijklmnopqrstuvwxyz"
    
    variable_list = []
    for i in range(n):
        variable_list.append(sym.symbols(variable_alphabet[i]))       
    
    lam_x = lambdify(variable_list, sympy_function_string, modules=['numpy'])
    
    x_vals = linspace(x_min, x_max, 100)
    y_vals = lam_x(x_vals)

    plt.plot(x_vals, y_vals)
    plt.show()
    
                
def get_critical_points_from_polynomial(coefficient_array, force_complete_poly_representation=False): 
    
    
    coefficient_array = return_numpy_representation(coefficient_array)
    
    #assert coefficient_array.shape[0] == interpretation_net_output_shape
        
    f = get_sympy_string_from_coefficients(coefficient_array, force_complete_poly_representation=force_complete_poly_representation)
        
    gradient = sym.derive_by_array(f, tuple(f.free_symbols))
        
    stationary_points = sym.solve(gradient, tuple(f.free_symbols))
    
    
    return f, gradient, stationary_points



#######################################################################################################################################################
########################################################################JUSTUS CODE####################################################################
#######################################################################################################################################################
# simplified the function for this usecase
def get_sympy_string_from_coefficients_fg(coefficient_array, variable_list):
    
    global list_of_monomial_identifiers
    
    f=0
    for i in range(sparsity):
        monomial = coefficient_array[i]
        for j in range(n):
            monomial = monomial*variable_list[j]**int(list_of_monomial_identifiers[i][j])
        f += monomial

    return f

# Method to shift a function(func) by a given distance(distance) for a given variable(variable)
def shift(func, distance, variable):
    a = variable
    f = func
    # substitude a by a-distance (shifting)
    f = f.subs(a, (a-distance))
    # expand function returns polynomial funtion as sum of monomials
    f = sym.expand(f)
    return f

# Method to bulge a function(func) by a given factor(factor) for a given variable(variable)
def bulge(func, factor, variable):
    a = variable
    f = func
    #substitude a by a*factor (bulging)
    f = f.subs(a, (factor*a))
    #expand function returns polynomial funtion as sum of monomials
    f = sym.expand(f)
    return f

def adjust_function(f, borders, variables):
    variables = list(f.free_symbols)
    width = [1 - border - random.uniform(border_min, border_max) for border in borders] # space between borders (only left border is saved)
    
    # check if the degree is 1 (there are no stationary points for functions of degree 1. Therefore they can't be adjusted with this function)
    higher_degree = False
    for variable in variables:
        if sym.degree(f, variable) > 1:
            higher_degree = True
            break;
    if not higher_degree:
        return f
    
    # special method for functions with 1 variable (inflection points are also used) There also are some extras for functions with more than 1 variable that functions with only 1 variable do not need
    if n==1:
        g = sym.diff(f, variables[0])
        #find extremums ()
        ext = sym.solveset(g, domain=sym.Reals)
        #find inflection points
        inflec = sym.calculus.util.stationary_points(g, variables[0], domain=sym.Reals)
        #critical points (joint extremums and inflection points)
        critical_points = Union(ext, inflec)
        # Test, if there are any critical points (Only case where a polynomial function has no critical point is a straight, which causes no problem)
        if not critical_points.is_empty: 
            # find infimum and supremum of set:
            left_critical_point = critical_points.inf
            right_critical_point = critical_points.sup
            # calculate distance between points:
            distance = right_critical_point - left_critical_point
            # only one critical point
            
            if distance == 0:
                # shift function so that the critical point is between border and 1-border
                bulge_factor = random.uniform(bulge_min, bulge_max)
                shift_distance = -(stationary_points[0][variables[0]]) + bulge_factor * (borders[0] + random.uniform(0, width[0]))
                f = shift(f, shift_distance, variables[0])
                f = bulge(f, bulge_factor, variables[0])
                #f = shift(f, -left_critical_point+random.uniform(borders[0], 1-borders[0]), variables[0])
             
            # check if function needs to be bulged 
            elif distance <= width[0]:
                # shift function so that the critical points are between border and 1-border
                f = shift(f, -left_critical_point+borders[0]+random.uniform(0, width[0]-distance), variables[0])
                
            else:
                bulge_factor = distance/width[0]
                shift_distance = -left_critical_point + bulge_factor * borders[0]
                f = shift(f, shift_distance, variables[0])
                # bulge the function
                f = bulge(f, distance/width[0], variables[0])
        return f
    
    
    # determine the number of variables that are used in the search for stationary points (probabilties in configs)
    number_of_used_variables = random.choices([n, random.randint(min_variables_used, max_variables_used)], [global_stationary_prob, 1-global_stationary_prob])[0]
    used_variables = []
    # helper function to get stationary points
    f_copy = f
    
    # select 'number_of_used_variables' many variables
    while len(used_variables) < number_of_used_variables and len(used_variables)<len(variables):
        variable = variables[random.randint(0, len(variables)-1)]
        if not variable in used_variables:
            used_variables.append(variable)
            
    # substitute all variables that are not used with constants that are in the intervall
    for variable in variables:
        if not variable in used_variables:
            f_copy = f_copy.subs(variable, random.uniform(x_min, x_max))
            
    # adjustment of the used_variables, because some variables might not be in the function
    used_variables = list(f_copy.free_symbols)
    number_of_used_variables = len(used_variables)
    
    # special search for the use of only one variable (also uses inflection points)
    if number_of_used_variables == 1:
        g = sym.diff(f_copy, used_variables[0], domain=sym.Reals)
        #find extremums ()
        ext = sym.solveset(g, used_variables[0], domain=sym.Reals)
        #find inflection points
        inflec = sym.calculus.util.stationary_points(g, used_variables[0])
        #critical points (joint extremums and inflection points) (called stationary_points to use the same code)
        critical_points = Union(ext, inflec)
        stationary_points = []
        
        # filter out stationary points that are not real
        if not type(critical_points) is sym.sets.fancysets.Reals:
            for point in critical_points:
                stationary_points.append({used_variables[0]: point})
                
    # get stationary points for the use of more than one variable
    else:
        f_copy = sym.expand(f_copy)
        gradient = sym.derive_by_array(f_copy, tuple(f_copy.free_symbols))
        stationary_points = sym.solve(gradient, tuple(f_copy.free_symbols), dict=True)
        if len(stationary_points) == 0:
            return f;
        length_helper = len(stationary_points) - 1
        used_variables = list(stationary_points[0].keys())
        number_of_used_variables = len(used_variables)
        
        # filter out stationary points that are not real 
        for i in range(len(stationary_points)):
            for j in range(number_of_used_variables):
                if not stationary_points[length_helper-i][used_variables[j]].is_real:
                    stationary_points.pop(length_helper-i)
                    break;
                    
    # no stationary points => nothing can be adjusted => just return functions
    if len(stationary_points) == 0:
        return f;
    # 1 stationary point => shift it inside the intervall for all used variables and bulge it randomly
    if len(stationary_points) == 1:
        for i in range(number_of_used_variables):
            bulge_factor = random.uniform(bulge_min, bulge_max)
            shift_distance = -(stationary_points[0][used_variables[i]]) + bulge_factor * (borders[i] + random.uniform(0, width[i]))
            f = shift(f, shift_distance, used_variables[i])
            f = bulge(f, bulge_factor, used_variables[i])
            
    # minimum of two stationary points => shift them to the border limits
    else:
        for i in range(len(used_variables)):
            critical_values = [stationary_points[j][used_variables[i]] for j in range(len(stationary_points))]
            minimum = min(critical_values)
            distance = max(critical_values) - minimum
            bulge_factor = distance/width[i]
            shift_distance = -minimum + bulge_factor * borders[i]
            f = shift(f, shift_distance, used_variables[i])
            f = bulge(f, bulge_factor, used_variables[i])
    return f

def prep_post_polynomial (borders, values):
    variable_alphabet = "abcdefghijklmnopqrstuvwxyz"
    variable_list = [sym.symbols(variable_alphabet[i]) for i in range(n)]
    list_of_monomial_dict_names = []
    
    global list_of_monomial_identifiers
    
    # get dictionary keys to retrieve function 
    for mono_string in list_of_monomial_identifiers:
        helper = 1
        for i in range(n):
            if mono_string[i] != "0":
                if(helper!=1):
                    helper = helper*variable_list[i]**int(mono_string[i])
                else:
                    helper = variable_list[i]**int(mono_string[i])
        list_of_monomial_dict_names.append(helper)
    
    # get sympy string for adjustments
    function = get_sympy_string_from_coefficients_fg(values, variable_list)
    
    # adjustment
    function_adjusted = adjust_function(function, borders, variable_list)
    
    # get list representation from sympy representation
    coeff_dict = function_adjusted.as_coefficients_dict()
    coeff_list = [coeff_dict[monomial] for monomial in list_of_monomial_dict_names]
    
    # possible divisor for the case that coefficient values are to high
    divider = abs(max(coeff_list, key=abs) / random.uniform(a_max/4, a_max))
    if divider > 1:
        coeff_list = [x / divider for x in coeff_list]
        
    #adjust the y-axis intercept so that function are spread better
    if coeff_list[0] != 0:
        multiplier0 = random.uniform(1, a_max / abs(coeff_list[0]))
        coeff_list[0] = coeff_list[0] * multiplier0
    # NaN can happen if one coefficent has values of infinity after bulging or shifting
    for i in range(sparsity):
        if math.isnan(coeff_list[i]):
            values = [random.uniform(a_min, a_max) for _ in range(sparsity)]
            return prep_post_polynomial(borders, values)
    return coeff_list


def get_polynomial_basic (sparsities ,change = 0):
    # change adjusts the lower degree probability. Other values than 0 are better for function generations with low degree because if you use 0 there will be a lot of functions of degree 0. Example value:
    # change = -((lower_degree_prob / (d-1)) - (0.01 * d))
    values = np.zeros(sparsity)
    degree_helper = 1
    for i in range(d):
        if(random.random() < (lower_degree_prob + i*change)):
            degree_helper += 1
        else:
            break
            
    #return random nonadjusted function
    if random.random()<a_random_prob:
        for i in range(max_monomials_random-1):
            values[random.randint(0, sparsities[-degree_helper]-1)] = random.uniform(a_min, a_max)
        values[0] = random.uniform(a_min, a_max)
        return values
    
    # degree_helper >= d => maximum degree = 1 => no stationary points => no adjustment possible
    if degree_helper >= d:
        for i in range(max_monomials-1):
            values[random.randint(0, sparsities[-degree_helper]-1)] = random.uniform(a_min, a_max)
        values[0] = random.uniform(a_min, a_max)
        return values
    
    # get random borders (minimum space between x_min (x_max) to the critical_points
    borders = [random.uniform(border_min, border_max) for i in range(n)]
    try:
        with timeout(5, exception=RuntimeError):
            coeff_list = prep_post_polynomial(borders, values)
    except:
        return get_polynomial_basic(sparsities, change = change)
    return coeff_list


#######################################################################################################################################################
###########################Manual calculations for comparison of polynomials based on function values (no TF!)#########################################
#######################################################################################################################################################
#@njit#(nopython=True)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!DEPRECATED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def calcualate_function_value(coefficient_list, lambda_input_entry, force_complete_poly_representation=False, list_of_monomial_identifiers=None, interpretation_net_output_monomials=None):
    
    #print('coefficient_list', coefficient_list)
    #print('lambda_input_entry', lambda_input_entry)
    
        
    result = 0   
        
    #try: #catch if this is lambda-net training
    #    config['interpretation_net_output_monomials'] == None
    #except NameError:
    #    config['interpretation_net_output_monomials'] = None
        
        
    if interpretation_net_output_monomials == None or force_complete_poly_representation:
        
        #print('coefficient_list', coefficient_list)
        
        #print(force_complete_poly_representation)
        #print(interpretation_net_output_monomials)
    
        #assert coefficient_list.shape[0] == sparsity, 'Shape of Coefficient List: ' + str(coefficient_list.shape) + str(interpretation_net_output_monomials) + str(coefficient_list)
        
        for coefficient_value, coefficient_multipliers in zip(coefficient_list, list_of_monomial_identifiers):
            #print('coefficient_value', coefficient_value)
            #print('coefficient_multipliers', coefficient_multipliers)
            value_without_coefficient = [lambda_input_value**coefficient_multiplier for coefficient_multiplier, lambda_input_value in zip(coefficient_multipliers, lambda_input_entry)]
            #print('value_without_coefficient', value_without_coefficient)
            
            #try:
            result += coefficient_value * reduce(lambda x, y: x*y, value_without_coefficient)
            #except TypeError:
            #    print('ERROR')
            #    print(lambda_input_entry)
            #    print(coefficient_list)
            #
            #    print(coefficient_value)
            #    print(value_without_coefficient)
    else:
        
        # Convert output array to monomial identifier index and corresponding coefficient
        #ASSERT
        #assert coefficient_list.shape[0] == interpretation_net_output_shape or coefficient_list.shape[0] == interpretation_net_output_shape + 1 + len(list_of_monomial_identifiers) 
        
        
        if coefficient_list.shape[0] == interpretation_net_output_shape:
            coefficients = coefficient_list[:interpretation_net_output_monomials]
            index_array = coefficient_list[interpretation_net_output_monomials:]

            #ASSERT
            #assert index_array.shape[0] == interpretation_net_output_monomials*sparsity
            index_list = np.split(index_array, interpretation_net_output_monomials)

            #ASSERT
            #assert len(index_list) == coefficients.shape[0] == interpretation_net_output_monomials
            indices = np.argmax(index_list, axis=1)
        else: 
            coefficients = coefficient_list[:interpretation_net_output_monomials+1]
            index_array = coefficient_list[interpretation_net_output_monomials+1:]

            #ASSERT
            #assert index_array.shape[0] == (interpretation_net_output_monomials+1)*sparsity
            index_list = np.split(index_array, interpretation_net_output_monomials+1)

            #ASSERT
            #assert len(index_list) == coefficients.shape[0] == interpretation_net_output_monomials+1
            indices = np.argmax(index_list, axis=1)            

        # Calculate monomial values without coefficient
        value_without_coefficient_list = []
        for coefficient_multipliers in list_of_monomial_identifiers:
            value_without_coefficient = [lambda_input_value**coefficient_multiplier for coefficient_multiplier, lambda_input_value in zip(coefficient_multipliers, lambda_input_entry)]
            value_without_coefficient_list.append(reduce(lambda x, y: x*y, value_without_coefficient))
        value_without_coefficient_by_indices = np.array(value_without_coefficient_list)[[indices]]

        # Select relevant monomial values without coefficient and calculate final polynomial
        for coefficient, monomial_index in zip(coefficients, indices):
            #TODOOOOO
            result += coefficient * value_without_coefficient_list[monomial_index]
        
    #print('result', result)
    return result

#@jit#@jit(nopython=True)
def calculate_function_values_from_polynomial(polynomial, lambda_input_data, force_complete_poly_representation=False, list_of_monomial_identifiers=None, interpretation_net_output_monomials=None):        
    
    
    #function_value_list = []       
    #for lambda_input_entry in lambda_input_data:
        #function_value = calcualate_function_value(polynomial, lambda_input_entry, force_complete_poly_representation=force_complete_poly_representation, list_of_monomial_identifiers=list_of_monomial_identifiers, interpretation_net_output_monomials=interpretation_net_output_monomials)
        #function_value_list.append(function_value)
        
    
    config = {
         'n': n,
         #'inet_loss': inet_loss,
         'sparsity': sparsity,
         #'lambda_network_layers': lambda_network_layers,
         #'interpretation_net_output_shape': interpretation_net_output_shape,
         'RANDOM_SEED': RANDOM_SEED,
         #'nas': nas,
         #'number_of_lambda_weights': number_of_lambda_weights,
         #'interpretation_net_output_monomials': interpretation_net_output_monomials,
         #'list_of_monomial_identifiers': list_of_monomial_identifiers,
         'x_min': x_min,
         'x_max': x_max,
         }
    
    try:
        config['interpretation_net_output_monomials'] = interpretation_net_output_monomials
    except:
        config['interpretation_net_output_monomials'] = None
        
        
    #print(list_of_monomial_identifiers)
    #print(polynomial)
    #print(lambda_input_data)
        
    function_value_list = calculate_poly_fv_tf_wrapper_new(return_float_tensor_representation(list_of_monomial_identifiers), return_float_tensor_representation(polynomial), return_float_tensor_representation(lambda_input_data), force_complete_poly_representation=force_complete_poly_representation, config=config)
        
    return np.nan_to_num(np.array(function_value_list))



def parallel_fv_calculation_from_polynomial(polynomial_list, lambda_input_list, force_complete_poly_representation=False, n_jobs_parallel_fv=10, backend='threading'):
        
    print(force_complete_poly_representation)
        
    polynomial_list = return_numpy_representation(polynomial_list)
    lambda_input_list = return_numpy_representation(lambda_input_list)
    
    #print(polynomial_list.shape)
    #print(type(polynomial_list))
    #print(polynomial_list.dtype)
    #print(polynomial_list)
    #print(polynomial_list[0].shape)
    #print(type(polynomial_list[0]))
    #print(polynomial_list[0].dtype)
    #print(polynomial_list[0])
    
    assert polynomial_list.shape[0] == lambda_input_list.shape[0] 
        
    if force_complete_poly_representation:
        assert polynomial_list.shape[1] == sparsity
    else:
        assert polynomial_list.shape[1] == interpretation_net_output_shape or polynomial_list.shape[1] == interpretation_net_output_shape + 1 + len(list_of_monomial_identifiers) , 'Poly Shape ' + str(polynomial_list.shape[1]) +' Output Monomials ' +  str(interpretation_net_output_shape) + str(polynomial_list[:2])
    assert lambda_input_list.shape[2] == n
                
    config = {'list_of_monomial_identifiers': list_of_monomial_identifiers, 
              'interpretation_net_output_monomials': interpretation_net_output_monomials}
        
    parallel = Parallel(n_jobs=n_jobs_parallel_fv, verbose=1, backend=backend)
    #polynomial_true_fv = parallel(delayed(calculate_function_values_from_polynomial)(polynomial, lambda_inputs, force_complete_poly_representation=force_complete_poly_representation, list_of_monomial_identifiers=list_of_monomial_identifiers, interpretation_net_output_monomials=interpretation_net_output_monomials) for polynomial, lambda_inputs in zip(polynomial_list, lambda_input_list))  
    
    config = {
         'n': n,
         #'inet_loss': inet_loss,
         'sparsity': sparsity,
         #'lambda_network_layers': lambda_network_layers,
         #'interpretation_net_output_shape': interpretation_net_output_shape,
         'RANDOM_SEED': RANDOM_SEED,
         #'nas': nas,
         #'number_of_lambda_weights': number_of_lambda_weights,
         #'interpretation_net_output_monomials': interpretation_net_output_monomials,
         #'list_of_monomial_identifiers': list_of_monomial_identifiers,
         'x_min': x_min,
         'x_max': x_max,
         'sparse_poly_representation_version': sparse_poly_representation_version,
        }
    
    try:
        config['interpretation_net_output_monomials'] = interpretation_net_output_monomials
    except:
        config['interpretation_net_output_monomials'] = None
       
    if use_gpu:
        polynomial_true_fv = parallel(delayed(calculate_poly_fv_tf_wrapper_new_no_tfFunction)(return_float_tensor_representation(list_of_monomial_identifiers), return_float_tensor_representation(polynomial), return_float_tensor_representation(lambda_inputs), force_complete_poly_representation=force_complete_poly_representation, config=config) for polynomial, lambda_inputs in zip(polynomial_list, lambda_input_list))      
    else:
        polynomial_true_fv = parallel(delayed(calculate_poly_fv_tf_wrapper_new)(return_float_tensor_representation(list_of_monomial_identifiers), return_float_tensor_representation(polynomial), return_float_tensor_representation(lambda_inputs), force_complete_poly_representation=force_complete_poly_representation, config=config) for polynomial, lambda_inputs in zip(polynomial_list, lambda_input_list))  
    
    del parallel   
    
    
    return np.array(polynomial_true_fv)


def calculate_function_values_from_sympy(function, data_points, variable_names=None):
    
    if function is None:
        return np.array([np.nan for i in range(data_points.shape[0])])
    try:
        if variable_names == None:
            function_vars = function.atoms(Symbol)
        else:
            function_vars = [sym.symbols(variable_name) for variable_name in variable_names]
        #print('function_vars', function_vars)
        lambda_function = lambdify([function_vars], function, modules=["scipy", "numpy"])
        #print('lambda_function', lambda_function)
        #print('data_points[0]', data_points[0])
        if len(function_vars) >= 1:
            function_values = [lambda_function(data_point) for data_point in data_points]
            
        else:
            function_values = [lambda_function() for i in range(data_points.shape[0])]
    except (NameError, KeyError) as e:
        #print(e)
        function_values = []
        for data_point in data_points:
            function_value = function.evalf(subs={var: data_point[index] for index, var in enumerate(list(function_vars))})
            try:
                function_value = float(function_value)
            except TypeError as te:
                #print(te)
                #print(function_value)
                function_value = np.inf
            function_values.append(function_value)
    function_values = np.nan_to_num(function_values).ravel()
                
    return function_values



def parallel_fv_calculation_from_sympy(function_list, lambda_input_list, n_jobs_parallel_fv=10, backend='threading', variable_names=None):
    backend='sequential'            
    lambda_input_list = return_numpy_representation(lambda_input_list)
    
    assert len(function_list) == lambda_input_list.shape[0], str(len(function_list)) + '\t' +  str(lambda_input_list.shape[0])
             
    parallel = Parallel(n_jobs=n_jobs_parallel_fv, verbose=1, backend=backend)
    polynomial_true_fv = parallel(delayed(calculate_function_values_from_sympy)(function, lambda_inputs, variable_names=variable_names) for function, lambda_inputs in zip(function_list, lambda_input_list))  
    del parallel   
    

    return np.array(polynomial_true_fv)



def sleep_minutes(minutes):
    time.sleep(int(60*minutes))
    
def sleep_hours(hours):
    time.sleep(int(60*60*hours))
    
    
def generate_paths(config=None, path_type='interpretation_net'):
    
    if config is not None:

        paths_dict = {}
        
        training_string = '_sameX' if config['data']['same_training_all_lambda_nets'] else '_diffX'

        laurent_str = '_laurent' if config['data']['laurent'] else ''
        monomial_vars_str = '_monvars_' + str(config['data']['monomial_vars']) if config['data']['monomial_vars'] != None else ''
        neg_d_str = '_negd_' + str(config['data']['neg_d']) + '_prob_' + str(config['data']['neg_d_prob']) if config['data']['neg_d'] != None else ''


        dataset_description_string = ('_var_' + str(config['data']['n']) + 
                                      '_d_' + str(config['data']['d']) + 
                                       laurent_str + 
                                       monomial_vars_str + 
                                       neg_d_str + 
                                       '_spars_' + str(config['data']['sample_sparsity']) + 
                                       '_amin_' + str(config['data']['a_min']) + 
                                       '_amax_' + str(config['data']['a_max']) + 
                                       #'_xmin_' + str(x_min) + 
                                       #'_xmax_' + str(x_max) + 
                                       '_xdist_' + str(config['data']['x_distrib']) + 
                                       '_noise_' + str(config['data']['noise_distrib']) + '_' + str(config['data']['noise']) 
                                       + '_' + config['data']['function_generation_type']
                                     )

        if config['data']['shift_polynomial']:         
            adjusted_dataset_string = ('bmin' + str(config['data']['border_min']) +
                                        'bmax' + str(config['data']['border_max']) +
                                        'lowd' + str(config['data']['lower_degree_prob']) +
                                        'arand' + str(config['data']['a_random_prob']))
        else:
            adjusted_dataset_string = ''



        if path_type == 'data_creation' or path_type == 'lambda_net': #Data Generation

            path_identifier_polynomial_data = ('poly_' + str(config['data']['polynomial_data_size']) + 
                                               '_train_' + str(config['lambda_net']['lambda_dataset_size']) + 
                                               dataset_description_string + 
                                               adjusted_dataset_string +
                                               training_string)            

            paths_dict['path_identifier_polynomial_data'] = path_identifier_polynomial_data

        if path_type == 'lambda_net' or path_type == 'interpretation_net': #Lambda-Net

            if config['data']['fixed_seed_lambda_training'] and config['data']['fixed_initialization_lambda_training']:
                seed_init_string = '_' + str(config['data']['number_different_lambda_trainings']) + '-FixSeedInit'
            elif config['data']['fixed_seed_lambda_training'] and not config['data']['fixed_initialization_lambda_training']:
                seed_init_string = '_' + str(config['data']['number_different_lambda_trainings']) + '-FixSeed'
            elif not config['data']['fixed_seed_lambda_training'] and config['data']['fixed_initialization_lambda_training']:
                seed_init_string = '_' + str(config['data']['number_different_lambda_trainings']) + '-FixInit'
            elif not config['data']['fixed_seed_lambda_training'] and not config['data']['fixed_initialization_lambda_training']:            
                seed_init_string = '_NoFixSeedInit'


            early_stopping_string = '_ES' + str(config['lambda_net']['early_stopping_min_delta_lambda']) + '_' if config['lambda_net']['early_stopping_lambda'] else ''

            lambda_layer_str = ''.join([str(neurons) + '-' for neurons in config['lambda_net']['lambda_network_layers']])
            lambda_net_identifier = '_' + lambda_layer_str + str(config['lambda_net']['epochs_lambda']) + 'e' + early_stopping_string + str(config['lambda_net']['batch_lambda']) + 'b' + '_' + config['lambda_net']['optimizer_lambda'] + '_' + config['lambda_net']['loss_lambda']

            path_identifier_lambda_net_data = ('lnets_' + str(config['data']['lambda_nets_total']) +
                                               lambda_net_identifier + 
                                               '_train_' + str(lambda_dataset_size) + 
                                               training_string + 
                                               seed_init_string + '_' + str(config['computation']['RANDOM_SEED']) +
                                               '/' +
                                               dataset_description_string[1:] + 
                                               adjusted_dataset_string)        

            paths_dict['path_identifier_lambda_net_data'] = path_identifier_lambda_net_data


        if path_type == 'interpretation_net': #Interpretation-Net   

            interpretation_network_layers_string = 'dense' + ''.join([str(neurons) + '-' for neurons in config['i_net']['dense_layers']])

            if config['i_net']['convolution_layers'] != None:
                interpretation_network_layers_string += 'conv' + str(config['i_net']['convolution_layers'])
            if config['i_net']['lstm_layers'] != None:
                interpretation_network_layers_string += 'lstm' + str(config['i_net']['lstm_layers'])

            interpretation_net_identifier = '_' + interpretation_network_layers_string + 'output_' + str(config['i_net']['interpretation_net_output_shape']) + '_drop' + str(config['i_net']['dropout']) + 'e' + str(config['i_net']['epochs']) + 'b' + str(config['i_net']['batch_size']) + '_' + config['i_net']['optimizer']

            path_identifier_interpretation_net_data = ('inet' + interpretation_net_identifier +
                                                       '/lnets_' + str(config['i_net']['interpretation_dataset_size']) +
                                                       lambda_net_identifier + 
                                                       '_train_' + str(config['lambda_net']['lambda_dataset_size']) + 
                                                       training_string + 
                                                       seed_init_string + '_' + str(config['computation']['RANDOM_SEED']) +
                                                       '/' +
                                                       dataset_description_string[1:] + 
                                                       adjusted_dataset_string)       


            paths_dict['path_identifier_interpretation_net_data'] = path_identifier_interpretation_net_data
            
        
        
        
    else:

        paths_dict = {}

        training_string = '_sameX' if same_training_all_lambda_nets else '_diffX'

        laurent_str = '_laurent' if laurent else ''
        monomial_vars_str = '_monvars_' + str(monomial_vars) if monomial_vars != None else ''
        neg_d_str = '_negd_' + str(neg_d) + '_prob_' + str(neg_d_prob) if neg_d != None else ''


        dataset_description_string = ('_var_' + str(n) + 
                                      '_d_' + str(d) + 
                                       laurent_str + 
                                       monomial_vars_str + 
                                       neg_d_str + 
                                       '_spars_' + str(sample_sparsity) + 
                                       '_amin_' + str(a_min) + 
                                       '_amax_' + str(a_max) + 
                                       #'_xmin_' + str(x_min) + 
                                       #'_xmax_' + str(x_max) + 
                                       '_xdist_' + str(x_distrib) + 
                                       '_noise_' + str(noise_distrib) + '_' + str(noise) 
                                       + '_' + function_generation_type
                                     )

        if shift_polynomial:         
            adjusted_dataset_string = ('bmin' + str(border_min) +
                                        'bmax' + str(border_max) +
                                        'lowd' + str(lower_degree_prob) +
                                        'arand' + str(a_random_prob))
        else:
            adjusted_dataset_string = ''



        if path_type == 'data_creation' or path_type == 'lambda_net': #Data Generation

            path_identifier_polynomial_data = ('poly_' + str(polynomial_data_size) + 
                                               '_train_' + str(lambda_dataset_size) + 
                                               dataset_description_string + 
                                               adjusted_dataset_string +
                                               training_string)            

            paths_dict['path_identifier_polynomial_data'] = path_identifier_polynomial_data

        if path_type == 'lambda_net' or path_type == 'interpretation_net': #Lambda-Net

            if fixed_seed_lambda_training and fixed_initialization_lambda_training:
                seed_init_string = '_' + str(number_different_lambda_trainings) + '-FixSeedInit'
            elif fixed_seed_lambda_training and not fixed_initialization_lambda_training:
                seed_init_string = '_' + str(number_different_lambda_trainings) + '-FixSeed'
            elif not fixed_seed_lambda_training and fixed_initialization_lambda_training:
                seed_init_string = '_' + str(number_different_lambda_trainings) + '-FixInit'
            elif not fixed_seed_lambda_training and not fixed_initialization_lambda_training:            
                seed_init_string = '_NoFixSeedInit'


            early_stopping_string = '_ES' + str(early_stopping_min_delta_lambda) + '_' if early_stopping_lambda else ''

            lambda_layer_str = ''.join([str(neurons) + '-' for neurons in lambda_network_layers])
            lambda_net_identifier = '_' + lambda_layer_str + str(epochs_lambda) + 'e' + early_stopping_string + str(batch_lambda) + 'b' + '_' + optimizer_lambda + '_' + loss_lambda

            path_identifier_lambda_net_data = ('lnets_' + str(lambda_nets_total) +
                                               lambda_net_identifier + 
                                               '_train_' + str(lambda_dataset_size) + 
                                               training_string + 
                                               seed_init_string + '_' + str(RANDOM_SEED) +
                                               '/' +
                                               dataset_description_string[1:] + 
                                               adjusted_dataset_string)        

            paths_dict['path_identifier_lambda_net_data'] = path_identifier_lambda_net_data


        if path_type == 'interpretation_net': #Interpretation-Net   

            interpretation_network_layers_string = 'dense' + ''.join([str(neurons) + '-' for neurons in dense_layers])

            if convolution_layers != None:
                interpretation_network_layers_string += 'conv' + str(convolution_layers)
            if lstm_layers != None:
                interpretation_network_layers_string += 'lstm' + str(lstm_layers)

            interpretation_net_identifier = '_' + interpretation_network_layers_string + 'output_' + str(interpretation_net_output_shape) + '_drop' + str(dropout) + 'e' + str(epochs) + 'b' + str(batch_size) + '_' + optimizer

            path_identifier_interpretation_net_data = ('inet' + interpretation_net_identifier +
                                                       '/lnets_' + str(interpretation_dataset_size) +
                                                       lambda_net_identifier + 
                                                       '_train_' + str(lambda_dataset_size) + 
                                                       training_string + 
                                                       seed_init_string + '_' + str(RANDOM_SEED) +
                                                       '/' +
                                                       dataset_description_string[1:] + 
                                                       adjusted_dataset_string)       


            paths_dict['path_identifier_interpretation_net_data'] = path_identifier_interpretation_net_data
        
    return paths_dict
    
def create_folders_inet():
    
    paths_dict = generate_paths(path_type = 'interpretation_net')
    
    try:
        # Create target Directory
        os.makedirs('./data/plotting/' + paths_dict['path_identifier_interpretation_net_data'] + '/')
        os.makedirs('./data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/')
    except FileExistsError:
        pass
    

def generate_directory_structure():
    
    directory_names = ['plotting', 'saved_polynomial_lists', 'results', 'saved_models', 'weights']
    if not os.path.exists('./data'):
        os.makedirs('./data')
        
        text_file = open('./data/.gitignore', 'w')
        text_file.write('*')
        text_file.close()  
        
    for directory_name in directory_names:
        path = './data/' + directory_name
        if not os.path.exists(path):
            os.makedirs(path)
            
            
def generate_lambda_net_directory():
    
    paths_dict = generate_paths(path_type = 'lambda_net')
    
    #clear files
    try:
        # Create target Directory
        os.makedirs('./data/weights/weights_' + paths_dict['path_identifier_lambda_net_data'])

    except FileExistsError:
        folder = './data/weights/weights_' + paths_dict['path_identifier_lambda_net_data']
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e)) 
    try:
        # Create target Directory
        os.makedirs('./data/results/weights_' + paths_dict['path_identifier_lambda_net_data'])
    except FileExistsError:
        pass
    
    
######################################################################################################################################################################################################################
########################################################################################  RANDOM FUNCTION GENERATION FROM ############################################################################################ 
################################# code adjusted, originally from: https://github.com/tirthajyoti/Machine-Learning-with-Python/tree/master/Random%20Function%20Generator ##############################################
######################################################################################################################################################################################################################

def symbolize(s):
    """
    Converts a a string (equation) to a SymPy symbol object
    """
        
    s1=s.replace(',','.')
    s2=s1.replace('^','**')
    s3=sympify(s2)
    
    return(s3)

def eval_multinomial(s,vals=None,symbolic_eval=False):
    """
    Evaluates polynomial at vals.
    vals can be simple list, dictionary, or tuple of values.
    vals can also contain symbols instead of real values provided those symbols have been declared before using SymPy
    """
    sym_s=symbolize(s)
    sym_set=sym_s.atoms(Symbol)
    sym_lst=[]

    
    for s in sym_set:
        sym_lst.append(str(s))
    sym_lst.sort()
    if symbolic_eval==False and len(sym_set)!=len(vals):
        print("Length of the input values did not match number of variables and symbolic evaluation is not selected")
        return None
    else:
        if type(vals)==list:
            sub=list(zip(sym_lst,vals))
        elif type(vals)==dict:
            l=list(vals.keys())
            l.sort()
            lst=[]
            for i in l:
                lst.append(vals[i])
            sub=list(zip(sym_lst,lst))
        elif type(vals)==tuple:
            sub=list(zip(sym_lst,list(vals)))
        result=sym_s.subs(sub)
    
    return result

def flip(y,p):
    lst=[]
    for i in range(len(y)):
        f=np.random.choice([1,0],p=[p,1-p])
        lst.append(f)
    lst=np.array(lst)
    return np.array(np.logical_xor(y,lst),dtype=int)

@tf.function
def calculate_poly_fv_tf_wrapper_new_no_tfFunction(list_of_monomial_identifiers, polynomial, evaluation_entry_list, force_complete_poly_representation=False, config=None):
    
    if config != None:
        globals().update(config) 
    def calculate_poly_fv_tf(evaluation_entry):  
        
        
        def calculate_monomial_with_coefficient_degree_by_var_wrapper(evaluation_entry):
            def calculate_monomial_with_coefficient_degree_by_var(input_list):     
                
                degree_by_var_per_monomial = input_list[0]
                coefficient = input_list[1]
                
                #degree_by_var_per_monomial = gewählter degree für jede variable in monomial
                monomial_value_without_coefficient = tf.math.reduce_prod(tf.vectorized_map(lambda x: x[0]**tf.dtypes.cast(x[1], tf.float32), (evaluation_entry, degree_by_var_per_monomial)))

                return coefficient*monomial_value_without_coefficient
            return calculate_monomial_with_coefficient_degree_by_var
        
        
        if interpretation_net_output_monomials == None or force_complete_poly_representation:
            monomials_without_coefficient = tf.vectorized_map(calculate_monomial_without_coefficient_tf_wrapper(evaluation_entry), (list_of_monomial_identifiers))      
            monomial_values = tf.vectorized_map(lambda x: x[0]*x[1], (monomials_without_coefficient, polynomial))
        else: 
            if sparse_poly_representation_version == 1:
                monomials_without_coefficient = tf.vectorized_map(calculate_monomial_without_coefficient_tf_wrapper(evaluation_entry), (list_of_monomial_identifiers))      
                
                coefficients = polynomial[:interpretation_net_output_monomials]
                index_array = polynomial[interpretation_net_output_monomials:]

                assert index_array.shape[0] == interpretation_net_output_monomials*sparsity, 'Shape of Coefficient Indices : ' + str(index_array.shape)

                index_list = tf.split(index_array, interpretation_net_output_monomials)

                assert len(index_list) == coefficients.shape[0] == interpretation_net_output_monomials, 'Shape of Coefficient Indices Split: ' + str(len(index_list))

                indices = tf.argmax(index_list, axis=1) 

                monomial_values = tf.vectorized_map(lambda x: tf.gather(monomials_without_coefficient, x[0])*x[1], (indices, coefficients)) 
            elif sparse_poly_representation_version == 2:
                coefficients = polynomial[:interpretation_net_output_monomials]
                index_array = polynomial[interpretation_net_output_monomials:]
                #tf.print('index_array.shape', index_array)
                
                assert index_array.shape[0] == interpretation_net_output_monomials*n*(d+1), 'Shape of Coefficient Indices : ' + str(index_array.shape)

                if False:
                    index_list_by_monomial = tf.split(index_array, n)

                    assert len(index_list_by_monomial) == coefficients.shape[0] == interpretation_net_output_monomials, 'Shape of Coefficient Indices Split: ' + str(len(index_list))

                    index_list_by_monomial_by_var = tf.split(index_list_by_monomial, d+1, axis=1)
                    degree_by_var_per_monomial_list = tf.argmax(index_list_by_monomial_by_var, axis=2) 
                else:
                    index_list_by_monomial = tf.transpose(tf.split(index_array, interpretation_net_output_monomials))

                    index_list_by_monomial_by_var = tf.split(index_list_by_monomial, n, axis=0)
                    index_list_by_monomial_by_var_new = []
                    for tensor in index_list_by_monomial_by_var:
                        index_list_by_monomial_by_var_new.append(tf.transpose(tensor))
                    index_list_by_monomial_by_var = index_list_by_monomial_by_var_new   
                    #tf.print('index_list_by_monomial_by_var', index_list_by_monomial_by_var)
                    degree_by_var_per_monomial_list = tf.transpose(tf.argmax(index_list_by_monomial_by_var, axis=2))                  
                
                #tf.print('degree_by_var_per_monomial_list', degree_by_var_per_monomial_list)
                #tf.print('evaluation_entry', evaluation_entry)
                #tf.print('coefficients', coefficients)

                monomial_values = tf.vectorized_map(calculate_monomial_with_coefficient_degree_by_var_wrapper(evaluation_entry), (degree_by_var_per_monomial_list, coefficients))                 
                #tf.print('monomial_values', monomial_values)
            
        polynomial_fv = tf.reduce_sum(monomial_values)    
        #tf.print(polynomial_fv)

        return polynomial_fv
            
    return tf.vectorized_map(calculate_poly_fv_tf, (evaluation_entry_list))



@tf.function(jit_compile=True)
def calculate_poly_fv_tf_wrapper_new(list_of_monomial_identifiers, polynomial, evaluation_entry_list, force_complete_poly_representation=False, config=None):
    
    if config != None:
        globals().update(config) 
    def calculate_poly_fv_tf(evaluation_entry):  
        
        
        def calculate_monomial_with_coefficient_degree_by_var_wrapper(evaluation_entry):
            def calculate_monomial_with_coefficient_degree_by_var(input_list):     
                
                degree_by_var_per_monomial = input_list[0]
                coefficient = input_list[1]
                
                #degree_by_var_per_monomial = gewählter degree für jede variable in monomial
                monomial_value_without_coefficient = tf.math.reduce_prod(tf.vectorized_map(lambda x: x[0]**tf.dtypes.cast(x[1], tf.float32), (evaluation_entry, degree_by_var_per_monomial)))

                return coefficient*monomial_value_without_coefficient
            return calculate_monomial_with_coefficient_degree_by_var
        
        
        if interpretation_net_output_monomials == None or force_complete_poly_representation:
            monomials_without_coefficient = tf.vectorized_map(calculate_monomial_without_coefficient_tf_wrapper(evaluation_entry), (list_of_monomial_identifiers))      
            monomial_values = tf.vectorized_map(lambda x: x[0]*x[1], (monomials_without_coefficient, polynomial))
        else: 
            if sparse_poly_representation_version == 1:
                monomials_without_coefficient = tf.vectorized_map(calculate_monomial_without_coefficient_tf_wrapper(evaluation_entry), (list_of_monomial_identifiers))      
                
                coefficients = polynomial[:interpretation_net_output_monomials]
                index_array = polynomial[interpretation_net_output_monomials:]

                assert index_array.shape[0] == interpretation_net_output_monomials*sparsity, 'Shape of Coefficient Indices : ' + str(index_array.shape)

                index_list = tf.split(index_array, interpretation_net_output_monomials)

                assert len(index_list) == coefficients.shape[0] == interpretation_net_output_monomials, 'Shape of Coefficient Indices Split: ' + str(len(index_list))

                indices = tf.argmax(index_list, axis=1) 

                monomial_values = tf.vectorized_map(lambda x: tf.gather(monomials_without_coefficient, x[0])*x[1], (indices, coefficients)) 
            elif sparse_poly_representation_version == 2:
                coefficients = polynomial[:interpretation_net_output_monomials]
                index_array = polynomial[interpretation_net_output_monomials:]
                #tf.print('index_array.shape', index_array)
                
                assert index_array.shape[0] == interpretation_net_output_monomials*n*(d+1), 'Shape of Coefficient Indices : ' + str(index_array.shape)

                if False:
                    index_list_by_monomial = tf.split(index_array, n)

                    assert len(index_list_by_monomial) == coefficients.shape[0] == interpretation_net_output_monomials, 'Shape of Coefficient Indices Split: ' + str(len(index_list))

                    index_list_by_monomial_by_var = tf.split(index_list_by_monomial, d+1, axis=1)
                    degree_by_var_per_monomial_list = tf.argmax(index_list_by_monomial_by_var, axis=2) 
                else:
                    index_list_by_monomial = tf.transpose(tf.split(index_array, interpretation_net_output_monomials))

                    index_list_by_monomial_by_var = tf.split(index_list_by_monomial, n, axis=0)
                    index_list_by_monomial_by_var_new = []
                    for tensor in index_list_by_monomial_by_var:
                        index_list_by_monomial_by_var_new.append(tf.transpose(tensor))
                    index_list_by_monomial_by_var = index_list_by_monomial_by_var_new   
                    #tf.print('index_list_by_monomial_by_var', index_list_by_monomial_by_var)
                    degree_by_var_per_monomial_list = tf.transpose(tf.argmax(index_list_by_monomial_by_var, axis=2))                  
                
                #tf.print('degree_by_var_per_monomial_list', degree_by_var_per_monomial_list)
                #tf.print('evaluation_entry', evaluation_entry)
                #tf.print('coefficients', coefficients)

                monomial_values = tf.vectorized_map(calculate_monomial_with_coefficient_degree_by_var_wrapper(evaluation_entry), (degree_by_var_per_monomial_list, coefficients))                 
                #tf.print('monomial_values', monomial_values)
            
        polynomial_fv = tf.reduce_sum(monomial_values)    
        #tf.print(polynomial_fv)

        return polynomial_fv
            
    return tf.vectorized_map(calculate_poly_fv_tf, (evaluation_entry_list))


#calculate intermediate term (without coefficient multiplication)
def calculate_monomial_without_coefficient_tf_wrapper(evaluation_entry):
    def calculate_monomial_without_coefficient_tf(coefficient_multiplier_term): 
        return tf.math.reduce_prod(tf.vectorized_map(lambda x: x[0]**x[1], (evaluation_entry, coefficient_multiplier_term)))
    return calculate_monomial_without_coefficient_tf



def gen_regression_symbolic(polynomial_array=None,
                            n_samples=100,
                            noise=0.0, 
                            noise_dist='normal', 
                            seed=42, 
                            sympy_calculation=True):
    
    from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3
    from sklearn.preprocessing import MinMaxScaler    
        
    np.random.seed(seed)
    random.seed(seed)
        
    if function_generation_type == 'polynomial':

        if polynomial_array is not None:
            sympy_string = get_sympy_string_from_coefficients(polynomial_array)
            sympy_function=sympify(sympy_string)

        if polynomial_array is None:
            sympy_function=''
            for i in range(1,n_features+1):
                c='x'+str(i)
                c+=np.random.choice(['+','-'],p=[0.5,0.5])
                sympy_function+=c
            sympy_function=sympy_function[:-1]

        n_features=len(sympy_function.atoms(Symbol))

        eval_results=[]

        eval_dataset = generate_random_data_points(low=x_min, high=x_max, size=n_samples, variables=max(1, n), distrib=x_distrib)


        config = {'list_of_monomial_identifiers': list_of_monomial_identifiers, 
                  'interpretation_net_output_monomials': interpretation_net_output_monomials}

        if sympy_calculation:
            for i in range(n_samples):
                eval_results.append(eval_multinomial(sympy_string, vals=list(eval_dataset[i])))
        elif not sympy_calculation and polynomial_array is not None:
            config = {
                 'n': n,
                 #'inet_loss': inet_loss,
                 'sparsity': sparsity,
                 #'lambda_network_layers': lambda_network_layers,
                 #'interpretation_net_output_shape': interpretation_net_output_shape,
                 'RANDOM_SEED': RANDOM_SEED,
                 #'nas': nas,
                 #'number_of_lambda_weights': number_of_lambda_weights,
                 'interpretation_net_output_monomials': interpretation_net_output_monomials,
                 #'list_of_monomial_identifiers': list_of_monomial_identifiers,
                 'x_min': x_min,
                 'x_max': x_max,
                 }

        try:
            config['interpretation_net_output_monomials'] = interpretation_net_output_monomials
        except:
            config['interpretation_net_output_monomials'] = None        

        eval_results = calculate_poly_fv_tf_wrapper_new(return_float_tensor_representation(list_of_monomial_identifiers), return_float_tensor_representation(polynomial_array), return_float_tensor_representation(eval_dataset), force_complete_poly_representation=True, config=config)



        eval_results=np.array(eval_results)
        eval_results=eval_results.reshape(n_samples,1)

        if noise_dist=='normal':
            noise_sample=noise*np.random.normal(loc=0, scale=np.max(eval_results)-np.min(eval_results),size=n_samples)
        elif noise_dist=='uniform':
            noise_sample=noise*np.random.uniform(low=-(np.max(eval_results)-np.min(eval_results)), high=np.max(eval_results)-np.min(eval_results),size=n_samples)

        noise_sample=noise_sample.reshape(n_samples,1)

        eval_results=eval_results+noise_sample
        
    elif function_generation_type == 'friedman1':
        
        eval_dataset, eval_results = make_friedman1(n_samples=n_samples, 
                                                    n_features=n, 
                                                    noise=noise, 
                                                    random_state=seed)

        if True:
            eval_results = MinMaxScaler(feature_range=(0, random.uniform(1, sample_sparsity*a_max))).fit_transform(eval_results.reshape(-1, 1))        
                  
    
    return polynomial_array, eval_dataset, eval_results


def generate_random_data_points(low, high, size, variables, distrib='uniform'):
    if distrib=='normal':
        list_of_data_points = []
        for _ in range(size):
            random_data_points = np.random.normal(loc=(low+high)/2, scale=(low+high)/4, size=variables)
            while max(random_data_points) > high and min(random_data_points) < low:
                random_poly = np.random.normal(loc=(low+high)/2, scale=1.0, size= variables)
            list_of_data_points.append(random_poly)
        list_of_data_points = np.array(list_of_polynomials)
        
    elif distrib=='uniform':
        list_of_data_points = np.random.uniform(low=low, high=high, size=(size, variables))
        
    return list_of_data_points




######################################################################################################################################################################################################################
###########################################################################################  PER NETWORK OPTIMIZATION ################################################################################################ 
######################################################################################################################################################################################################################



def per_network_poly_optimization_tf(per_network_dataset_size, 
                                  lambda_network_weights, 
                                  list_of_monomial_identifiers_numbers, 
                                  config, 
                                  optimizer = tf.optimizers.Adam,
                                  lr=0.05, 
                                  max_steps = 1000, 
                                  early_stopping=10, 
                                  restarts=5, 
                                  printing=True,
                                  return_error=False):
    
    
    from utilities.metrics import calculate_poly_fv_tf_wrapper
    from utilities.metrics import r2_keras_loss

    ########################################### GENERATE RELEVANT PARAMETERS FOR OPTIMIZATION ########################################################
            
    globals().update(config)
        
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)    
    if int(tf.__version__[0]) >= 2:
        tf.random.set_seed(RANDOM_SEED)
    else:
        tf.set_random_seed(RANDOM_SEED)       
    

    base_model = Sequential()

    base_model.add(Dense(lambda_network_layers[0], activation='relu', input_dim=n))

    for neurons in lambda_network_layers[1:]:
        base_model.add(Dense(neurons, activation='relu'))

    base_model.add(Dense(1))
    
    weights_structure = base_model.get_weights()
    
    
    random_lambda_input_data = np.random.uniform(low=x_min, high=x_max, size=(per_network_dataset_size, max(1, n)))
    random_lambda_input_data = tf.dtypes.cast(tf.convert_to_tensor(random_lambda_input_data), tf.float32)
    list_of_monomial_identifiers_numbers = tf.dtypes.cast(tf.convert_to_tensor(list_of_monomial_identifiers_numbers), tf.float32)
    
    model_lambda_placeholder = tf.keras.models.clone_model(base_model)  
    
    dims = [np_arrays.shape for np_arrays in weights_structure]
    

    lambda_network_weights = tf.dtypes.cast(tf.convert_to_tensor(lambda_network_weights), tf.float32)
    
    #CALCULATE LAMBDA FV HERE FOR EVALUATION DATASET
    # build models
    start = 0
    layers = []
    for i in range(len(dims)//2):

        # set weights of layer
        index = i*2
        size = np.product(dims[index])
        weights_tf_true = tf.reshape(lambda_network_weights[start:start+size], dims[index])
        model_lambda_placeholder.layers[i].weights[0].assign(weights_tf_true)
        start += size

        # set biases of layer
        index += 1
        size = np.product(dims[index])
        biases_tf_true = tf.reshape(lambda_network_weights[start:start+size], dims[index])
        model_lambda_placeholder.layers[i].weights[1].assign(biases_tf_true)
        start += size


    lambda_fv = tf.keras.backend.flatten(model_lambda_placeholder(random_lambda_input_data))    
    

    
    ########################################### OPTIMIZATION ########################################################
        
    current_monomial_degree = tf.Variable(0, dtype=tf.int64)
    best_result = np.inf
    
    for current_iteration in range(restarts):
                
        @tf.function(jit_compile=True) 
        def function_to_optimize():
            
            poly_optimize = poly_optimize_input[0]

            if interpretation_net_output_monomials != None:
                poly_optimize_coeffs = poly_optimize[:interpretation_net_output_monomials]
                poly_optimize_identifiers_list = []
                if sparse_poly_representation_version == 1:
                    for i in range(interpretation_net_output_monomials):
                        poly_optimize_identifiers = tf.math.softmax(poly_optimize[sparsity*i+interpretation_net_output_monomials:sparsity*(i+1)+interpretation_net_output_monomials])
                        poly_optimize_identifiers_list.append(poly_optimize_identifiers)
                    poly_optimize_identifiers_list = tf.keras.backend.flatten(poly_optimize_identifiers_list)
                elif sparse_poly_representation_version != 2:
                    for i in range(interpretation_net_output_monomials):
                        for j in range(n):
                            poly_optimize_identifiers = tf.math.softmax(poly_optimize[i*n*j*(d+1)+interpretation_net_output_monomials:(i+1)*n*j*(d+1)+interpretation_net_output_monomials])
                            poly_optimize_identifiers_list.append(poly_optimize_identifiers)
                    poly_optimize_identifiers_list = tf.keras.backend.flatten(poly_optimize_identifiers_list)                
                poly_optimize = tf.concat([poly_optimize_coeffs, poly_optimize_identifiers_list], axis=0)

            poly_optimize_fv_list = tf.vectorized_map(calculate_poly_fv_tf_wrapper(list_of_monomial_identifiers_numbers, poly_optimize, current_monomial_degree, config=config), (random_lambda_input_data))

            error = None
            if inet_loss == 'mae':
                error = tf.keras.losses.MAE(lambda_fv, poly_optimize_fv_list)
            elif inet_loss == 'r2':
                error = r2_keras_loss(lambda_fv, poly_optimize_fv_list)  
            else:
                raise SystemExit('Unknown I-Net Metric: ' + inet_loss)                

            error = tf.where(tf.math.is_nan(error), tf.fill(tf.shape(error), np.inf), error)   

            return error 

    
            
        opt = optimizer(learning_rate=lr)
        
        poly_optimize_input = tf.Variable(tf.random.uniform([1, interpretation_net_output_shape]))
        
        stop_counter = 0
        best_result_iteration = np.inf

        for current_step in range(max_steps):
            if stop_counter>=early_stopping:
                break
            
            opt.minimize(function_to_optimize, var_list=[poly_optimize_input])
            current_result = function_to_optimize()
            if printing:
                clear_output(wait=True)
                print("Current best: {} \n Curr_res: {} \n Iteration {}, Step {}".format(best_result_iteration,current_result, current_iteration, current_step))
 
            stop_counter += 1
            if current_result < best_result_iteration:
                best_result_iteration = current_result
                stop_counter = 0
                best_poly_optimize_iteration = tf.identity(poly_optimize_input)
                
        if best_result_iteration < best_result:
            best_result = best_result_iteration
            best_poly_optimize = tf.identity(best_poly_optimize_iteration)
            

    per_network_poly = best_poly_optimize[0].numpy()
    
    if printing:
        print("Optimization terminated at {}".format(best_result))
        
    if return_error:
        return best_result, per_network_poly
    
    return per_network_poly



def per_network_poly_optimization_scipy(per_network_dataset_size, 
                                          lambda_network_weights, 
                                          list_of_monomial_identifiers_numbers, 
                                          config, 
                                          optimizer = 'Nelder-Mead',
                                          jac = None,
                                          max_steps = 1000, 
                                          restarts=5, 
                                          printing=True,
                                          return_error=False):

    from utilities.metrics import calculate_poly_fv_tf_wrapper

    def copy( self ):
        return tf.identity(self)
    tf.Tensor.copy = copy


    ########################################### GENERATE RELEVANT PARAMETERS FOR OPTIMIZATION ########################################################

    globals().update(config)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)    
    if int(tf.__version__[0]) >= 2:
        tf.random.set_seed(RANDOM_SEED)
    else:
        tf.set_random_seed(RANDOM_SEED)       


    base_model = Sequential()

    base_model.add(Dense(lambda_network_layers[0], activation='relu', input_dim=n))

    for neurons in lambda_network_layers[1:]:
        base_model.add(Dense(neurons, activation='relu'))

    base_model.add(Dense(1))

    weights_structure = base_model.get_weights()


    random_lambda_input_data = np.random.uniform(low=x_min, high=x_max, size=(per_network_dataset_size, max(1, n)))
    random_lambda_input_data = tf.dtypes.cast(tf.convert_to_tensor(random_lambda_input_data), tf.float32)
    list_of_monomial_identifiers_numbers = tf.dtypes.cast(tf.convert_to_tensor(list_of_monomial_identifiers_numbers), tf.float32)

    model_lambda_placeholder = tf.keras.models.clone_model(base_model)  

    dims = [np_arrays.shape for np_arrays in weights_structure]


    lambda_network_weights = tf.dtypes.cast(tf.convert_to_tensor(lambda_network_weights), tf.float32)

    #CALCULATE LAMBDA FV HERE FOR EVALUATION DATASET
    # build models
    start = 0
    layers = []
    for i in range(len(dims)//2):

        # set weights of layer
        index = i*2
        size = np.product(dims[index])
        weights_tf_true = tf.reshape(lambda_network_weights[start:start+size], dims[index])
        model_lambda_placeholder.layers[i].weights[0].assign(weights_tf_true)
        start += size

        # set biases of layer
        index += 1
        size = np.product(dims[index])
        biases_tf_true = tf.reshape(lambda_network_weights[start:start+size], dims[index])
        model_lambda_placeholder.layers[i].weights[1].assign(biases_tf_true)
        start += size


    lambda_fv = tf.keras.backend.flatten(model_lambda_placeholder(random_lambda_input_data))    



    ########################################### OPTIMIZATION ########################################################

    current_monomial_degree = tf.Variable(0, dtype=tf.int64)
    best_result = np.inf

    for current_iteration in range(restarts):

        def function_to_optimize_scipy_wrapper(current_monomial_degree):
            @tf.function(jit_compile=True) 
            def function_to_optimize_scipy(poly_optimize_input):   

                #poly_optimize = tf.cast(tf.constant(poly_optimize_input), tf.float32)
                poly_optimize = tf.cast(poly_optimize_input, tf.float32)

                if interpretation_net_output_monomials != None:
                    poly_optimize_coeffs = poly_optimize[:interpretation_net_output_monomials]
                    poly_optimize_identifiers_list = []
                    if sparse_poly_representation_version == 1:
                        for i in range(interpretation_net_output_monomials):
                            poly_optimize_identifiers = tf.math.softmax(poly_optimize[sparsity*i+interpretation_net_output_monomials:sparsity*(i+1)+interpretation_net_output_monomials])
                            poly_optimize_identifiers_list.append(poly_optimize_identifiers)
                        poly_optimize_identifiers_list = tf.keras.backend.flatten(poly_optimize_identifiers_list)
                    else:
                        for i in range(interpretation_net_output_monomials):
                            for j in range(n):
                                poly_optimize_identifiers = tf.math.softmax(poly_optimize[i*n*j*(d+1)+interpretation_net_output_monomials:(i+1)*n*j*(d+1)+interpretation_net_output_monomials])
                                poly_optimize_identifiers_list.append(poly_optimize_identifiers)
                        poly_optimize_identifiers_list = tf.keras.backend.flatten(poly_optimize_identifiers_list)                
                    poly_optimize = tf.concat([poly_optimize_coeffs, poly_optimize_identifiers_list], axis=0)

                poly_optimize_fv_list = tf.vectorized_map(calculate_poly_fv_tf_wrapper(list_of_monomial_identifiers_numbers, poly_optimize, current_monomial_degree, config=config), (random_lambda_input_data))

                error = None
                if inet_loss == 'mae':
                    error = tf.keras.losses.MAE(lambda_fv, poly_optimize_fv_list)
                elif inet_loss == 'r2':
                    error = r2_keras_loss(lambda_fv, poly_optimize_fv_list)  
                else:
                    raise SystemExit('Unknown I-Net Metric: ' + inet_loss)                

                error = tf.where(tf.math.is_nan(error), tf.fill(tf.shape(error), np.inf), error)   

                return error
            return function_to_optimize_scipy


        poly_optimize_input = tf.random.uniform([1, interpretation_net_output_shape])    

        def function_to_optimize_scipy_grad_wrapper(current_monomial_degree):
            def function_to_optimize_scipy_grad(poly_optimize_input):

                error = function_to_optimize_scipy_wrapper(current_monomial_degree)(poly_optimize_input)
                error = error.numpy()
                return error
            return function_to_optimize_scipy_grad

        stop_counter = 0


        if jac=='fprime':
            jac = lambda x: optimize.approx_fprime(x, function_to_optimize_scipy_grad_wrapper(current_monomial_degree), 0.01)

        #tf.print(interpretation_net_output_monomials)
        #tf.print(config)        
        opt_res = minimize(function_to_optimize_scipy_wrapper(current_monomial_degree), poly_optimize_input, method=optimizer, jac=jac, options={'maxfun': None, 'maxiter': max_steps})
        print(opt_res)
        #opt_res = minimize(function_to_optimize_scipy_wrapper(current_monomial_degree), poly_optimize_input, method=optimizer, options={'maxfun': None, 'maxiter': max_steps})

        best_result_iteration = opt_res.fun
        best_poly_optimize_iteration = opt_res.x

        if best_result_iteration < best_result:
            best_result = best_result_iteration
            best_poly_optimize = best_poly_optimize_iteration

    per_network_poly = best_poly_optimize

    if printing:
        print("Optimization terminated at {}".format(best_result))

    if return_error:
        return best_result, per_network_poly
    
    return per_network_poly



def symbolic_regression(lambda_net, 
                          config,
                          symbolic_regression_hyperparams,
                          printing = True,
                          return_error = False):

    from pysymbolic_adjusted.algorithms.symbolic_expressions import symbolic_regressor
        
    globals().update(config) 
    
    
    if isinstance(lambda_net, tf.keras.Sequential):
        model = lambda_net
    else:
        model = lambda_net.return_model(config=config)
    
        
    try:
        
        symbolic_reg, r2_score, time_required   = symbolic_regressor(model, symbolic_regression_hyperparams['dataset_size'], [x_min, x_max], sample_sparsity, n_vars=config['n'], printing=printing, max_optimization_minutes=max_optimization_minutes)
    except MemoryError as e:
        print(e)
        print(traceback.print_exc())     
        if return_error:
            return np.nan, None, np.nan
        else:
            return None, np.nan         
            
    if return_error:
        return r2_score, symbolic_reg, time_required
    
    return symbolic_reg, time_required
      
    
    
 
    
def symbolic_metamodeling(lambda_net, 
                          config,
                          metamodeling_hyperparams,
                          printing = True,
                          return_error = False,
                          return_expression = 'approx', #'approx', #'exact',
                          function_metamodeling = False,
                          force_polynomial=False):
    
    
    
    from pysymbolic_adjusted.algorithms.symbolic_metamodeling import symbolic_metamodel
    from pysymbolic_adjusted.algorithms.symbolic_expressions import get_symbolic_model
    
        
    
    ########################################### GENERATE RELEVANT PARAMETERS FOR OPTIMIZATION ########################################################
            
    globals().update(config) 
    
    global x_min
    
    if isinstance(lambda_net, tf.keras.Sequential):
        model = lambda_net
    else:
        model = lambda_net.return_model(config=config)
    
    if x_min == 0:
        x_min = 1e-5
    
    
    ########################################### OPTIMIZATION ########################################################
    
    if function_metamodeling:    
        start = time.time()

        try: 
            with timeout(60*max_optimization_minutes, exception=RuntimeError): #in seconds
                symbolic_model, r2_score = get_symbolic_model(model, metamodeling_hyperparams['dataset_size'], [x_min, x_max], n_vars=config['n'])
                symbolic_model.approximation_order = d
                
        except (RuntimeError, AttributeError, MemoryError, ValueError) as e:
            print(e)
            print(traceback.print_exc())                   
                
            if return_error:
                return np.nan, None, np.nan
            else:
                return None, np.nan
            
        if return_expression == 'exact':
            metamodel_function = symbolic_model.exact_expression()
            #print(metamodel_function)
        elif return_expression == 'approx':
            metamodel_function = symbolic_model.approx_expression()       
            
        end = time.time()
        runtime = end-start
        if return_error:
            return r2_score, metamodel_function, runtime
            
    else:   
        random_lambda_input_data = np.random.uniform(low=x_min, high=x_max, size=(metamodeling_hyperparams['dataset_size'], max(1, n)))
        
        if metamodeling_hyperparams['batch_size'] == None:
            metamodeling_hyperparams['batch_size'] = random_lambda_input_data.shape[0]

        start = time.time()
        
        try: 
            with timeout(60*max_optimization_minutes, exception=RuntimeError): #in seconds
                metamodel = symbolic_metamodel(model, random_lambda_input_data, mode="regression", approximation_order = d, force_polynomial=force_polynomial)
                metamodel.fit(num_iter=metamodeling_hyperparams['num_iter'], batch_size=metamodeling_hyperparams['batch_size'], learning_rate=metamodeling_hyperparams['learning_rate'])    
        except (RuntimeError, AttributeError, MemoryError, ValueError) as e:
            print(e)
            print(traceback.print_exc())            
        
            if return_error:
                return np.nan, None, np.nan
            else:
                return None, np.nan
        

        if return_expression == 'exact':
            metamodel_function = metamodel.exact_expression
            #print(metamodel_function)
        elif return_expression == 'approx':
            metamodel_function = metamodel.approx_expression
            #print(metamodel_function)

        end = time.time()
        runtime = end-start
        
        if return_error:
            random_lambda_input_data_preds_metamodel = metamodel.evaluate(random_lambda_input_data)
            random_lambda_input_data_preds_lambda_net = model.predict(random_lambda_input_data)

            error = mean_absolute_error(random_lambda_input_data_preds_lambda_net, random_lambda_input_data_preds_metamodel)        

            return error, metamodel_function, runtime
    
    return metamodel_function, runtime




def symbolic_metamodeling_original(lambda_net, 
                          config,
                          metamodeling_hyperparams,
                          printing = True,
                          return_error = False,
                          return_expression = 'approx', #'approx', #'exact',
                          function_metamodeling = False,
                          force_polynomial=False):
    
    
    
    from pysymbolic_original.algorithms.symbolic_metamodeling import symbolic_metamodel
    from pysymbolic_original.algorithms.symbolic_expressions import get_symbolic_model
    
    ########################################### GENERATE RELEVANT PARAMETERS FOR OPTIMIZATION ########################################################
            
    globals().update(config) 
    
    global x_min
    
    if isinstance(lambda_net, tf.keras.Sequential):
        model = lambda_net
    else:
        model = lambda_net.return_model(config=config)
    
    if x_min == 0:
        x_min = 1e-5
    
    
    ########################################### OPTIMIZATION ########################################################
    if function_metamodeling:    
        start = time.time()
        
        try:
            with timeout(60*max_optimization_minutes, exception=RuntimeError): #in seconds
                symbolic_model, r2_score = get_symbolic_model(model, metamodeling_hyperparams['dataset_size'], [x_min, x_max])
                symbolic_model.approximation_order = d
                
        except (RuntimeError, AttributeError, MemoryError, ValueError) as e:
            print(e)
            print(traceback.print_exc())            
            
            if return_error:
                return np.nan, None, np.nan
            else:
                return None, np.nan
        
        
        if return_expression == 'exact':
            metamodel_function = symbolic_model.exact_expression()
            #print(metamodel_function)
        elif return_expression == 'approx':
            metamodel_function = symbolic_model.approx_expression()       
            
        end = time.time()
        runtime = end-start
        
        if return_error:
            return r2_score, metamodel_function, runtime
            
    else:   
        random_lambda_input_data = np.random.uniform(low=x_min, high=x_max, size=(metamodeling_hyperparams['dataset_size'], max(1, n)))
        
        if metamodeling_hyperparams['batch_size'] == None:
            metamodeling_hyperparams['batch_size'] = random_lambda_input_data.shape[0]
            
        start = time.time()
            
        try:
            with timeout(60*max_optimization_minutes, exception=RuntimeError): #in seconds
                metamodel = symbolic_metamodel(model, random_lambda_input_data, mode="regression", verbosity=printing)
                metamodel.fit(num_iter=metamodeling_hyperparams['num_iter'], 
                              batch_size=metamodeling_hyperparams['batch_size'], 
                              learning_rate=metamodeling_hyperparams['learning_rate'])    
        except (RuntimeError, AttributeError, MemoryError, ValueError) as e:
            print(e)
            print(traceback.print_exc())                 
            if return_error:
                return np.nan, None, np.nan
            else:
                return None, np.nan
            
        if return_expression == 'exact':
            metamodel_function = metamodel.exact_expression
            #print(metamodel_function)
        elif return_expression == 'approx':
            metamodel_function = metamodel.approx_expression
            #print(metamodel_function)

        end = time.time()
        runtime = end-start
        
        if return_error:
            random_lambda_input_data_preds_metamodel = metamodel.evaluate(random_lambda_input_data)
            random_lambda_input_data_preds_lambda_net = model.predict(random_lambda_input_data)

            error = mean_absolute_error(random_lambda_input_data_preds_lambda_net, random_lambda_input_data_preds_metamodel)        

            return error, metamodel_function, runtime
    
    return metamodel_function, runtime




def per_network_poly_optimization_slow(per_network_dataset_size, 
                                  lambda_network_weights, 
                                  #poly_representation, 
                                  list_of_monomial_identifiers_numbers, 
                                  config, 
                                  lr=0.05, 
                                  max_steps = 1000, 
                                  early_stopping=10, 
                                  restarts=5, 
                                  printing=True):
    
    def function_to_optimize():       
            
        poly_optimize = poly_optimize_input[0]

        
        if interpretation_net_output_monomials != None:
            poly_optimize_coeffs = poly_optimize[:interpretation_net_output_monomials]
            poly_optimize_identifiers_list = []
            if sparse_poly_representation_version == 1:
                for i in range(interpretation_net_output_monomials):
                    poly_optimize_identifiers = tf.math.softmax(poly_optimize[sparsity*i+interpretation_net_output_monomials:sparsity*(i+1)+interpretation_net_output_monomials])
                    poly_optimize_identifiers_list.append(poly_optimize_identifiers)
                poly_optimize_identifiers_list = tf.keras.backend.flatten(poly_optimize_identifiers_list)
            else:
                for i in range(interpretation_net_output_monomials):
                    for j in range(n):
                        poly_optimize_identifiers = tf.math.softmax(poly_optimize[i*n*j*(d+1)+interpretation_net_output_monomials:(i+1)*n*j*(d+1)+interpretation_net_output_monomials])
                        poly_optimize_identifiers_list.append(poly_optimize_identifiers)
                poly_optimize_identifiers_list = tf.keras.backend.flatten(poly_optimize_identifiers_list)                
            poly_optimize = tf.concat([poly_optimize_coeffs, poly_optimize_identifiers_list], axis=0)
                    
        poly_optimize = tf.convert_to_tensor(poly_optimize, dtype=tf.float32)
        
        poly_optimize_fv_list = []
        for lambda_input_entry in random_lambda_input_data:
            result = 0   
            
            value_without_coefficient_list = []
            for coefficient_multipliers in list_of_monomial_identifiers:
                value_without_coefficient = [lambda_input_value**coefficient_multiplier for coefficient_multiplier, lambda_input_value in zip(coefficient_multipliers, lambda_input_entry)]
                value_without_coefficient_list.append(reduce(lambda x, y: x*y, value_without_coefficient))
            
            
            if interpretation_net_output_monomials == None:
                result = tf.reduce_sum(tf.vectorized_map(lambda x: x[0]*x[1], (value_without_coefficient_list, poly_optimize)))
            else:
                coefficients = poly_optimize[:interpretation_net_output_monomials]
                index_array = poly_optimize[interpretation_net_output_monomials:]
                
                index_list = tf.split(index_array, interpretation_net_output_monomials)
                
                indices = tf.argmax(index_list, axis=1) 
                
                result = tf.reduce_sum(tf.vectorized_map(lambda x: tf.gather(value_without_coefficient_list, x[0])*x[1], (indices, coefficients)))             
                   
            poly_optimize_fv_list.append(result)
            

        #CALCULATE LAMBDA FV HERE FOR EVALUATION DATASET
        # build models
        start = 0
        layers = []
        for i in range(len(dims)//2):

            # set weights of layer
            index = i*2
            size = np.product(dims[index])
            weights_tf_true = tf.reshape(lambda_network_weights[start:start+size], dims[index])
            model_lambda_placeholder.layers[i].weights[0].assign(weights_tf_true)
            start += size

            # set biases of layer
            index += 1
            size = np.product(dims[index])
            biases_tf_true = tf.reshape(lambda_network_weights[start:start+size], dims[index])
            model_lambda_placeholder.layers[i].weights[1].assign(biases_tf_true)
            start += size


        lambda_fv = tf.keras.backend.flatten(model_lambda_placeholder(random_lambda_input_data))


        error = None
        if inet_loss == 'mae':
            error = tf.keras.losses.MAE(lambda_fv, poly_optimize_fv_list)
        elif inet_loss == 'r2':
            error = r2_keras_loss(lambda_fv, poly_optimize_fv_list)  
        else:
            raise SystemExit('Unknown I-Net Metric: ' + inet_loss)                

        error = tf.where(tf.math.is_nan(error), tf.fill(tf.shape(error), np.inf), error)        
                    

        return error #tf.reduce_mean(poly_optimize)#result        
  
    globals().update(config)
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)    
    if int(tf.__version__[0]) >= 2:
        tf.random.set_seed(RANDOM_SEED)
    else:
        tf.set_random_seed(RANDOM_SEED)       
    

    base_model = Sequential()

    base_model.add(Dense(lambda_network_layers[0], activation='relu', input_dim=n))

    for neurons in lambda_network_layers[1:]:
        base_model.add(Dense(neurons, activation='relu'))

    base_model.add(Dense(1))
    
    weights_structure = base_model.get_weights()
    
    #base_model = generate_base_model()
    
    random_lambda_input_data = np.random.uniform(low=x_min, high=x_max, size=(per_network_dataset_size, max(1, n)))

    
    model_lambda_placeholder = tf.keras.models.clone_model(base_model)  
    
    dims = [np_arrays.shape for np_arrays in weights_structure]
    
    
    lambda_network_weights = tf.dtypes.cast(tf.convert_to_tensor(lambda_network_weights), tf.float32)
    
    
    
    
    
        
    
    best_result = np.inf

    for current_iteration in range(restarts):
        
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        
        poly_optimize_input = tf.Variable(tf.random.uniform([1, interpretation_net_output_shape]))
        
        stop_counter = 0
        best_result_iteration = np.inf

        for current_step in range(max_steps):
            if stop_counter>=early_stopping:
                break
            
            opt.minimize(function_to_optimize, var_list=[poly_optimize_input])
            current_result = function_to_optimize()
            if printing:
                clear_output(wait=True)
                print("Current best: {} \n Curr_res: {} \n Iteration {}, Step {}".format(best_result_iteration,current_result, current_iteration, current_step), end='\r')
 
            stop_counter += 1
            if current_result < best_result_iteration:
                best_result_iteration = current_result
                stop_counter = 0
                best_poly_optimize_iteration = tf.identity(poly_optimize_input)
                
        if best_result_iteration < best_result:
            best_result = best_result_iteration
            best_poly_optimize = tf.identity(best_poly_optimize_iteration)

    per_network_poly = best_poly_optimize[0].numpy()
    
    if printing:
        print("Optimization terminated at {}".format(best_result))
        
        
    return per_network_poly





