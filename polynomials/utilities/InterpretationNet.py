#######################################################################################################################################################
#######################################################################Imports#########################################################################
#######################################################################################################################################################

import itertools 
#from tqdm import tqdm_notebook as tqdm
#import pickle
import cloudpickle
import dill 
from numba import cuda

import traceback

import numpy as np
import pandas as pd
import scipy as sp

from functools import reduce
from more_itertools import random_product 

#import math
from joblib import Parallel, delayed


from collections.abc import Iterable
#from scipy.integrate import quad

from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score, mean_absolute_error, r2_score
from similaritymeasures import frechet_dist, area_between_two_curves, dtw

from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras import Input, Model
import tensorflow as tf

import autokeras as ak
from autokeras import adapters, analysers
from tensorflow.python.util import nest

import random 

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from matplotlib import pyplot as plt
import seaborn as sns

from sympy import Symbol, sympify, lambdify, abc, SympifyError

#udf import
from utilities.LambdaNet import *
from utilities.metrics import *
from utilities.utility_functions import *

from tqdm import tqdm_notebook as tqdm

import time

#######################################################################################################################################################
#############################################################Setting relevant parameters from current config###########################################
#######################################################################################################################################################

def initialize_InterpretationNet_config_from_curent_notebook(config):
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
######################################################################AUTOKERAS BLOCKS#################################################################
#######################################################################################################################################################

class CombinedOutputInet(ak.Head):

    def __init__(self, loss = None, metrics = None, output_dim=None, **kwargs):
        super().__init__(loss=loss, metrics=metrics, **kwargs)
        self.output_dim = output_dim

    def get_config(self):
        config = super().get_config()
        return config

    def build(self, hp, inputs=None):    
        #inputs = nest.flatten(inputs)
        #if len(inputs) == 1:
        #    return inputs
        output_node = concatenate(inputs)           
        return output_node

    def config_from_analyser(self, analyser):
        super().config_from_analyser(analyser)
        self._add_one_dimension = len(analyser.shape) == 1

    def get_adapter(self):
        return adapters.RegressionAdapter(name=self.name)

    def get_analyser(self):
        return analysers.RegressionAnalyser(
            name=self.name, output_dim=self.output_dim
        )

    def get_hyper_preprocessors(self):
        hyper_preprocessors = []
        if self._add_one_dimension:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.AddOneDimension())
            )
        return hyper_preprocessors            

class ClassificationDenseInet(ak.Block):

    def build(self, hp, inputs=None):
        # Get the input_node from inputs.
        input_node = tf.nest.flatten(inputs)[0]
        layer = Dense(sparsity, activation='softmax')
        output_node = layer(input_node)
        return output_node    
    
class ClassificationDenseInetDegree(ak.Block):

    def build(self, hp, inputs=None):
        # Get the input_node from inputs.
        input_node = tf.nest.flatten(inputs)[0]
        layer = Dense(d+1, activation='softmax')
        output_node = layer(input_node)
        return output_node    
    

class RegressionDenseInet(ak.Block):

    def build(self, hp, inputs=None):
        # Get the input_node from inputs.
        input_node = tf.nest.flatten(inputs)[0]
        layer = Dense(interpretation_net_output_monomials)       
        output_node = layer(input_node)
        return output_node     





#######################################################################################################################################################
#################################################################I-NET RESULT CALCULATION##############################################################
#######################################################################################################################################################
    
def interpretation_net_training(lambda_net_train_dataset_list, 
                                         lambda_net_valid_dataset_list, 
                                         lambda_net_test_dataset_list):

    epochs_save_range_lambda = range(epoch_start//each_epochs_save_lambda, epochs_lambda//each_epochs_save_lambda) if each_epochs_save_lambda == 1 else range(epoch_start//each_epochs_save_lambda, epochs_lambda//each_epochs_save_lambda+1) if multi_epoch_analysis else range(1,2)
    
    n_jobs_inet_training = n_jobs
    if n_jobs==1 or (samples_list != None and len(samples_list) == 1) or (len(lambda_net_train_dataset_list) == 1 and samples_list == None) or use_gpu:
        n_jobs_inet_training = 1
    verbose = 0 if n_jobs_inet_training == 1 else 11
        
    save_string_list = []      
    for i in range(len(lambda_net_train_dataset_list)):
        save_string_list.append('')

            
    if samples_list == None:      
        
        print('----------------------------------------------- TRAINING INTERPRETATION NET -----------------------------------------------')
        
        start = time.time() 
        
        
        parallel_inet = Parallel(n_jobs=n_jobs_inet_training, verbose=verbose, backend='multiprocessing')     
        results_list = parallel_inet(delayed(train_inet)(lambda_net_train_dataset,
                                                           lambda_net_valid_dataset,
                                                           lambda_net_test_dataset,
                                                           current_jobs=n_jobs_inet_training,
                                                           callback_names=['early_stopping'],
                                                           save_string='epochs_' + str(save_epochs)) for lambda_net_train_dataset,
                                                                                                         lambda_net_valid_dataset,
                                                                                                         lambda_net_test_dataset,
                                                                                                         save_epochs in zip(lambda_net_train_dataset_list,
                                                                                                                                          lambda_net_valid_dataset_list,
                                                                                                                                          lambda_net_test_dataset_list,
                                                                                                                                          list(epochs_save_range_lambda)))          
        del parallel_inet
                
        end = time.time()     
        inet_train_time = (end - start) 
        minutes, seconds = divmod(int(inet_train_time), 60)
        hours, minutes = divmod(minutes, 60)        
        print('Training Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')
        
        if train_model:
            history_list = [result[0] for result in results_list]
        else:
            paths_dict = generate_paths(path_type = 'interpretation_net')

            path = './data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/history_epochs' + '.pkl'
            with open(path, 'rb') as f:
                history_list = pickle.load(f)  
                
        valid_data_list = [result[1] for result in results_list]
        X_valid_list = [valid_data[0] for valid_data in valid_data_list]
        y_valid_list = [valid_data[1] for valid_data in valid_data_list]
        
        test_data_list = [result[2] for result in results_list]
        X_test_list = [test_data[0] for test_data in test_data_list]
        y_test_list = [test_data[1] for test_data in test_data_list]   
        
        
        loss_function_list = [result[3] for result in results_list]
        metrics_list = [result[4] for result in results_list]
        
        #cuda.select_device(int(gpu_numbers))
        #cuda.close()    
        tf.keras.backend.clear_session()
    
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------ LOADING MODELS -----------------------------------------------------')
        
        start = time.time() 
        
        identifier_type = 'epochs'
        model_list = load_inets(identifier_type=identifier_type, path_identifier_list=list(epochs_save_range_lambda), loss_function_list=loss_function_list, metrics_list=metrics_list)
        
        end = time.time()     
        inet_train_time = (end - start) 
        minutes, seconds = divmod(int(inet_train_time), 60)
        hours, minutes = divmod(minutes, 60)        
        print('Loading Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')     
        
    else:
        parallel_inet = Parallel(n_jobs=n_jobs_inet_training, verbose=verbose, backend='multiprocessing') 
        results_list = parallel_inet(delayed(train_inet)(lambda_net_train_dataset.sample(samples),
                                                          lambda_net_valid_dataset,
                                                          lambda_net_test_dataset, 
                                                          current_jobs=n_jobs_inet_training,
                                                          callback_names=['early_stopping'],
                                                          save_string='samples_' + str(samples)) for samples in samples_list)     

        del parallel_inet
                
        end = time.time()     
        inet_train_time = (end - start) 
        minutes, seconds = divmod(int(inet_train_time), 60)
        hours, minutes = divmod(minutes, 60)        
        print('Training Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')
        
        if train_model:
            history_list = [result[0] for result in results_list]
        else:
            paths_dict = generate_paths(path_type = 'interpretation_net')

            path = './data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/history_samples' + '.pkl'
            with open(path, 'rb') as f:
                history_list = pickle.load(f)      
        
        
        valid_data_list = [result[1] for result in results_list]
        X_valid_list = [valid_data[0] for valid_data in valid_data_list]
        y_valid_list = [valid_data[1] for valid_data in valid_data_list]
        
        test_data_list = [result[2] for result in results_list]
        X_test_list = [test_data[0] for test_data in test_data_list]
        y_test_list = [test_data[1] for test_data in test_data_list]   
        
        loss_function_list = [result[3] for result in results_list]
        metrics_list = [result[4] for result in results_list]
        
        #cuda.select_device(int(gpu_numbers))
        #cuda.close()  
        tf.keras.backend.clear_session()
        
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------ LOADING MODELS -----------------------------------------------------')
        
        start = time.time() 
        
        identifier_type = 'samples'
        model_list = load_inets(identifier_type=identifier_type, path_identifier_list=samples_list, loss_function_list=loss_function_list, metrics_list=metrics_list)
        
        end = time.time()     
        inet_train_time = (end - start) 
        minutes, seconds = divmod(int(inet_train_time), 60)
        hours, minutes = divmod(minutes, 60)        
        print('Loading Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')     

    if not nas:
        generate_history_plots(history_list, by=identifier_type)
        #save_results(history_list, scores_test_list, by=identifier_type)    
        save_results(history_list=history_list, by=identifier_type)    
    

            
    return ((X_valid_list, y_valid_list), 
            (X_test_list, y_test_list),
            
            history_list, 
            
            #scores_valid_list
            #scores_test_list, 
            
            #function_values_valid_list, 
            #function_values_test_list, 
            
            #polynomial_dict_valid_list,
            #polynomial_dict_test_list,
            
            #distrib_dict_valid_list,
            #distrib_dict_test_list,
            
            #[identifier_type, samples_list, loss_function_list, metrics_list],
            
            model_list)
    
    
#######################################################################################################################################################
######################################################################I-NET TRAINING###################################################################
#######################################################################################################################################################

def load_inets(identifier_type, path_identifier_list, loss_function_list, metrics_list):
    
    
    paths_dict = generate_paths(path_type = 'interpretation_net')

    generic_path_identifier = str(data_reshape_version) + '_' + paths_dict['path_identifier_interpretation_net_data']
    if nas:
        generic_path_identifier = nas_type + '_' + generic_path_identifier
    
    save_string_list = []
    for path_identifier in path_identifier_list:
        save_string_list.append(str(identifier_type) + '_' + str(path_identifier))
       
    directory = './data/saved_models/'
    
    

    model_list = []
    from tensorflow.keras.utils import CustomObjectScope
    for save_string, loss_function, metrics in zip(save_string_list, loss_function_list, metrics_list):
        loss_function = dill.loads(loss_function)
        metrics = dill.loads(metrics)         
        
        #with CustomObjectScope({'custom_loss': loss_function}):
        custom_object_dict = {}
        custom_object_dict[loss_function.__name__] = loss_function
        for metric in  metrics:
            custom_object_dict[metric.__name__] = metric        
        model = tf.keras.models.load_model(directory + generic_path_identifier + save_string, custom_objects=custom_object_dict) # #, compile=False
        model_list.append(model)
        
    return model_list


def normalize_lambda_net(flat_weights, random_evaluation_dataset, base_model=None, config=None): 
        
    if base_model is None:
        base_model = generate_base_model()
    else:
        base_model = dill.loads(base_model)
        
    from utilities.LambdaNet import weights_to_model
                
    model = weights_to_model(flat_weights, config=config, base_model=base_model)
            
    model_preds_random_data = model.predict(random_evaluation_dataset)
    
    min_preds = model_preds_random_data.min()
    max_preds = model_preds_random_data.max()

    
    model_preds_random_data_normalized = (model_preds_random_data-min_preds)/(max_preds-min_preds)

    shaped_weights = model.get_weights()

    normalization_factor = (max_preds-min_preds)#0.01
    #print(normalization_factor)

    normalization_factor_per_layer = normalization_factor ** (1/(len(shaped_weights)/2))
    #print(normalization_factor_per_layer)

    numer_of_layers = int(len(shaped_weights)/2)
    #print(numer_of_layers)

    shaped_weights_normalized = []
    current_bias_normalization_factor = normalization_factor_per_layer
    current_bias_normalization_factor_reverse = normalization_factor_per_layer ** (len(shaped_weights)/2)
    
    for index, (weights, biases) in enumerate(pairwise(shaped_weights)):
        #print('current_bias_normalization_factor', current_bias_normalization_factor)
        #print('current_bias_normalization_factor_reverse', current_bias_normalization_factor_reverse)
        #print('normalization_factor_per_layer', normalization_factor_per_layer)          
        if index == numer_of_layers-1:
            weights = weights/normalization_factor_per_layer#weights * normalization_factor_per_layer
            biases = biases/current_bias_normalization_factor - min_preds/normalization_factor #biases * current_bias_normalization_factor            
        else:
            weights = weights/normalization_factor_per_layer#weights * normalization_factor_per_layer
            biases = biases/current_bias_normalization_factor#biases * current_bias_normalization_factor            

        #weights = (weights-min_preds/current_bias_normalization_factor_reverse)/normalization_factor_per_layer#weights * normalization_factor_per_layer
        #biases = (biases-min_preds/current_bias_normalization_factor_reverse)/normalization_factor_per_layer#biases * current_bias_normalization_factor
        shaped_weights_normalized.append(weights)
        shaped_weights_normalized.append(biases)

        current_bias_normalization_factor = current_bias_normalization_factor * normalization_factor_per_layer
        current_bias_normalization_factor_reverse = current_bias_normalization_factor_reverse / normalization_factor_per_layer  
    flat_weights_normalized = list(flatten(shaped_weights_normalized))  
    
    return flat_weights_normalized, (min_preds, max_preds)
    
def make_inet_prediction(model, networks_to_interpret, network_data=None, lambda_trained_normalized=False, inet_training_normalized=False, normalization_parameter_dict=None):
    
    global list_of_monomial_identifiers
        
        
    start = time.time()
        
    if inet_training_normalized:
        if network_data is None:
            network_data = np.random.uniform(low=x_min, high=x_max, size=(random_evaluation_dataset_size, n)) 

        inet_predictions_denormalized = [] 

        networks_to_interpret = return_numpy_representation(networks_to_interpret)

        if len(networks_to_interpret.shape) == 1:
            networks_to_interpret = np.array([networks_to_interpret]) 

        for network_index, network_to_interpret in enumerate(networks_to_interpret):

            print(network_to_interpret.shape)
            
            
            if not lambda_trained_normalized:    
                network_to_interpret_normalized, (network_to_interpret_min, network_to_interpret_max) = normalize_lambda_net(network_to_interpret, network_data)
            else:
                network_to_interpret_normalized = network_to_interpret
                network_to_interpret_min = normalization_parameter_dict['min'][network_index]
                network_to_interpret_max = normalization_parameter_dict['max'][network_index]

            inet_prediction_normalized = model.predict(np.array([network_to_interpret_normalized]))[0][:interpretation_net_output_shape]

            #print(inet_prediction_normalized)

            normalization_factor = network_to_interpret_max-network_to_interpret_min
            #print('normalization_factor', normalization_factor)
            #print('min', network_to_interpret_min)
            #print('max', network_to_interpret_max)


            if interpretation_net_output_monomials == None:
                inet_prediction_denormalized = []
                for monomial_identifier, monomial in zip(list_of_monomial_identifiers, inet_prediction_normalized):

                    #print(monomial)

                    if np.sum(np.abs(monomial_identifier)) == 0:
                        #print(monomial_identifier, monomial)
                        monomial = monomial * normalization_factor + network_to_interpret_min
                        #print(monomial_identifier, monomial)
                    else:
                        #print(monomial_identifier, monomial)
                        monomial = monomial * normalization_factor
                        #print(monomial_identifier, monomial)
                    #print(monomial)
                    inet_prediction_denormalized.append(monomial)
                #print(inet_prediction_denormalized)


            else:
                # ACHTUNG: WENN interpretation_net_output_monomials != None MUSS SICHERGESTELLT WERDEN, DASS COEFFICIENT IN POLYNOM-REPRÄSENTATION ENTHALTEN IST ODER HINZUGEFÜGT WERDEN

                constant_monomial = None
                for index, monomial_identifier in enumerate(list_of_monomial_identifiers):
                    if np.sum(np.abs(monomial_identifier)) == 0:
                        constant_monomial = index


                inet_prediction_normalized_coefficients = inet_prediction_normalized[:interpretation_net_output_monomials]
                inet_prediction_normalized_index_array = inet_prediction_normalized[interpretation_net_output_monomials:]

                inet_prediction_normalized_index_list = np.split(inet_prediction_normalized_index_array, interpretation_net_output_monomials)

                inet_prediction_normalized_indices = np.argmax(inet_prediction_normalized_index_list, axis=1) 

                inet_prediction_denormalized = None

                if False:
                    if constant_monomial in inet_prediction_normalized_indices:
                        inet_prediction_denormalized_coefficients = []
                        for monomial_index, monomial_coefficient in zip(inet_prediction_normalized_indices, inet_prediction_normalized_coefficients):
                            if monomial_index != constant_monomial:
                                denormalized_coefficient = monomial_coefficient * normalization_factor
                                inet_prediction_denormalized_coefficients.append(denormalized_coefficient)
                            else:
                                denormalized_coefficient = monomial_coefficient * normalization_factor + network_to_interpret_min
                                inet_prediction_denormalized_coefficients.append(denormalized_coefficient)          

                        inet_prediction_denormalized = np.hstack([inet_prediction_denormalized_coefficients, inet_prediction_normalized_index_array])         
                    else:
                        inet_prediction_denormalized_coefficients = np.multiply(inet_prediction_normalized_coefficients, normalization_factor)
                        inet_prediction_denormalized_coefficients = np.hstack([inet_prediction_denormalized_coefficients, network_to_interpret_min])

                        constant_monomial_identifier = [0 for i in range(len(list_of_monomial_identifiers))]
                        constant_monomial_identifier[constant_monomial] = 1

                        inet_prediction_normalized_index_array = np.hstack([inet_prediction_normalized_index_array, constant_monomial_identifier])

                        inet_prediction_denormalized = np.hstack([inet_prediction_denormalized_coefficients, inet_prediction_normalized_index_array])

                inet_prediction_denormalized_coefficients = np.multiply(inet_prediction_normalized_coefficients, normalization_factor)
                inet_prediction_denormalized_coefficients = np.hstack([inet_prediction_denormalized_coefficients, network_to_interpret_min])

                constant_monomial_identifier = [0 for i in range(len(list_of_monomial_identifiers))]
                constant_monomial_identifier[constant_monomial] = 1

                inet_prediction_normalized_index_array = np.hstack([inet_prediction_normalized_index_array, constant_monomial_identifier])

                inet_prediction_denormalized = np.hstack([inet_prediction_denormalized_coefficients, inet_prediction_normalized_index_array])


            inet_predictions_denormalized.append(inet_prediction_denormalized)

        if len(inet_predictions_denormalized) == 1:
            end = time.time()
            runtime = end-start
            return np.array(inet_predictions_denormalized[0]), runtime

        end = time.time()
        runtime = end-start        
        return np.array(inet_predictions_denormalized), runtime

    elif not lambda_trained_normalized:
        if len(networks_to_interpret.shape) == 1:
            network_to_interpret = np.array([networks_to_interpret])         
            inet_prediction = model.predict(np.array([network_to_interpret]))[0][:interpretation_net_output_shape]
            
            end = time.time()
            runtime = end-start            
            return inet_prediction, runtime
        else:
            inet_predictions = model.predict(networks_to_interpret)[:,:interpretation_net_output_shape]
            
            end = time.time()
            runtime = end-start            
            return inet_predictions, runtime
    else: #(if not inet_training_normalized and lambda_trained_normalized)
        return None, np.nan
        
def generate_inet_train_data(lambda_net_dataset, data_reshape_version=1):
    X_data = None
    X_data_flat = None
    y_data = None
    normalization_parameter_dict = None
    
    X_data = np.array(lambda_net_dataset.weight_list)
    
    if normalize_inet_data: 
        config={'optimizer_lambda': optimizer_lambda,
               'loss_lambda': loss_lambda}
        
        
        normalization_parameter_dict = {
            'min': [],
            'max': []
        }

        parallel_normalize_lambda = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')
        results_normalize_lambda = parallel_normalize_lambda(delayed(normalize_lambda_net)(flat_weights, random_evaluation_dataset, dill.dumps(base_model), config=config) for flat_weights in X_data)         
        del parallel_normalize_lambda

        X_data = np.array([result[0] for result in results_normalize_lambda])
        normalization_parameter_dict['min'] = [result[1][0] for result in results_normalize_lambda]
        normalization_parameter_dict['max'] = [result[1][1] for result in results_normalize_lambda]


        
    if evaluate_with_real_function: #target polynomial as inet target
        y_data = np.array(lambda_net_dataset.target_polynomial_list)

        
        if convolution_layers != None or lstm_layers != None or (nas and nas_type != 'SEQUENTIAL'):
            if data_reshape_version == None:
                data_reshape_version = 2
            X_data, X_data_flat = restructure_data_cnn_lstm(X_data, version=data_reshape_version, subsequences=None)

            
    else: #lstsq lambda pred polynomial as inet target
        y_data = np.array(lambda_net_dataset.lstsq_lambda_pred_polynomial_list)
        
        if convolution_layers != None or lstm_layers != None or (nas and nas_type != 'SEQUENTIAL'):
            if data_reshape_version == None:
                data_reshape_version = 2
            X_data, X_data_flat = restructure_data_cnn_lstm(X_data, version=data_reshape_version, subsequences=None)

    
    return X_data, X_data_flat, y_data, normalization_parameter_dict
        
def train_inet(lambda_net_train_dataset,
              lambda_net_valid_dataset,
              lambda_net_test_dataset, 
              current_jobs,
              callback_names = [],
              save_string = None):
    
   
    global optimizer
    global loss
    global data_reshape_version
    
    paths_dict = generate_paths(path_type = 'interpretation_net')
    
    ############################## DATA PREPARATION ###############################
    
    base_model = generate_base_model()
    random_evaluation_dataset = np.random.uniform(low=x_min, high=x_max, size=(random_evaluation_dataset_size, n))
            
    weights_structure = base_model.get_weights()
    dims = [np_arrays.shape for np_arrays in weights_structure]         

    (X_train, X_train_flat, y_train, normalization_parameter_train_dict) = generate_inet_train_data(lambda_net_train_dataset, data_reshape_version=data_reshape_version)
    (X_valid, X_valid_flat, y_valid, normalization_parameter_valid_dict) = generate_inet_train_data(lambda_net_valid_dataset, data_reshape_version=data_reshape_version)
    (X_test, X_test_flat, y_test, normalization_parameter_test_dict) = generate_inet_train_data(lambda_net_test_dataset, data_reshape_version=data_reshape_version)
    
    ############################## OBJECTIVE SPECIFICATION AND LOSS FUNCTION ADJUSTMENTS ###############################
    current_monomial_degree = tf.Variable(0, dtype=tf.int64)
    metrics = []
    if consider_labels_training:
        if (not evaluate_with_real_function and sample_sparsity is not None) or sparse_poly_representation_version==1:
            raise SystemExit('No coefficient-based optimization possible with reduced output monomials - Please change settings')         
        loss_function = inet_coefficient_loss_wrapper(inet_loss, list_of_monomial_identifiers)
        
        for inet_metric in list(flatten([inet_metrics, inet_loss])):
            #metrics.append(inet_poly_fv_loss_wrapper(inet_metric, random_evaluation_dataset, list_of_monomial_identifiers, current_monomial_degree, base_model))     
            metrics.append(inet_lambda_fv_loss_wrapper(inet_metric, random_evaluation_dataset, list_of_monomial_identifiers, current_monomial_degree, base_model, weights_structure, dims)) 
    else:
        if evaluate_with_real_function:
            loss_function = inet_poly_fv_loss_wrapper(inet_loss, random_evaluation_dataset, list_of_monomial_identifiers, current_monomial_degree, base_model)
            #for inet_metric in inet_metrics:
                #metrics.append(inet_poly_fv_loss_wrapper(inet_metric, random_evaluation_dataset, list_of_monomial_identifiers, current_monomial_degree, base_model)) 
            for inet_metric in list(flatten([inet_metrics, inet_loss])):
                metrics.append(inet_coefficient_loss_wrapper(inet_metric, list_of_monomial_identifiers))            
                metrics.append(inet_lambda_fv_loss_wrapper(inet_metric, random_evaluation_dataset, list_of_monomial_identifiers, current_monomial_degree, base_model, weights_structure, dims)) 
        else:
            loss_function = inet_lambda_fv_loss_wrapper(inet_loss, random_evaluation_dataset, list_of_monomial_identifiers, current_monomial_degree, base_model, weights_structure, dims)
            for inet_metric in inet_metrics:
                metrics.append(inet_lambda_fv_loss_wrapper(inet_metric, random_evaluation_dataset, list_of_monomial_identifiers, current_monomial_degree, base_model, weights_structure, dims)) 
            #for inet_metric in list(flatten([inet_metrics, inet_loss])):
                #COEFFICIENT LOSS NOT POSSIBLE IF sample_sparsity is not None --> LSTSQ POLY FÜR VEGLEICH HAT NICHT DIE GLEICHE STRUKTUR
                #metrics.append(inet_coefficient_loss_wrapper(inet_metric, list_of_monomial_identifiers))            
                #metrics.append(inet_poly_fv_loss_wrapper(inet_metric, random_evaluation_dataset, list_of_monomial_identifiers, current_monomial_degree, base_model)) 
                
    if convolution_layers != None or lstm_layers != None or (nas and nas_type != 'SEQUENTIAL'):
        y_train_model = np.hstack((y_train, X_train_flat))   
        valid_data = (X_valid, np.hstack((y_valid, X_valid_flat)))   
    else:
        y_train_model = np.hstack((y_train, X_train))   
        valid_data = (X_valid, np.hstack((y_valid, X_valid)))                   
              

        
    ############################## BUILD MODEL ###############################
    if train_model:
        if nas:
            from tensorflow.keras.utils import CustomObjectScope

            custom_object_dict = {}
            loss_function_name = loss_function.__name__
            custom_object_dict[loss_function_name] = loss_function
            metric_names = []
            for metric in  metrics:
                metric_name = metric.__name__
                metric_names.append(metric_name)
                custom_object_dict[metric_name] = metric  

            #print(custom_object_dict)    
            #print(metric_names)
            #print(loss_function_name)

            with CustomObjectScope(custom_object_dict):
                if nas_type == 'SEQUENTIAL':
                    input_node = ak.Input()

                    if nas_type =='SEQUENTIAL-NORM':
                        hidden_node = ak.Normalization()(input_node)
                        hidden_node = ak.DenseBlock()(hidden_node)
                    else:
                        hidden_node = ak.DenseBlock()(input_node)

                    #print('interpretation_net_output_monomials', interpretation_net_output_monomials)

                    if interpretation_net_output_monomials == None:
                        output_node = ak.RegressionHead()(hidden_node)  
                        #output_node = ak.RegressionHead(output_dim=sparsity)(hidden_node)  
                    else:
                        #outputs_coeff = ak.RegressionHead(output_dim=interpretation_net_output_monomials)(hidden_node)  
                        outputs_coeff = RegressionDenseInet()(hidden_node)  
                        outputs_list = [outputs_coeff]

                        if sparse_poly_representation_version == 1:
                            for outputs_index in range(interpretation_net_output_monomials):
                                outputs_identifer =  ClassificationDenseInet()(hidden_node)
                                outputs_list.append(outputs_identifer)                                    
                        elif sparse_poly_representation_version == 2:
                            for outputs_index in range(interpretation_net_output_monomials):
                                for var_index in range(n):
                                    outputs_identifer =  ClassificationDenseInetDegree()(hidden_node)
                                    outputs_list.append(outputs_identifer)
                        #print('outputs_list', outputs_list)

                        #output_node = CombinedOutputInet(output_dim=interpretation_net_output_shape)(outputs_list)
                        output_node = CombinedOutputInet()(outputs_list)

                elif nas_type == 'CNN': 
                    input_node = ak.Input()
                    hidden_node = ak.ConvBlock()(input_node)
                    hidden_node = ak.DenseBlock()(hidden_node)

                    if interpretation_net_output_monomials == None:
                        output_node = ak.RegressionHead()(hidden_node)  
                        #output_node = ak.RegressionHead(output_dim=sparsity)(hidden_node)  
                    else:
                        #outputs_coeff = ak.RegressionHead(output_dim=interpretation_net_output_monomials)(hidden_node)  
                        outputs_coeff = RegressionDenseInet()(hidden_node)  
                        outputs_list = [outputs_coeff]
                        if sparse_poly_representation_version == 1:
                            for outputs_index in range(interpretation_net_output_monomials):
                                outputs_identifer =  ClassificationDenseInet()(hidden_node)
                                outputs_list.append(outputs_identifer)                                    
                        elif sparse_poly_representation_version == 2:
                            for outputs_index in range(interpretation_net_output_monomials):
                                for var_index in range(n):
                                    outputs_identifer =  ClassificationDenseInetDegree()(hidden_node)
                                    outputs_list.append(outputs_identifer)

                        output_node = CombinedOutputInet()(outputs_list)


                elif nas_type == 'LSTM':
                    input_node = ak.Input()
                    hidden_node = ak.RNNBlock()(input_node)
                    hidden_node = ak.DenseBlock()(hidden_node)

                    if interpretation_net_output_monomials == None:
                        output_node = ak.RegressionHead()(hidden_node)  
                        #output_node = ak.RegressionHead(output_dim=sparsity)(hidden_node)  
                    else:
                        #outputs_coeff = ak.RegressionHead(output_dim=interpretation_net_output_monomials)(hidden_node)  
                        outputs_coeff = RegressionDenseInet()(hidden_node)  
                        outputs_list = [outputs_coeff]
                        if sparse_poly_representation_version == 1:
                            for outputs_index in range(interpretation_net_output_monomials):
                                outputs_identifer =  ClassificationDenseInet()(hidden_node)
                                outputs_list.append(outputs_identifer)                                    
                        elif sparse_poly_representation_version == 2:
                            for outputs_index in range(interpretation_net_output_monomials):
                                for var_index in range(n):
                                    outputs_identifer =  ClassificationDenseInetDegree()(hidden_node)
                                    outputs_list.append(outputs_identifer)

                        output_node = CombinedOutputInet()(outputs_list)            

                elif nas_type == 'CNN-LSTM': 
                    input_node = ak.Input()
                    hidden_node = ak.ConvBlock()(input_node)
                    hidden_node = ak.RNNBlock()(hidden_node)
                    hidden_node = ak.DenseBlock()(hidden_node)

                    if interpretation_net_output_monomials == None:
                        output_node = ak.RegressionHead()(hidden_node)  
                        #output_node = ak.RegressionHead(output_dim=sparsity)(hidden_node)  
                    else:
                        #outputs_coeff = ak.RegressionHead(output_dim=interpretation_net_output_monomials)(hidden_node)  
                        outputs_coeff = RegressionDenseInet()(hidden_node)  
                        outputs_list = [outputs_coeff]
                        if sparse_poly_representation_version == 1:
                            for outputs_index in range(interpretation_net_output_monomials):
                                outputs_identifer =  ClassificationDenseInet()(hidden_node)
                                outputs_list.append(outputs_identifer)                                    
                        elif sparse_poly_representation_version == 2:
                            for outputs_index in range(interpretation_net_output_monomials):
                                for var_index in range(n):
                                    outputs_identifer =  ClassificationDenseInetDegree()(hidden_node)
                                    outputs_list.append(outputs_identifer)

                        output_node = CombinedOutputInet()(outputs_list)           

                elif nas_type == 'CNN-LSTM-parallel':                         
                    input_node = ak.Input()
                    hidden_node1 = ak.ConvBlock()(input_node)
                    hidden_node2 = ak.RNNBlock()(input_node)
                    hidden_node = ak.Merge()([hidden_node1, hidden_node2])
                    hidden_node = ak.DenseBlock()(hidden_node)

                    if interpretation_net_output_monomials == None:
                        output_node = ak.RegressionHead()(hidden_node)  
                        #output_node = ak.RegressionHead(output_dim=sparsity)(hidden_node)  
                    else:
                        #outputs_coeff = ak.RegressionHead(output_dim=interpretation_net_output_monomials)(hidden_node)  
                        outputs_coeff = RegressionDenseInet()(hidden_node)  
                        outputs_list = [outputs_coeff]
                        if sparse_poly_representation_version == 1:
                            for outputs_index in range(interpretation_net_output_monomials):
                                outputs_identifer =  ClassificationDenseInet()(hidden_node)
                                outputs_list.append(outputs_identifer)                                    
                        elif sparse_poly_representation_version == 2:
                            for outputs_index in range(interpretation_net_output_monomials):
                                for var_index in range(n):
                                    outputs_identifer =  ClassificationDenseInetDegree()(hidden_node)
                                    outputs_list.append(outputs_identifer)

                        output_node = CombinedOutputInet()(outputs_list)            

                directory = './data/autokeras/' + nas_type + '_' + str(data_reshape_version) + '_' + paths_dict['path_identifier_interpretation_net_data'] + save_string

                auto_model = ak.AutoModel(inputs=input_node, 
                                    outputs=output_node,
                                    loss=loss_function_name,
                                    metrics=metric_names,
                                    objective='val_loss',
                                    overwrite=True,
                                    tuner='greedy',#'hyperband',#"bayesian",
                                    max_trials=nas_trials,
                                    directory=directory,
                                    seed=RANDOM_SEED+1)

                ############################## PREDICTION ###############################


                auto_model.fit(
                    x=X_train,
                    y=y_train_model,
                    validation_data=valid_data,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=return_callbacks_from_string('early_stopping'),
                    )


                history = auto_model.tuner.oracle.get_best_trials(min(nas_trials, 5))
                model = auto_model.export_model()

                model.save('./data/saved_models/' + nas_type + '_' + str(data_reshape_version) + '_' + paths_dict['path_identifier_interpretation_net_data'] + save_string)

        else: 
            inputs = Input(shape=X_train.shape[1], name='input')

            #hidden = Dense(dense_layers[0], activation='relu', name='hidden1_' + str(dense_layers[0]))(inputs)
            hidden = tf.keras.layers.Dense(dense_layers[0], name='hidden1_' + str(dense_layers[0]))(inputs)
            hidden = tf.keras.layers.Activation(activation='relu', name='activation1_' + 'relu')(hidden)

            if dropout > 0:
                #hidden = Dropout(dropout, name='dropout1_' + str(dropout))(hidden)
                hidden = tf.keras.layers.Dropout(dropout, name='dropout1_' + str(dropout))(hidden)

            for layer_index, neurons in enumerate(dense_layers[1:]):
                if dropout > 0 and layer_index > 0:
                    #hidden = Dropout(dropout, name='dropout' + str(layer_index+2) + '_' + str(dropout))(hidden)  
                    hidden = tf.keras.layers.Dropout(dropout, name='dropout' + str(layer_index+2) + '_' + str(dropout))(hidden)

                #hidden = Dense(neurons, activation='relu', name='hidden' + str(layer_index+2) + '_' + str(neurons))(hidden)
                hidden = tf.keras.layers.Dense(neurons, name='hidden' + str(layer_index+2) + '_' + str(neurons))(hidden)
                hidden = tf.keras.layers.Activation(activation='relu', name='activation'  + str(layer_index+2) + '_relu')(hidden)

            if dropout_output > 0:
                #hidden = Dropout(dropout_output, name='dropout_output_' + str(dropout_output))(hidden)            
                hidden = tf.keras.layers.Dropout(dropout_output, name='dropout_output_' + str(dropout_output))(hidden)

            if interpretation_net_output_monomials == None:
                #outputs = Dense(sparsity, name='output_' + str(neurons))(hidden)
                outputs = tf.keras.layers.Dense(sparsity, name='output_' + str(neurons))(hidden)
            else:
                #outputs_coeff = Dense(interpretation_net_output_monomials, name='output_coeff_' + str(interpretation_net_output_monomials))(hidden)
                outputs_coeff = tf.keras.layers.Dense(interpretation_net_output_monomials, name='output_coeff_' + str(interpretation_net_output_monomials))(hidden)

                outputs_list = [outputs_coeff]

                if sparse_poly_representation_version == 1:
                    for outputs_index in range(interpretation_net_output_monomials):
                        #outputs_identifer = Dense(sparsity, activation='softmax', name='output_identifier' + str(outputs_index+1) + '_' + str(sparsity))(hidden)
                        outputs_identifer = tf.keras.layers.Dense(sparsity, activation='softmax', name='output_identifier' + str(outputs_index+1) + '_' + str(sparsity))(hidden)
                        outputs_list.append(outputs_identifer)

                elif sparse_poly_representation_version == 2:
                    for outputs_index in range(interpretation_net_output_monomials):
                        for var_index in range(n):
                            #outputs_identifer = Dense(sparsity, activation='softmax', name='output_identifier' + str(outputs_index+1) + '_' + str(sparsity))(hidden)
                            outputs_identifer = tf.keras.layers.Dense(d+1, activation='softmax', name='output_identifier' + '_mon' +  str(outputs_index+1) + '_var' + str(var_index+1) + '_' + str(sparsity))(hidden)
                            outputs_list.append(outputs_identifer)       

                outputs = concatenate(outputs_list, name='output_combined')



            model = Model(inputs=inputs, outputs=outputs)

            callbacks = return_callbacks_from_string(callback_names)            

            if optimizer == "custom":
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

            model.compile(optimizer=optimizer,
                          loss=loss_function,
                          metrics=metrics
                         )

            verbosity = 1 #if n_jobs ==1 else 0

            ############################## PREDICTION ###############################
            history = model.fit(X_train,
                      y_train_model,
                      epochs=epochs, 
                      batch_size=batch_size, 
                      validation_data=valid_data,
                      callbacks=callbacks,
                      verbose=verbosity)

            history = history.history
            
            model.save('./data/saved_models/' + str(data_reshape_version) + '_' + paths_dict['path_identifier_interpretation_net_data'] + save_string)
    else:
        history = None
        
    return history, (X_valid, y_valid), (X_test, y_test), dill.dumps(loss_function), dill.dumps(metrics) 
     
def calculate_all_function_values(lambda_net_dataset, polynomial_dict):
          
    n_jobs_parallel_fv = n_jobs
    backend='threading'

    if n_jobs_parallel_fv <= 5:
        n_jobs_parallel_fv = 10

    #backend='threading' 
    #backend='sequential' 

    with tf.device('/CPU:0'):        
        function_value_dict = {
            'lambda_preds': np.nan_to_num(lambda_net_dataset.make_prediction_on_test_data()),
            'target_polynomials': np.nan_to_num(lambda_net_dataset.return_target_poly_fvs_on_test_data(n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)),          
            'lstsq_lambda_pred_polynomials': np.nan_to_num(lambda_net_dataset.return_lstsq_lambda_pred_polynomial_fvs_on_test_data(n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)),         
            'lstsq_target_polynomials': np.nan_to_num(lambda_net_dataset.return_lstsq_target_polynomial_fvs_on_test_data(n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)),   
            'inet_polynomials': np.nan_to_num(parallel_fv_calculation_from_polynomial(polynomial_dict['inet_polynomials'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)),      
        }


        try:
            print('metamodel_poly')
            variable_names = ['X' + str(i) for i in range(n)]
            function_values = parallel_fv_calculation_from_sympy(polynomial_dict['metamodel_poly'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend, variable_names=variable_names)
            function_value_dict['metamodel_poly'] =  function_values#np.nan_to_num(function_values)
        except KeyError as ke:
            print('Exit', KeyError)    

        try:
            print('metamodel_functions')
            variable_names = ['X' + str(i) for i in range(n)]
            function_values = parallel_fv_calculation_from_sympy(polynomial_dict['metamodel_functions'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend, variable_names=variable_names)
            function_value_dict['metamodel_functions'] = function_values#np.nan_to_num(function_values)
        except KeyError as ke:
            print('Exit', KeyError)    

        try:
            print('metamodel_functions_no_GD')
            #variable_names = ['X' + str(i) for i in range(n)] if n > 1 else ['x']
            function_values = parallel_fv_calculation_from_sympy(polynomial_dict['metametamodel_functions_no_GD'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)
            function_value_dict['metamodel_functions_no_GD'] = function_values#np.nan_to_num(function_values)
        except KeyError as ke:
            print('Exit', KeyError)    
            
        try:
            print('metamodel_functions_no_GD_poly')
            #variable_names = ['X' + str(i) for i in range(n)] if n > 1 else ['x']
            function_values = parallel_fv_calculation_from_sympy(polynomial_dict['metamodel_functions_no_GD_poly'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)
            function_value_dict['metamodel_functions_no_GD_poly'] = function_values#np.nan_to_num(function_values)
        except KeyError as ke:
            print('Exit', KeyError)             

        try:
            print('symbolic_regression_functions')
            variable_names = ['X' + str(i) for i in range(n)]
            #variable_names[0] = 'x'        
            function_values = parallel_fv_calculation_from_sympy(polynomial_dict['symbolic_regression_functions'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend, variable_names=variable_names)
            function_value_dict['symbolic_regression_functions'] = function_values#np.nan_to_num(function_values)

            #print(function_values)

            #for function_value in function_values:
            #    if np.isnan(function_value).any() or np.isinf(function_value).any():
            #        print(function_value)

            #print(function_values[-2])

            #for function_value in function_value_dict['symbolic_regression_functions']:
            #    if np.isnan(function_value).any() or np.isinf(function_value).any():
            #        print(function_value)        
            #print(function_value_dict['symbolic_regression_functions'][-2])

        except KeyError as ke:
            print('Exit', KeyError)   
            
        try:
            print('polynomial_regression_functions')
            variable_names = ['x' + str(i) for i in range(n)]
            #variable_names[0] = 'x'        
            function_values = parallel_fv_calculation_from_sympy(polynomial_dict['polynomial_regression_functions'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend, variable_names=variable_names)
            function_value_dict['polynomial_regression_functions'] = function_values#np.nan_to_num(function_values)

        except KeyError as ke:
            print('Exit', KeyError)               
            
            
            
        try:
            print('per_network_polynomials')
            function_values = parallel_fv_calculation_from_polynomial(polynomial_dict['per_network_polynomials'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)        
            function_value_dict['per_network_polynomials'] = function_values#np.nan_to_num(function_values)
        except KeyError as ke:
            print('Exit', KeyError)    


    return function_value_dict
    
def evaluate_all_predictions(function_value_dict, polynomial_dict):
    
    ############################## EVALUATION ###############################
    evaluation_key_list = []
    evaluation_scores_list = []
    evaluation_distrib_list = []
    runtime_distrib_list = []
    
    key_list = []
    for combination in itertools.combinations(function_value_dict.keys(), r=2):
        key_1 = combination[0]
        key_2 = combination[1]
        
        try:
            polynomials_1 = polynomial_dict[key_1]
            if type(polynomials_1[0]) != np.ndarray and type(polynomials_1[0]) != list:
                polynomials_1 = None            
        except KeyError:
            polynomials_1 = None
            
        try:
            polynomials_2 = polynomial_dict[key_2]
            if type(polynomials_2[0]) != np.ndarray and type(polynomials_2[0]) != list:
                polynomials_2 = None
        except KeyError:
            polynomials_2 = None
                        
        function_values_1 = function_value_dict[key_1]
        function_values_2 = function_value_dict[key_2]
        
        
        evaluation_key = key_1 + '_VS_' + key_2
        print(evaluation_key)
        evaluation_key_list.append(evaluation_key)
                
        evaluation_scores, evaluation_distrib = evaluate_interpretation_net(polynomials_1, 
                                                                            polynomials_2, 
                                                                            function_values_1, 
                                                                            function_values_2)        
        evaluation_scores_list.append(evaluation_scores)
        evaluation_distrib_list.append(evaluation_distrib)
        
    scores_dict = pd.DataFrame(data=evaluation_scores_list,
                               index=evaluation_key_list)        
        
    
    mae_distrib_dict = pd.DataFrame(data=[evaluation_distrib['MAE'] for evaluation_distrib in evaluation_distrib_list],
                                    index=evaluation_key_list)
    
        
    r2_distrib_dict = pd.DataFrame(data=[evaluation_distrib['R2'] for evaluation_distrib in evaluation_distrib_list],
                                    index=evaluation_key_list)
 
    
    distrib_dicts = {'MAE': mae_distrib_dict, 
                     'R2': r2_distrib_dict}        
        
    
    return scores_dict, distrib_dicts

def per_network_poly_generation(lambda_net_dataset, optimization_type='scipy', backend='loky'): 
        
    printing = True if n_jobs == 1 else False
    #if use_gpu and False:
        #os.environ['CUDA_VISIBLE_DEVICES'] = gpu_numbers if use_gpu else ''
        #os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        #os.environ['XLA_FLAGS'] =  '--xla_gpu_cuda_data_dir=/usr/lib/cuda-10.1'     
        #backend = 'sequential'
        #printing = True
        
    #backend = 'sequential'
    #printing = True       
    #per_network_optimization_dataset_size = 5

    if optimization_type=='tf':
        
        per_network_hyperparams = {
            'optimizer': tf.keras.optimizers.RMSprop,
            'lr': 0.02,
            'max_steps': 500,
            'early_stopping': 10,
            'restarts': 3,
            'per_network_dataset_size': per_network_optimization_dataset_size,
        }


        lambda_network_weights_list = np.array(lambda_net_dataset.weight_list)


        config = {
                 'n': n,
                 'inet_loss': inet_loss,
                 'sparsity': sparsity,
                 'lambda_network_layers': lambda_network_layers,
                 'interpretation_net_output_shape': interpretation_net_output_shape,
                 'RANDOM_SEED': RANDOM_SEED,
                 'nas': nas,
                 'number_of_lambda_weights': number_of_lambda_weights,
                 'interpretation_net_output_monomials': interpretation_net_output_monomials,
                 #'list_of_monomial_identifiers': list_of_monomial_identifiers,
                 'x_min': x_min,
                 'x_max': x_max,
                 'sparse_poly_representation_version': sparse_poly_representation_version,
                }

        with tf.device('/CPU:0'):
            parallel_per_network = Parallel(n_jobs=n_jobs, verbose=1, backend=backend)

            per_network_optimization_polynomials = parallel_per_network(delayed(per_network_poly_optimization_tf)(per_network_hyperparams['per_network_dataset_size'], 
                                                                                                                  lambda_network_weights, 
                                                                                                                  list_of_monomial_identifiers, 
                                                                                                                  config,
                                                                                                                  optimizer = per_network_hyperparams['optimizer'],
                                                                                                                  lr = per_network_hyperparams['lr'], 
                                                                                                                  max_steps = per_network_hyperparams['max_steps'], 
                                                                                                                  early_stopping = per_network_hyperparams['early_stopping'], 
                                                                                                                  restarts = per_network_hyperparams['restarts'],
                                                                                                                  printing = printing,
                                                                                                                  return_error = True) for lambda_network_weights in lambda_network_weights_list)      

            del parallel_per_network

    elif optimization_type=='scipy':    
        per_network_hyperparams = {
            'optimizer':  'Powell',
            'jac': 'fprime',
            'max_steps': 500,
            'restarts': 3,
            'per_network_dataset_size': per_network_optimization_dataset_size,
        }

        
        lambda_network_weights_list = np.array(lambda_net_dataset.weight_list)


        config = {
                 'n': n,
                 'inet_loss': inet_loss,
                 'sparsity': sparsity,
                 'lambda_network_layers': lambda_network_layers,
                 'interpretation_net_output_shape': interpretation_net_output_shape,
                 'RANDOM_SEED': RANDOM_SEED,
                 'nas': nas,
                 'number_of_lambda_weights': number_of_lambda_weights,
                 'interpretation_net_output_monomials': interpretation_net_output_monomials,
                 'x_min': x_min,
                 'x_max': x_max,
                 'sparse_poly_representation_version': sparse_poly_representation_version,     
                'max_optimization_minutes': max_optimization_minutes,
                 }
        with tf.device('/CPU:0'):
            if False:
                result = per_network_poly_optimization_scipy(per_network_hyperparams['per_network_dataset_size'], 
                                                  lambda_network_weights_list[0], 
                                                  list_of_monomial_identifiers, 
                                                  config,
                                                  optimizer = per_network_hyperparams['optimizer'],
                                                  jac = per_network_hyperparams['jac'],
                                                  max_steps = per_network_hyperparams['max_steps'], 
                                                  restarts = per_network_hyperparams['restarts'],
                                                  printing = True,
                                                  return_error = True)
                print(result)        
            
            parallel_per_network = Parallel(n_jobs=n_jobs, verbose=1, backend=backend)

            result_list_per_network = parallel_per_network(delayed(per_network_poly_optimization_scipy)(per_network_hyperparams['per_network_dataset_size'], 
                                                                                                                      lambda_network_weights, 
                                                                                                                      list_of_monomial_identifiers, 
                                                                                                                      config,
                                                                                                                      optimizer = per_network_hyperparams['optimizer'],
                                                                                                                      jac = per_network_hyperparams['jac'],
                                                                                                                      max_steps = per_network_hyperparams['max_steps'], 
                                                                                                                      restarts = per_network_hyperparams['restarts'],
                                                                                                                      printing = printing,
                                                                                                                      return_error = True) for lambda_network_weights in lambda_network_weights_list)      
            per_network_optimization_errors = [result[0] for result in result_list_per_network]
            per_network_optimization_polynomials = [result[1] for result in result_list_per_network]          

            del parallel_per_network    
    
    #if use_gpu:
        #os.environ['CUDA_VISIBLE_DEVICES'] = gpu_numbers    
        
    return per_network_optimization_polynomials

def symbolic_regression_function_generation(lambda_net_dataset, backend='loky'):
        
    printing = True if n_jobs == 1 else False
    
    #backend='multiprocessing'
    
    symbolic_regression_hyperparams = {
        'dataset_size': per_network_optimization_dataset_size,
    }
    #backend='sequential'
    
    config = {
            'n': n,
            'd': d,
            'inet_loss': inet_loss,
            'sparsity': sparsity,
            'sample_sparsity': sample_sparsity,
            'lambda_network_layers': lambda_network_layers,
            'interpretation_net_output_shape': interpretation_net_output_shape,
            'RANDOM_SEED': RANDOM_SEED,
            'nas': nas,
            'number_of_lambda_weights': number_of_lambda_weights,
            'interpretation_net_output_monomials': interpretation_net_output_monomials,
            'fixed_initialization_lambda_training': fixed_initialization_lambda_training,
            'dropout': dropout,
            'lambda_network_layers': lambda_network_layers,
            'optimizer_lambda': optimizer_lambda,
            'loss_lambda': loss_lambda,        
             #'list_of_monomial_identifiers': list_of_monomial_identifiers,
             'x_min': x_min,
             'x_max': x_max,
             'sparse_poly_representation_version': sparse_poly_representation_version,
            'max_optimization_minutes': max_optimization_minutes,
             }

    parallel_symbolic_regression = Parallel(n_jobs=n_jobs, verbose=11, backend=backend)

    return_error = False
    
    result_list_symbolic_regression = parallel_symbolic_regression(delayed(symbolic_regression)(lambda_net, 
                                                                                  config,
                                                                                  symbolic_regression_hyperparams,
                                                                                  printing = printing,
                                                                                  return_error = return_error) for lambda_net in lambda_net_dataset.lambda_net_list)      

    del parallel_symbolic_regression  
    
    if return_error:
        symbolic_regression_errors = [result[0] for result in result_list_symbolic_regression]
        symbolic_regression_functions = [result[1] for result in result_list_symbolic_regression]   
        symbolic_regression_runtimes = [result[2] for result in result_list_symbolic_regression]   
    else:
        symbolic_regression_functions = [result[0] for result in result_list_symbolic_regression]   
        symbolic_regression_runtimes = [result[1] for result in result_list_symbolic_regression] 
        
        return symbolic_regression_functions, symbolic_regression_runtimes
    
    return symbolic_regression_errors, symbolic_regression_functions, symbolic_regression_runtimes


def polynomial_regression_function_generation(lambda_net_dataset, backend='loky'):
        
    printing = True if n_jobs == 1 else False
    
    #backend='multiprocessing'
    
    polynomial_regression_hyperparams = {
        'dataset_size': per_network_optimization_dataset_size,
    }
    #backend='sequential'
    
    config = {
            'n': n,
            'd': d,
            'inet_loss': inet_loss,
            'sparsity': sparsity,
            'sample_sparsity': sample_sparsity,
            'lambda_network_layers': lambda_network_layers,
            'interpretation_net_output_shape': interpretation_net_output_shape,
            'RANDOM_SEED': RANDOM_SEED,
            'nas': nas,
            'number_of_lambda_weights': number_of_lambda_weights,
            'interpretation_net_output_monomials': interpretation_net_output_monomials,
            'fixed_initialization_lambda_training': fixed_initialization_lambda_training,
            'dropout': dropout,
            'lambda_network_layers': lambda_network_layers,
            'optimizer_lambda': optimizer_lambda,
            'loss_lambda': loss_lambda,        
             #'list_of_monomial_identifiers': list_of_monomial_identifiers,
             'x_min': x_min,
             'x_max': x_max,
             'sparse_poly_representation_version': sparse_poly_representation_version,
            'max_optimization_minutes': max_optimization_minutes,
             }

    parallel_polynomial_regression = Parallel(n_jobs=n_jobs, verbose=11, backend=backend)

    return_error = False
    
    result_list_polynomial_regression = parallel_polynomial_regression(delayed(polynomial_regression)(lambda_net, 
                                                                                  config,
                                                                                  polynomial_regression_hyperparams,
                                                                                  printing = printing,
                                                                                  return_error = return_error) for lambda_net in lambda_net_dataset.lambda_net_list)      

    del parallel_polynomial_regression  
    
    if return_error:
        polynomial_regression_errors = [result[0] for result in result_list_polynomial_regression]
        polynomial_regression_functions = [result[1] for result in result_list_polynomial_regression]   
        polynomial_regression_runtimes = [result[2] for result in result_list_polynomial_regression]   
    else:
        polynomial_regression_functions = [result[0] for result in result_list_polynomial_regression]   
        polynomial_regression_runtimes = [result[1] for result in result_list_polynomial_regression] 
        
        return polynomial_regression_functions, polynomial_regression_runtimes
    
    return polynomial_regression_errors, polynomial_regression_functions, polynomial_regression_runtimes



def symbolic_metamodeling_function_generation(lambda_net_dataset, return_expression='approx', function_metamodeling=True, force_polynomial=False, backend='loky'):
                
    printing = True if n_jobs == 1 else False
        
    metamodeling_hyperparams = {
        'num_iter': 10,#500,
        'batch_size': None,
        'learning_rate': 0.01,        
        'dataset_size': per_network_optimization_dataset_size,
    }

    #list_of_monomial_identifiers_numbers = np.array([list(monomial_identifiers) for monomial_identifiers in list_of_monomial_identifiers]).astype(float)  

    #printing = True if n_jobs == 1 else False

    #lambda_network_weights_list = np.array(lambda_net_dataset.weight_list)
    #print('HERE')
    #backend = 'sequential'

    config = {
            'n': n,
            'd': d,
            'inet_loss': inet_loss,
            'sparsity': sparsity,
            'lambda_network_layers': lambda_network_layers,
            'interpretation_net_output_shape': interpretation_net_output_shape,
            'RANDOM_SEED': RANDOM_SEED,
            'nas': nas,
            'number_of_lambda_weights': number_of_lambda_weights,
            'interpretation_net_output_monomials': interpretation_net_output_monomials,
            'fixed_initialization_lambda_training': fixed_initialization_lambda_training,
            'dropout': dropout,
            'lambda_network_layers': lambda_network_layers,
            'optimizer_lambda': optimizer_lambda,
            'loss_lambda': loss_lambda,        
             #'list_of_monomial_identifiers': list_of_monomial_identifiers,
             'x_min': x_min,
             'x_max': x_max,
            'sparse_poly_representation_version': sparse_poly_representation_version,
            'max_optimization_minutes': max_optimization_minutes,
             }
    
    parallel_metamodeling = Parallel(n_jobs=n_jobs, verbose=11, backend=backend)

    return_error = False 
    
    force_polynomial = True 
    
    if adjusted_symbolic_metamodeling_code:
    
        result_list_metamodeling = parallel_metamodeling(delayed(symbolic_metamodeling)(lambda_net, 
                                                                                      config,
                                                                                      metamodeling_hyperparams,
                                                                                      printing = printing,
                                                                                      return_error = return_error,
                                                                                      return_expression=return_expression,
                                                                                      function_metamodeling=function_metamodeling,
                                                                                      force_polynomial=force_polynomial) for lambda_net in lambda_net_dataset.lambda_net_list)      

    else:
        
        result_list_metamodeling = parallel_metamodeling(delayed(symbolic_metamodeling_original)(lambda_net, 
                                                                                      config,
                                                                                      metamodeling_hyperparams,
                                                                                      printing = printing,
                                                                                      return_error = return_error,
                                                                                      return_expression=return_expression,
                                                                                      function_metamodeling=function_metamodeling,
                                                                                      force_polynomial=force_polynomial) for lambda_net in lambda_net_dataset.lambda_net_list)          
        
    del parallel_metamodeling  
    
    if return_error:
        metamodeling_errors = [result[0] for result in result_list_metamodeling]
        metamodeling_polynomials = [result[1] for result in result_list_metamodeling]   
        metamodeling_runtimes = [result[2] for result in result_list_metamodeling]   
    else:
        metamodeling_polynomials = [result[0] for result in result_list_metamodeling]   
        metamodeling_runtimes = [result[1] for result in result_list_metamodeling]          
        return metamodeling_polynomials, metamodeling_runtimes
        
    
    return metamodeling_errors, metamodeling_polynomials, metamodeling_runtimes
    
    
    
def reduce_polynomials(polynomial_list):
    
    return
    
#######################################################################################################################################################
################################################################SAVING AND PLOTTING RESULTS############################################################
#######################################################################################################################################################    
    
    
def generate_history_plots(history_list, by='epochs'):
    
    paths_dict = generate_paths(path_type = 'interpretation_net')
    
    for i, history in enumerate(history_list):  
        
        if by == 'epochs':
            index= (i+1)*each_epochs_save_lambda if each_epochs_save_lambda==1 else i*each_epochs_save_lambda if i > 1 else each_epochs_save_lambda if i==1 else 1
        elif by == 'samples':
            index = i
        
        plt.plot(history[list(history.keys())[1]])
        plt.plot(history[list(history.keys())[len(history.keys())//2+1]])
        plt.title('model ' + list(history.keys())[len(history.keys())//2+1])
        plt.ylabel('metric')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        if by == 'epochs':
            plt.savefig('./data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/' + list(history.keys())[len(history.keys())//2+1] + '_epoch_' + str(index).zfill(3) + '.png')
        elif by == 'samples':
            plt.savefig('./data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/' + list(history.keys())[len(history.keys())//2+1] + '_samples_' + str(samples_list[index]).zfill(5) + '.png')
        plt.clf()
        
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        if by == 'epochs':
            plt.savefig('./data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/loss_' + '_epoch_' + str(index).zfill(3) + '.png')    
        elif by == 'samples':
            plt.savefig('./data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/loss_' + '_samples_' + str(samples_list[index]).zfill(5) + '.png')    
        if i < len(history_list)-1:
            plt.clf() 
            
            
def save_results(history_list=None, scores_list=None, by='epochs'):
    
    paths_dict = generate_paths(path_type = 'interpretation_net')
    if history_list is not None:
        if by == 'epochs':
            path = './data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/history_epochs' + '.pkl'
        elif by == 'samples':
            path = './data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/history_samples' + '.pkl'
        with open(path, 'wb') as f:
            pickle.dump(history_list, f, protocol=2)   
        
    if scores_list is not None: 
        if by == 'epochs':
            path = './data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/scores_epochs' + '.pkl'
        elif by == 'samples':
            path = './data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/scores_samples' + '.pkl'    
        with open(path, 'wb') as f:
            pickle.dump(scores_list, f, protocol=2)  
        

def generate_inet_comparison_plot(scores_list, plot_metric_list, ylim=None):
        
    paths_dict = generate_paths(path_type = 'interpretation_net')
    
    
    keys = ['target_polynomials', 'lstsq_target_polynomials', 'lstsq_lambda_pred_polynomials', 'inet_polynomials', 'per_network_polynomials',  'metamodel_poly', 'metamodel_functions', 'metamodel_functions_no_GD', 'symbolic_regression_functions']
    
    evaluation_key_list = []
    for combination in itertools.combinations(keys, r=2):
        key_1 = combination[0]
        key_2 = combination[1]
        
        evaluation_key = key_1 + '_VS_' + key_2
        
        if evaluation_key in scores_list.index:
            evaluation_key_list.append(evaluation_key)
    
    epochs_save_range_lambda = range(epoch_start//each_epochs_save_lambda, epochs_lambda//each_epochs_save_lambda) if each_epochs_save_lambda == 1 else range(epoch_start//each_epochs_save_lambda, epochs_lambda//each_epochs_save_lambda+1) if multi_epoch_analysis else range(1,2)


    if samples_list == None:
        x_axis_steps = [(i+1)*each_epochs_save_lambda if each_epochs_save_lambda==1 else i*each_epochs_save_lambda if i > 1 else each_epochs_save_lambda if i==1 else 1 for i in epochs_save_range_lambda]
        x_max = epochs_lambda
    else:
        x_axis_steps = samples_list
        x_max = samples_list[-1]

    #Plot Polynom, lamdba net, and Interpration net
    length_plt = len(plot_metric_list)
    if length_plt >= 2:
        fig, ax = plt.subplots(length_plt//2, 2, figsize=(30,20))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(20,10))

    for index, metric in enumerate(plot_metric_list):
        
        plot_scores_dict = {}
        for key in evaluation_key_list:
            try:
                scores_list[-1][metric].loc[key]
                plot_scores_dict[key] = []
            except:
                #print(key + 'not in scores_list')
                continue
            
        
        for scores in scores_list:
            for key in evaluation_key_list:
                try:
                    plot_scores_dict[key].append(scores[metric].loc[key])
                except:
                    #print(key + 'not in scores_list')
                    continue
                                        
            
        plot_df = pd.DataFrame(data=np.vstack(plot_scores_dict.values()).T, 
                               index=x_axis_steps,
                               columns=plot_scores_dict.keys())

        if length_plt >= 2:
            ax[index//2, index%2].set_title(metric)
            sns.set(font_scale = 1.25)
            p = sns.lineplot(data=plot_df, ax=ax[index//2, index%2])
        else:
            ax.set_title(metric)
            sns.set(font_scale = 1.25)
            p = sns.lineplot(data=plot_df, ax=ax)

        if ylim != None:
            p.set(ylim=ylim)

        p.set_yticklabels(np.round(p.get_yticks(), 2), size = 20)
        p.set_xticklabels(p.get_xticks(), size = 20)     
        
        #p.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        p.legend(loc='upper center', bbox_to_anchor=(0.47, -0.1),
          fancybox=False, shadow=False, ncol=2, fontsize=12)   
        
    plt.subplots_adjust(wspace=0.1, hspace=0.75)
    
    location = './data/plotting/'
    folder = paths_dict['path_identifier_interpretation_net_data'] + '/'
    if samples_list == None:
        file = 'multi_epoch' + '.pdf'
    else:
        file = 'sample_list' + '-'.join([str(samples_list[0]), str(samples_list[-1])]) + '.pdf'

    path = location + folder + file

    plt.savefig(path, format='pdf')
    plt.show()






def plot_and_save_single_polynomial_prediction_evaluation(lambda_net_test_dataset_list, function_values_test_list, polynomial_dict_test_list, rand_index=1, plot_type=2):
    
    paths_dict = generate_paths(path_type = 'interpretation_net')
    
    x_vars = ['x' + str(i) for i in range(1, n+1)]

    columns = x_vars.copy()
    columns.append('FVs')

    columns_single = x_vars.copy()

    vars_plot = lambda_net_test_dataset_list[-1].X_test_data_list[rand_index]    
    
    custom_representation_keys_fixed = ['target_polynomials']#['target_polynomials', 'lstsq_target_polynomials', 'lstsq_lambda_pred_polynomials']
    custom_representation_keys_dynamic = ['inet_polynomials']#['inet_polynomials', 'per_network_polynomials']
    sympy_representation_keys = ['metamodel_functions', 'metamodel_poly', 'symbolic_regression_functions']#['metamodel_poly', 'metamodel_functions', 'metamodel_functions_no_GD', 'symbolic_regression_functions']
    
    #keys = ['target_polynomials', 'lstsq_target_polynomials', 'lstsq_lambda_pred_polynomials', 'inet_polynomials', 'per_network_polynomials', 'metamodel_functions']
    
    
    
    
    #lambda_train_data = lambda_net_test_dataset_list[-1].y_test_data_list[rand_index].ravel()
    #lambda_train_data_size = lambda_train_data.shape[0]
    #lambda_train_data_str = np.array(['Lambda Train Data' for i in range(lambda_train_data_size)])  
    #columns_single.append('Lambda Train Data')
    
    lambda_model_preds = function_values_test_list[-1]['lambda_preds'][rand_index].ravel()
    eval_size_plot = lambda_model_preds.shape[0]
    lambda_model_preds_str = np.array(['Lambda Model Preds' for i in range(eval_size_plot)])
    columns_single.append('Lambda Model Preds')
    
    identifier_list = [lambda_model_preds_str]#[lambda_train_data_str, lambda_model_preds_str]
    plot_data_single_list = [vars_plot, lambda_model_preds]#[vars_plot, lambda_train_data, lambda_model_preds]
    for key in custom_representation_keys_fixed:
        try:
            polynomial_by_key = polynomial_dict_test_list[-1][key][rand_index]
        except:
            continue            
        polynomial_by_key_string = get_sympy_string_from_coefficients(polynomial_by_key, force_complete_poly_representation=True, round_digits=4)
        polynomial_by_key_fvs = function_values_test_list[-1][key][rand_index]                
            
        plot_data_single_list.append(polynomial_by_key_fvs)
        columns_single.append(key)
        identifier_list.append(np.array([key for i in range(eval_size_plot)]))
    
    for key in custom_representation_keys_dynamic:
        try:
            polynomial_by_key = polynomial_dict_test_list[-1][key][rand_index]
        except:
            continue        
        polynomial_by_key_string = get_sympy_string_from_coefficients(polynomial_by_key, round_digits=4)
        polynomial_by_key_fvs = function_values_test_list[-1][key][rand_index]                
            
        plot_data_single_list.append(polynomial_by_key_fvs)
        columns_single.append(key)
        identifier_list.append(np.array([key for i in range(eval_size_plot)]))
        
    for key in sympy_representation_keys:
        try:
            function_by_key = polynomial_dict_test_list[-1][key][rand_index]
        except:
            continue
        function_by_key_string = str(function_by_key)
        function_by_key_fvs = function_values_test_list[-1][key][rand_index]                
            
        plot_data_single_list.append(function_by_key_fvs)
        columns_single.append(key)
        identifier_list.append(np.array([key for i in range(eval_size_plot)]))        
    
    identifier = np.concatenate(identifier_list)
    plot_data_single = pd.DataFrame(data=np.column_stack(plot_data_single_list), columns=columns_single)
    vars_plot_all_preds = np.vstack([vars_plot for i in range(len(columns_single[n:]))])
    preds_plot_all = np.vstack(plot_data_single_list[1:]).ravel()         
        
    plot_data = pd.DataFrame(data=np.column_stack([vars_plot_all_preds, preds_plot_all]), columns=columns)
    plot_data['Identifier'] = identifier       
     
    
    location = './data/plotting/'
    folder = paths_dict['path_identifier_interpretation_net_data'] + '/'
        
    if plot_type == 1:
        
        
        pp = sns.pairplot(data=plot_data,
                      #kind='reg',
                      hue='Identifier',
                      y_vars=['FVs'],
                      x_vars=x_vars, 
                      height=7.5,
                      aspect=2)
        file = 'pp3in1_' + str(rand_index).zfill(3) + '.pdf'                 
        
    elif plot_type == 2:

        pp = sns.pairplot(data=plot_data,
                          #kind='reg',
                          hue='Identifier',
                          #y_vars=['FVs'],
                          #x_vars=x_vars, 
                          height=10//n)
             
        file = 'pp3in1_extended_' + str(rand_index).zfill(3) + '.pdf'  
        
    elif plot_type == 3:
        
        pp = sns.pairplot(data=plot_data_single,
                          #kind='reg',
                          y_vars=columns_single[n:],
                          x_vars=x_vars, 
                          height=3,
                          aspect=3)

        file = 'pp1_' + str(rand_index).zfill(3) + '.pdf'                   
        
    path = location + folder + file
    pp.savefig(path, format='pdf')
    plt.show()    
    
    if False:
        real_poly_VS_lstsq_target_poly_mae = mean_absolute_error(real_poly_fvs, lstsq_target_poly)
        real_poly_VS_lstsq_target_poly_r2 = r2_score(real_poly_fvs, lstsq_target_poly)        

        real_poly_VS_inet_poly_mae = mean_absolute_error(real_poly_fvs, inet_poly_fvs)
        real_poly_VS_inet_poly_r2 = r2_score(real_poly_fvs, inet_poly_fvs)    

        real_poly_VS_perNet_poly_mae = mean_absolute_error(real_poly_fvs, per_network_opt_poly_fvs)
        real_poly_VS_perNet_poly_r2 = r2_score(real_poly_fvs, per_network_opt_poly_fvs)    

        real_poly_VS_lambda_model_preds_mae = mean_absolute_error(real_poly_fvs, lambda_model_preds)
        real_poly_VS_lambda_model_preds_r2 = r2_score(real_poly_fvs, lambda_model_preds)

        real_poly_VS_lstsq_lambda_preds_poly_mae = mean_absolute_error(real_poly_fvs, lstsq_lambda_preds_poly)
        real_poly_VS_lstsq_lambda_preds_poly_r2 = r2_score(real_poly_fvs, lstsq_lambda_preds_poly)   

        from prettytable import PrettyTable

        tab = PrettyTable()

        tab.field_names = ["Comparison",  "MAE", "R2-Score", "Poly 1", "Poly 2"]
        tab._max_width = {"Poly 1" : 50, "Poly 2" : 50}

        tab.add_row(["Target Poly \n vs. \n LSTSQ Target Poly \n", real_poly_VS_lstsq_target_poly_mae, real_poly_VS_lstsq_target_poly_r2, polynomial_target_string, polynomial_lstsq_target_string])
        tab.add_row(["Target Poly \n vs. \n I-Net Poly \n", real_poly_VS_inet_poly_mae, real_poly_VS_inet_poly_r2, polynomial_target_string, polynomial_inet_string])
        tab.add_row(["Target Poly \n vs. \n Per Network Opt Poly \n", real_poly_VS_perNet_poly_mae, real_poly_VS_perNet_poly_r2, polynomial_target_string, polynomial_per_network_opt_string])
        tab.add_row(["Target Poly \n vs. \n Lambda Preds \n", real_poly_VS_lambda_model_preds_mae, real_poly_VS_lambda_model_preds_r2, polynomial_target_string, '-'])
        tab.add_row(["Target Poly \n vs. \n LSTSQ Lambda Preds Poly \n", real_poly_VS_lstsq_lambda_preds_poly_mae, real_poly_VS_lstsq_lambda_preds_poly_r2, polynomial_target_string, polynomial_lstsq_lambda_string])

        print(tab)

            
def restructure_data_cnn_lstm(X_data, version=2, subsequences=None):

    #version == 0: one sequence for biases and one sequence for weights per layer (padded to maximum size)
    #version == 1: each path from input bias to output bias combines in one sequence for biases and one sequence for weights per layer (no. columns == number of paths and no. rows = number of layers/length of path)
    #version == 2:each path from input bias to output bias combines in one sequence for biases and one sequence for weights per layer + transpose matrices  (no. columns == number of layers/length of path and no. rows = number of paths )
    
    base_model = generate_base_model()
       
    X_data_flat = X_data

    shaped_weights_list = []
    for data in tqdm(X_data):
        shaped_weights = shape_flat_weights(data, base_model.get_weights())
        shaped_weights_list.append(shaped_weights)

    max_size = 0
    for weights in shaped_weights:
        max_size = max(max_size, max(weights.shape))      


    if version == 0: #one sequence for biases and one sequence for weights per layer (padded to maximum size)
        X_data_list = []
        for shaped_weights in tqdm(shaped_weights_list):
            padded_network_parameters_list = []
            for layer_weights, biases in pairwise(shaped_weights):
                padded_weights_list = []
                for weights in layer_weights:
                    padded_weights = np.pad(weights, (int(np.floor((max_size-weights.shape[0])/2)), int(np.ceil((max_size-weights.shape[0])/2))), 'constant')
                    padded_weights_list.append(padded_weights)
                padded_biases = np.pad(biases, (int(np.floor((max_size-biases.shape[0])/2)), int(np.ceil((max_size-biases.shape[0])/2))), 'constant')
                padded_network_parameters_list.append(padded_biases)
                padded_network_parameters_list.extend(padded_weights_list)   
            X_data_list.append(padded_network_parameters_list)
        X_data = np.array(X_data_list)    

    elif version == 1 or version == 2: #each path from input bias to output bias combines in one sequence for biases and one sequence for weights per layer
        lambda_net_structure = list(flatten([n, lambda_network_layers, 1]))                    
        number_of_paths = reduce(lambda x, y: x * y, lambda_net_structure)

        X_data_list = []
        for shaped_weights in tqdm(shaped_weights_list):        
            network_parameters_sequence_list = np.array([]).reshape(number_of_paths, 0)    
            for layer_index, (weights, biases) in zip(range(1, len(lambda_net_structure)), pairwise(shaped_weights)):

                layer_neurons = lambda_net_structure[layer_index]    
                previous_layer_neurons = lambda_net_structure[layer_index-1]

                assert biases.shape[0] == layer_neurons
                assert weights.shape[0]*weights.shape[1] == previous_layer_neurons*layer_neurons

                bias_multiplier = number_of_paths//layer_neurons
                weight_multiplier = number_of_paths//(previous_layer_neurons * layer_neurons)

                extended_bias_list = []
                for bias in biases:
                    extended_bias = np.tile(bias, (bias_multiplier,1))
                    extended_bias_list.extend(extended_bias)


                extended_weights_list = []
                for weight in weights.flatten():
                    extended_weights = np.tile(weight, (weight_multiplier,1))
                    extended_weights_list.extend(extended_weights)      

                network_parameters_sequence = np.concatenate([extended_weights_list, extended_bias_list], axis=1)
                network_parameters_sequence_list = np.hstack([network_parameters_sequence_list, network_parameters_sequence])


            number_of_paths = network_parameters_sequence_list.shape[0]
            number_of_unique_paths = np.unique(network_parameters_sequence_list, axis=0).shape[0]
            number_of_nonUnique_paths = number_of_paths-number_of_unique_paths

            if number_of_nonUnique_paths > 0:
                print("Number of non-unique rows: " + str(number_of_nonUnique_paths))
                print(network_parameters_sequence_list)

            X_data_list.append(network_parameters_sequence_list)
        X_data = np.array(X_data_list)          
        
        if version == 2: #transpose matrices (if false, no. columns == number of paths and no. rows = number of layers/length of path)
            X_data = np.transpose(X_data, (0, 2, 1))

    if lstm_layers != None and cnn_layers != None: #generate subsequences for cnn-lstm
        subsequences = 1 #for each bias+weights
        timesteps = X_train.shape[1]//subsequences

        X_data = X_data.reshape((X_data.shape[0], subsequences, timesteps, X_data.shape[2]))

    return X_data, X_data_flat