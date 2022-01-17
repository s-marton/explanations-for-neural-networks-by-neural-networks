#######################################################################################################################################################
#######################################################################Imports#########################################################################
#######################################################################################################################################################

#from itertools import product       # forms cartesian products
#from tqdm import tqdm_notebook as tqdm
#import pickle
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


import tensorflow as tf
import random 

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#udf import
#from utilities.LambdaNet import *
from utilities.metrics import *
from utilities.utility_functions import *

import copy

#######################################################################################################################################################
#############################################################Setting relevant parameters from current config###########################################
#######################################################################################################################################################

def initialize_LambdaNet_config_from_curent_notebook(config):
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
##################################################################Lambda Net Wrapper###################################################################
#######################################################################################################################################################
class LambdaNetDataset():
    lambda_net_list = None
    
    weight_list = None
    
    train_settings_list = None
    index_list = None
    
    target_polynomial_list = None
    lstsq_lambda_pred_polynomial_list = None
    lstsq_target_polynomial_list = None    
        
    X_test_data_list = None
    y_test_data_list = None
    
    def __init__(self, lambda_net_list):
        
        self.lambda_net_list = lambda_net_list
        
        self.weight_list = [lambda_net.weights for lambda_net in lambda_net_list]
        
        self.train_settings_list = {}
        for key in lambda_net_list[0].train_settings.keys():
            self.train_settings_list[key] = []   
        for lambda_net in lambda_net_list:
            for key in lambda_net.train_settings.keys():
                self.train_settings_list[key].append(lambda_net.train_settings[key])
        
        self.index_list = [lambda_net.index for lambda_net in lambda_net_list]
        
        self.target_polynomial_list = [lambda_net.target_polynomial for lambda_net in lambda_net_list]
        self.lstsq_lambda_pred_polynomial_list = [lambda_net.lstsq_lambda_pred_polynomial for lambda_net in lambda_net_list]
        self.lstsq_target_polynomial_list = [lambda_net.lstsq_target_polynomial for lambda_net in lambda_net_list]
      
        self.X_test_data_list = [lambda_net.X_test_data for lambda_net in lambda_net_list]
        self.y_test_data_list = [lambda_net.y_test_data for lambda_net in lambda_net_list]
    
    def __repr__(self):
        return str(self.as_pandas().head())
    def __str__(self):
        return str(self.as_pandas().head())
    
    def __len__(self):
        return len(self.lambda_net_list)
    

        
    def make_prediction_on_dataset(self, evaluation_dataset):  
        assert evaluation_dataset.shape[1] == n
        lambda_network_preds_list = []
        
        for weights in self.weight_list:
            lambda_network_preds = weights_to_pred(weights, evaluation_dataset)
            lambda_network_preds_list.append(lambda_network_preds)
        
        return np.array(lambda_network_preds_list)
    
    def make_prediction_on_test_data(self):
        lambda_network_preds_list = []
        for lambda_net in self.lambda_net_list:
            lambda_network_preds = lambda_net.make_prediction_on_test_data()
            lambda_network_preds_list.append(lambda_network_preds)
            
        return np.array(lambda_network_preds_list)
                
        
    def return_target_poly_fvs_on_dataset(self, evaluation_dataset, n_jobs_parallel_fv=10, backend='threading'):       
        from utilities.utility_functions import parallel_fv_calculation_from_polynomial
        
        assert evaluation_dataset.shape[1] == n, 'evaluation dataset has wrong shape ' + str(evaluation_dataset.shape) + ' but required (x, ' + str(n) + ')'  
        target_poly_fvs_list = parallel_fv_calculation_from_polynomial(self.target_polynomial_list, [evaluation_dataset for _ in range(len(self.target_polynomial_list))], force_complete_poly_representation=True, n_jobs_parallel_fv=10, backend='threading')
            
        return np.array(target_poly_fvs_list)
    
    def return_target_poly_fvs_on_test_data(self, n_jobs_parallel_fv=10, backend='threading'):          
        from utilities.utility_functions import parallel_fv_calculation_from_polynomial
        
        target_poly_fvs_list = parallel_fv_calculation_from_polynomial(self.target_polynomial_list, self.X_test_data_list, force_complete_poly_representation=True, n_jobs_parallel_fv=10, backend='threading')
        
        return np.array(target_poly_fvs_list)
    
    def return_lstsq_lambda_pred_polynomial_fvs_on_dataset(self, evaluation_dataset, n_jobs_parallel_fv=10, backend='threading'):
        from utilities.utility_functions import parallel_fv_calculation_from_polynomial
        
        assert evaluation_dataset.shape[1] == n, 'evaluation dataset has wrong shape ' + str(evaluation_dataset.shape) + ' but required (x, ' + str(n) + ')'
        lstsq_lambda_pred_polynomial_fvs_list = parallel_fv_calculation_from_polynomial(self.lstsq_lambda_pred_polynomial_list, [evaluation_dataset for _ in range(len(self.target_polynomial_list))], force_complete_poly_representation=True, n_jobs_parallel_fv=10, backend='threading')
            
        return np.array(lstsq_lambda_pred_polynomial_fvs_list)
    
    def return_lstsq_lambda_pred_polynomial_fvs_on_test_data(self, n_jobs_parallel_fv=10, backend='threading'):
        from utilities.utility_functions import parallel_fv_calculation_from_polynomial
        
        lstsq_lambda_pred_polynomial_fvs_list = parallel_fv_calculation_from_polynomial(self.lstsq_lambda_pred_polynomial_list, self.X_test_data_list, force_complete_poly_representation=True, n_jobs_parallel_fv=10, backend='threading')
            
        return np.array(lstsq_lambda_pred_polynomial_fvs_list)
    
    def return_lstsq_target_polynomial_fvs_on_dataset(self, evaluation_dataset, n_jobs_parallel_fv=10, backend='threading'):
        from utilities.utility_functions import parallel_fv_calculation_from_polynomial
        
        assert evaluation_dataset.shape[1] == n, 'evaluation dataset has wrong shape ' + str(evaluation_dataset.shape) + ' but required (x, ' + str(n) + ')'
        lstsq_target_polynomial_fvs_list = parallel_fv_calculation_from_polynomial(self.lstsq_target_polynomial_list, [evaluation_dataset for _ in range(len(self.target_polynomial_list))], force_complete_poly_representation=True, n_jobs_parallel_fv=10, backend='threading')
            
        return np.array(lstsq_target_polynomial_fvs_list)
    
    def return_lstsq_target_polynomial_fvs_on_test_data(self, n_jobs_parallel_fv=10, backend='threading'):
        from utilities.utility_functions import parallel_fv_calculation_from_polynomial
        
        lstsq_target_polynomial_fvs_list = parallel_fv_calculation_from_polynomial(self.lstsq_target_polynomial_list, self.X_test_data_list, force_complete_poly_representation=True, n_jobs_parallel_fv=10, backend='threading')
            
        return np.array(lstsq_target_polynomial_fvs_list)
    
    def as_pandas(self):  
                
        lambda_dataframe = pd.DataFrame(data=[lambda_net.as_array() for lambda_net in self.lambda_net_list], 
                                columns=self.lambda_net_list[0].return_column_names(), 
                                index=[lambda_net.index for lambda_net in self.lambda_net_list])
        lambda_dataframe['seed'] = lambda_dataframe['seed'].astype(int)
        
        return lambda_dataframe

    
    def get_lambda_nets_by_seed(self, seed_list):
        lambda_nets_by_seed = []
        for lambda_net in self.lambda_net_list:
            if lambda_net.train_settings['seed'] in seed_list:
                lambda_nets_by_seed.append(lambda_net)
    
        return LambdaNetDataset(lambda_nets_by_seed)
    
    def get_lambda_nets_by_lambda_index(self, lambda_index_list):
        lambda_nets_by_lambda_index = []
        for lambda_net in self.lambda_net_list:
            if lambda_net.index in lambda_index_list:
                lambda_nets_by_lambda_index.append(lambda_net)
    
        return LambdaNetDataset(lambda_nets_by_lambda_index) 
    
    def get_lambda_net_by_lambda_index(self, lambda_index):
        for lambda_net in self.lambda_net_list:
            if lambda_net.index in lambda_index:
                return lambda_net
    
        return None
    
    def sample(self, size, seed=42):
        
        assert isinstance(size, int) or isinstance(size, float), 'Wrong sample size specified'
        
        random.seed(seed)
        
        sample_lambda_net_list = None
        if isinstance(size, int):
            sample_lambda_net_list = random.sample(self.lambda_net_list, size)
        elif isinstance(size, float):
            size = int(np.round(len(self.lambda_net_list)*size))
            sample_lambda_net_list = random.sample(self.lambda_net_list, size)
            
        return LambdaNetDataset(sample_lambda_net_list)
    

class LambdaNet():
    weights = None
    model = None
    
    train_settings = None
    index = None
    
    target_polynomial = None
    lstsq_lambda_pred_polynomial = None
    lstsq_target_polynomial = None
    
    X_test_data = None
    y_test_data = None
    
    def __init__(self, line_weights, line_X_data, line_y_data):
        assert isinstance(line_weights, np.ndarray), 'line is no array: ' + str(line_weights) 
        
        from utilities.utility_functions import shaped_network_parameters_to_array, normal_neural_net, shape_flat_network_parameters, generate_base_model
        
        self.index = int(line_weights[0])
        try:
            self.train_settings = {'seed': int(line_weights[1])}
        except ValueError:
            self.train_settings = {'seed': -1}
            
        self.target_polynomial = line_weights[range(2, sparsity+2)].astype(float)
        self.lstsq_lambda_pred_polynomial = line_weights[range(sparsity+2, sparsity*2+2)].astype(float)
        self.lstsq_target_polynomial = line_weights[range(sparsity*2+2, sparsity*3+2)].astype(float)
        assert self.target_polynomial.shape[0] == sparsity, 'target polynomial has incorrect shape ' + str(self.target_polynomial.shape[0]) + ' but should be ' + str(sparsity)
        assert self.lstsq_lambda_pred_polynomial.shape[0] == sparsity, 'lstsq lambda pred polynomial has incorrect shape ' + str(self.lstsq_lambda_pred_polynomial.shape[0]) + ' but should be ' + str(sparsity)
        assert self.lstsq_target_polynomial.shape[0] == sparsity, 'lstsq target polynomial has incorrect shape ' + str(self.lstsq_target_polynomial.shape[0]) + ' but should be ' + str(sparsity)
        
        self.weights = line_weights[sparsity*3+2:].astype(float)
        
        assert self.weights.shape[0] == number_of_lambda_weights, 'weights have incorrect shape ' + str(self.weights.shape[0]) + ' but should be ' + str(number_of_lambda_weights)
        
        line_X_data = line_X_data[1:]
        line_y_data = line_y_data[1:]
        self.X_test_data = np.transpose(np.array([line_X_data[i::n] for i in range(n)]))
        self.y_test_data = line_y_data.reshape(-1,1)
                
        if normalize_lambda_nets:
            if self.index == 1:
                print('NORMALIZE PRE')
                print(self.weights)
                print(weights_to_pred(self.weights, self.X_test_data[:5]))
            self.weights = shaped_network_parameters_to_array(normal_neural_net(shape_flat_network_parameters(copy.deepcopy(self.weights), generate_base_model().get_weights())))            
            if self.index == 1:
                print('NORMALIZE AFTER')
                print(self.weights)    
                print(weights_to_pred(self.weights, self.X_test_data[:5]))
        

    def __repr__(self):
        return str(self.weights)
    def __str__(self):
        return str(self.weights)
        
    def make_prediction_on_dataset(self, evaluation_dataset):  
        assert evaluation_dataset.shape[1] == n
        lambda_network_preds = weights_to_pred(self.weights, evaluation_dataset)
        
        return lambda_network_preds
    
    def make_prediction_on_test_data(self):        
        lambda_network_preds = weights_to_pred(self.weights, self.X_test_data)
        
        return lambda_network_preds               
        
    def return_target_poly_fvs_on_dataset(self, evaluation_dataset, n_jobs_parallel_fv=10, backend='threading'):
        from utilities.utility_functions import parallel_fv_calculation_from_polynomial
        
        assert evaluation_dataset.shape[1] == n, 'evaluation dataset has wrong shape ' + str(evaluation_dataset.shape) + ' but required (x, ' + str(n) + ')'
        target_poly_fvs = parallel_fv_calculation_from_polynomial([self.target_polynomial], [evaluation_dataset], force_complete_poly_representation=True, n_jobs_parallel_fv=10, backend='threading')
    
        return target_poly_fvs
    
    def return_target_poly_fvs_on_test_data(self, n_jobs_parallel_fv=10, backend='threading'):
        from utilities.utility_functions import parallel_fv_calculation_from_polynomial
        
        target_poly_fvs = parallel_fv_calculation_from_polynomial([self.target_polynomial], [self.X_test_data], force_complete_poly_representation=True, n_jobs_parallel_fv=10, backend='threading')
    
        return target_poly_fvs    
    
    
    
    def return_lstsq_lambda_pred_polynomial_fvs_on_dataset(self, evaluation_dataset, n_jobs_parallel_fv=10, backend='threading'):
        from utilities.utility_functions import parallel_fv_calculation_from_polynomial
        
        assert evaluation_dataset.shape[1] == n, 'evaluation dataset has wrong shape ' + str(evaluation_dataset.shape) + ' but required (x, ' + str(n) + ')'
        lstsq_lambda_pred_polynomial_fvs = parallel_fv_calculation_from_polynomial([self.lstsq_lambda_pred_polynomial], [evaluation_dataset], force_complete_poly_representation=True, n_jobs_parallel_fv=10, backend='threading')
    
        return lstsq_lambda_pred_polynomial_fvs
    
    def return_lstsq_lambda_pred_polynomial_fvs_on_test_data(self, n_jobs_parallel_fv=10, backend='threading'):
        from utilities.utility_functions import parallel_fv_calculation_from_polynomial
        
        lstsq_lambda_pred_polynomial_fvs = parallel_fv_calculation_from_polynomial([self.lstsq_lambda_pred_polynomial], [self.X_test_data], force_complete_poly_representation=True, n_jobs_parallel_fv=10, backend='threading')
    
        return lstsq_lambda_pred_polynomial_fvs     
    
    def return_lstsq_target_polynomial_fvs_on_dataset(self, evaluation_dataset, n_jobs_parallel_fv=10, backend='threading'):
        from utilities.utility_functions import parallel_fv_calculation_from_polynomial
        
        assert evaluation_dataset.shape[1] == n, 'evaluation dataset has wrong shape ' + str(evaluation_dataset.shape) + ' but required (x, ' + str(n) + ')'
        lstsq_target_polynomial_fvs = parallel_fv_calculation_from_polynomial([self.lstsq_target_polynomial], [evaluation_dataset], force_complete_poly_representation=True, n_jobs_parallel_fv=10, backend='threading')
    
        return lstsq_target_polynomial_fvs
    
    def return_lstsq_target_polynomial_fvs_on_test_data(self, n_jobs_parallel_fv=10, backend='threading'):
        from utilities.utility_functions import parallel_fv_calculation_from_polynomial
        
        lstsq_target_polynomial_fvs = parallel_fv_calculation_from_polynomial([self.lstsq_target_polynomial], [self.X_test_data], force_complete_poly_representation=True, n_jobs_parallel_fv=10, backend='threading')
    
        return lstsq_target_polynomial_fvs  
    
    def as_pandas(self): 
        columns = return_column_names(self)
        data = as_array(self)
        
        df = pd.DataFrame(data=data, columns=columns, index=[self.index])
        df['seed'] = df['seed'].astype(int)
        
        return df
    
    def as_array(self):
        data = np.hstack([self.train_settings['seed'], self.target_polynomial, self.lstsq_lambda_pred_polynomial, self.lstsq_target_polynomial, self.weights])
        return data
    
    def return_column_names(self):  
        
        from utilities.utility_functions import flatten
        
        list_of_monomial_identifiers_string = [''.join(str(e) for e in monomial_identifier) for monomial_identifier in list_of_monomial_identifiers] if n > 1 else [str(monomial_identifier[0]) for monomial_identifier in list_of_monomial_identifiers]
        
        target_polynomial_identifiers = [monomial_identifiers + str('-target') for monomial_identifiers in list_of_monomial_identifiers_string]
        lstsq_lambda_pred_polynomial_identifiers = [monomial_identifiers + str('-lstsq_lambda') for monomial_identifiers in list_of_monomial_identifiers_string]
        lstsq_target_polynomial_identifiers = [monomial_identifiers + str('-lstsq_target') for monomial_identifiers in list_of_monomial_identifiers_string]

        weight_identifiers = ['wb_' + str(i) for i in range(self.weights.shape[0])]
        
        columns = list(flatten(['seed', target_polynomial_identifiers, lstsq_lambda_pred_polynomial_identifiers, lstsq_target_polynomial_identifiers, weight_identifiers]))
                
        return columns 

    def return_model(self, config=None):
        model = weights_to_model(self.weights, config)
        
        return model    
    
        
def split_LambdaNetDataset(dataset, test_split, random_seed='RANDOM_SEED'):
    
    if random_seed == 'RANDOM_SEED':
        random_seed = RANDOM_SEED
    
    assert isinstance(dataset, LambdaNetDataset) 
    
    lambda_nets_list = dataset.lambda_net_list
    
    if len(lambda_nets_list) == test_split:
        return None, dataset
    elif isinstance(test_split, int) or isinstance(test_split, float):
        lambda_nets_train_list, lambda_nets_test_list = train_test_split(lambda_nets_list, test_size=test_split, random_state=random_seed)     
    elif isinstance(test_split, list):
        lambda_nets_test_list = [lambda_nets_list[i] for i in test_split]
        lambda_nets_train_list = list(set(lambda_nets_list) - set(lambda_nets_test_list))
        #lambda_nets_train_list = lambda_nets_list.copy()
        #for i in sorted(test_split, reverse=True):
        #    del lambda_nets_train_list[i]           
    assert len(lambda_nets_list) == len(lambda_nets_train_list) + len(lambda_nets_test_list)
    
    return LambdaNetDataset(lambda_nets_train_list), LambdaNetDataset(lambda_nets_test_list)
                                                                                                 
def generate_base_model(): #without dropout
    base_model = Sequential()

    base_model.add(Dense(lambda_network_layers[0], activation='relu', input_dim=n))

    for neurons in lambda_network_layers[1:]:
        base_model.add(Dense(neurons, activation='relu'))

    base_model.add(Dense(1))
    
    return base_model

def shape_flat_weights(flat_weights, target_weights):
    
    from utilities.utility_functions import flatten
    
    #print('shape_flat_weights')
    
    shaped_weights =[]
    start = 0
    for el in target_weights:
        target_shape = el.shape
        size = len(list(flatten(el)))
        shaped_el = np.reshape(flat_weights[start:start+size], target_shape)
        shaped_weights.append(shaped_el)
        start += size

    return shaped_weights

def weights_to_pred(weights, x, base_model=None):

    if base_model is None:
        base_model = generate_base_model()
    else:
        base_model = tf.keras.models.clone_model(base_model)
    base_model_weights = base_model.get_weights()
    
    # Shape weights (flat) into correct model structure
    shaped_weights = shape_flat_weights(weights, base_model_weights)
    
    model = tf.keras.models.clone_model(base_model)
    
    # Make prediction
    model.set_weights(shaped_weights)
    y = model.predict(x).ravel()
    return y

    
def weights_to_model(weights, config=None, base_model=None):
    
    #print('W-FUNCTION START')
    
    if config != None:
        globals().update(config) 
    
    if base_model is None:

        base_model = Sequential()

        #kerase defaults: kernel_initializer='glorot_uniform', bias_initializer='zeros'               
        if fixed_initialization_lambda_training:
            base_model.add(Dense(lambda_network_layers[0], activation='relu', input_dim=n, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=current_seed), bias_initializer='zeros'))
        else:
            base_model.add(Dense(lambda_network_layers[0], activation='relu', input_dim=n))

        if dropout > 0:
            base_model.add(Dropout(dropout))

        for neurons in lambda_network_layers[1:]:
            if fixed_initialization_lambda_training:
                base_model.add(Dense(neurons, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=current_seed), bias_initializer='zeros'))
            else:
                base_model.add(Dense(neurons, activation='relu'))
            if dropout > 0:
                base_model.add(Dropout(dropout))   

        if fixed_initialization_lambda_training:
            base_model.add(Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.train_settings['seed']), bias_initializer='zeros'))
        else:
            base_model.add(Dense(1))        
        
        
    else:
        base_model = tf.keras.models.clone_model(base_model)
    
    base_model_weights = base_model.get_weights()    

        

    # Shape weights (flat) into correct model structure
    shaped_weights = shape_flat_weights(weights, base_model_weights)
    
    model = tf.keras.models.clone_model(base_model)

    model.set_weights(shaped_weights)
    
    model.compile(optimizer=optimizer_lambda,
                  loss=loss_lambda,
                  metrics=[r2_keras_loss, 'mae', tf.keras.metrics.RootMeanSquaredError()])    
    
    
    return model  


#######################################################################################################################################################
#################################################################Lambda Net TRAINING###################################################################
#######################################################################################################################################################

def train_nn(lambda_index,
             X_data_lambda, 
             y_data_real_lambda, 
             polynomial, 
             seed_list, 
             callbacks=None, 
             return_history=False, 
             each_epochs_save=None, 
             printing=False, 
             return_model=False):
    
    from utilities.utility_functions import generate_paths, calculate_function_values_from_polynomial, pairwise
    
    global loss_lambda
    global list_of_monomial_identifiers
        
    if polynomial is not None:
        paths_dict = generate_paths(path_type = 'lambda_net')
    else: 
        paths_dict = generate_paths(path_type = 'interpretation_net')
    
    current_seed = None
    if fixed_seed_lambda_training or fixed_initialization_lambda_training:
        current_seed = seed_list[lambda_index%number_different_lambda_trainings]
    
    if fixed_seed_lambda_training:
        random.seed(current_seed)
        np.random.seed(current_seed)
        if int(tf.__version__[0]) >= 2:
            tf.random.set_seed(current_seed)
        else:
            tf.set_random_seed(current_seed) 
            
            
    if each_epochs_save_lambda != None:
        epochs_save_range = range(1, epochs_lambda//each_epochs_save_lambda+1) if each_epochs_save_lambda == 1 else range(epochs_lambda//each_epochs_save_lambda+1)
    else:
        epochs_save_range = None
    
    if isinstance(X_data_lambda, pd.DataFrame):
        X_data_lambda = X_data_lambda.values
    if isinstance(y_data_real_lambda, pd.DataFrame):
        y_data_real_lambda = y_data_real_lambda.values
                
    X_train_lambda_with_valid, X_test_lambda, y_train_real_lambda_with_valid, y_test_real_lambda = train_test_split(X_data_lambda, y_data_real_lambda, test_size=0.25, random_state=current_seed)           
    X_train_lambda, X_valid_lambda, y_train_real_lambda, y_valid_real_lambda = train_test_split(X_train_lambda_with_valid, y_train_real_lambda_with_valid, test_size=0.25, random_state=current_seed)           
     
        
    model = Sequential()
    
    #kerase defaults: kernel_initializer='glorot_uniform', bias_initializer='zeros'               
    if fixed_initialization_lambda_training:
        model.add(Dense(lambda_network_layers[0], activation='relu', input_dim=X_data_lambda.shape[1], kernel_initializer=tf.keras.initializers.GlorotUniform(seed=current_seed), bias_initializer='zeros'))
    else:
        model.add(Dense(lambda_network_layers[0], activation='relu', input_dim=X_data_lambda.shape[1]))
        
    if dropout > 0:
        model.add(Dropout(dropout))

    for neurons in lambda_network_layers[1:]:
        if fixed_initialization_lambda_training:
            model.add(Dense(neurons, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=current_seed), bias_initializer='zeros'))
        else:
            model.add(Dense(neurons, activation='relu'))
        if dropout > 0:
            model.add(Dropout(dropout))   
    
    if fixed_initialization_lambda_training:
        model.add(Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=current_seed), bias_initializer='zeros'))
    else:
        model.add(Dense(1))
    
    try:
        loss_lambda = tf.keras.losses.get(loss_lambda)
    except ValueError as error_message:
        if loss_lambda == 'r2':
            loss_lambda = r2_keras_loss
        else:
            print(error_message)
        
    model.compile(optimizer=optimizer_lambda,
                  loss=loss_lambda,
                  metrics=[r2_keras_loss, 'mae', tf.keras.metrics.RootMeanSquaredError()]
                 )
        
    if early_stopping_lambda:
        if callbacks == None:
            callbacks = []
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=early_stopping_min_delta_lambda, verbose=0, mode='min', restore_best_weights=True)
        callbacks.append(early_stopping)
    
    weights = []
    polynomial_lstsq_pred_list = []
    polynomial_lstsq_true_list = []

    lstsq_data = np.random.uniform(low=x_min, high=x_max, size=(random_evaluation_dataset_size, n)) #y_train_pred_lambda.ravel()
    terms_matrix = generate_term_matric_for_lstsq(lstsq_data, list_of_monomial_identifiers)

    
    terms_matrix_train = generate_term_matric_for_lstsq(X_train_lambda, list_of_monomial_identifiers)
        

    #y_train_real_lambda = y_train_real_lambda.astype(float)  
    #y_valid_real_lambda = y_valid_real_lambda.astype(float)
    #y_test_real_lambda = y_test_real_lambda.astype(float)

        
    if each_epochs_save == None or each_epochs_save==epochs_lambda:
        
        model_history = model.fit(X_train_lambda,
                      y_train_real_lambda, 
                      epochs=epochs_lambda, 
                      batch_size=batch_lambda, 
                      callbacks=callbacks,
                      validation_data=(X_valid_lambda, y_valid_real_lambda),
                      verbose=0,
                      workers=0)
        
        weights.append(model.get_weights())
        
        history = model_history.history
        
        y_train_pred_lambda = model.predict(X_train_lambda) 
        y_valid_pred_lambda = model.predict(X_valid_lambda)                
        y_test_pred_lambda = model.predict(X_test_lambda)
    
        y_random_pred_lambda = model.predict(lstsq_data)
        
        
        polynomial_lstsq_pred, _, _, _ = np.linalg.lstsq(terms_matrix, y_random_pred_lambda.ravel(), rcond=-1)#[::-1]
        polynomial_lstsq_true, _, _, _ = np.linalg.lstsq(terms_matrix_train, y_train_real_lambda.ravel(), rcond=-1)#[::-1] 
        polynomial_lstsq_pred_list.append(polynomial_lstsq_pred)
        polynomial_lstsq_true_list.append(polynomial_lstsq_true)
        
        
        y_train_pred_lambda_poly_lstsq = calculate_function_values_from_polynomial(polynomial_lstsq_pred, X_train_lambda, force_complete_poly_representation=True, list_of_monomial_identifiers=list_of_monomial_identifiers)
        y_train_real_lambda_poly_lstsq = calculate_function_values_from_polynomial(polynomial_lstsq_true, X_train_lambda, force_complete_poly_representation=True, list_of_monomial_identifiers=list_of_monomial_identifiers)
        y_valid_pred_lambda_poly_lstsq = calculate_function_values_from_polynomial(polynomial_lstsq_pred, X_valid_lambda, force_complete_poly_representation=True, list_of_monomial_identifiers=list_of_monomial_identifiers)
        y_valid_real_lambda_poly_lstsq = calculate_function_values_from_polynomial(polynomial_lstsq_true, X_valid_lambda, force_complete_poly_representation=True, list_of_monomial_identifiers=list_of_monomial_identifiers)   
        y_test_pred_lambda_poly_lstsq = calculate_function_values_from_polynomial(polynomial_lstsq_pred, X_test_lambda, force_complete_poly_representation=True, list_of_monomial_identifiers=list_of_monomial_identifiers)
        y_test_real_lambda_poly_lstsq = calculate_function_values_from_polynomial(polynomial_lstsq_true, X_test_lambda, force_complete_poly_representation=True, list_of_monomial_identifiers=list_of_monomial_identifiers)  
        
        
        pred_list = {'lambda_index': lambda_index,
                          'y_train_real_lambda': y_train_real_lambda, 
                          'y_train_pred_lambda': y_train_pred_lambda, 
                          'y_train_pred_lambda_poly_lstsq': y_train_pred_lambda_poly_lstsq,
                          #y_train_real_lambda_poly_lstsq,
                          'X_train_lambda': X_train_lambda, 
                          'y_valid_real_lambda': y_valid_real_lambda,
                          'y_valid_pred_lambda': y_valid_pred_lambda, 
                          'y_valid_pred_lambda_poly_lstsq': y_valid_pred_lambda_poly_lstsq,
                          #y_valid_real_lambda_poly_lstsq,
                          'X_valid_lambda': X_valid_lambda, 
                          'y_test_real_lambda': y_test_real_lambda, 
                          'y_test_pred_lambda': y_test_pred_lambda, 
                          'y_test_pred_lambda_poly_lstsq': y_test_pred_lambda_poly_lstsq, 
                          #y_test_real_lambda_poly_lstsq,
                          'X_test_lambda': X_test_lambda}      
        
        scores_train, std_train, mean_train = evaluate_lambda_net('TRAIN', X_train_lambda, y_train_real_lambda, y_train_pred_lambda, y_train_pred_lambda_poly_lstsq, y_train_real_lambda_poly_lstsq)
        scores_valid, std_valid, mean_valid = evaluate_lambda_net('VALID', X_valid_lambda, y_valid_real_lambda, y_valid_pred_lambda, y_valid_pred_lambda_poly_lstsq, y_valid_real_lambda_poly_lstsq)
        scores_test, std_test, mean_test = evaluate_lambda_net('TEST', X_test_lambda, y_test_real_lambda, y_test_pred_lambda, y_test_pred_lambda_poly_lstsq, y_test_real_lambda_poly_lstsq)

        scores_std = {}
        for aDict in (std_train, std_valid, std_test):
            scores_std.update(aDict)      
        scores_mean = {}
        for aDict in (mean_train, mean_valid, mean_test):
            scores_mean.update(aDict)
        
        scores_list = [lambda_index,
                     scores_train,
                     scores_valid,
                     scores_test,
                     scores_std,
                     scores_mean]            
                            
    else:
        scores_list = []
        pred_list = []
        for i in epochs_save_range:
            train_epochs_step = each_epochs_save if i > 1 else max(each_epochs_save-1, 1) if i==1 else 1
            
            model_history = model.fit(X_train_lambda, 
                      y_train_real_lambda, 
                      epochs=train_epochs_step, 
                      batch_size=batch_lambda, 
                      callbacks=callbacks,
                      validation_data=(X_valid_lambda, y_valid_real_lambda),
                      verbose=0,
                      workers=1,
                      use_multiprocessing=False)
            
            #history adjustment for continuing training
            if i == 0 and each_epochs_save != 1 or i == 1 and each_epochs_save == 1:
                history = model_history.history
            else:
                history = mergeDict(history, model_history.history)

            weights.append(model.get_weights())
            
            y_train_pred_lambda = model.predict(X_train_lambda)                
            y_valid_pred_lambda = model.predict(X_valid_lambda)                
            y_test_pred_lambda = model.predict(X_test_lambda)        
            
            y_random_pred_lambda = model.predict(lstsq_data)
    
            polynomial_lstsq_pred, _, _, _ = np.linalg.lstsq(terms_matrix, y_random_pred_lambda.ravel(), rcond=-1)#[::-1] 
            #does not change over time
            if i == 0 and each_epochs_save != 1 or i == 1 and each_epochs_save == 1:
                polynomial_lstsq_true, _, _, _ = np.linalg.lstsq(terms_matrix_train, y_train_real_lambda.ravel(), rcond=-1)#[::-1] 
            polynomial_lstsq_pred_list.append(polynomial_lstsq_pred)
            polynomial_lstsq_true_list.append(polynomial_lstsq_true)       
            

            
            y_train_pred_lambda_poly_lstsq = calculate_function_values_from_polynomial(polynomial_lstsq_pred, X_train_lambda, force_complete_poly_representation=True, list_of_monomial_identifiers=list_of_monomial_identifiers)
            y_valid_pred_lambda_poly_lstsq = calculate_function_values_from_polynomial(polynomial_lstsq_pred, X_valid_lambda, force_complete_poly_representation=True, list_of_monomial_identifiers=list_of_monomial_identifiers)
            y_test_pred_lambda_poly_lstsq = calculate_function_values_from_polynomial(polynomial_lstsq_pred, X_test_lambda, force_complete_poly_representation=True, list_of_monomial_identifiers=list_of_monomial_identifiers)           
            if i == 0 and each_epochs_save != 1 or i == 1 and each_epochs_save == 1:
                y_train_real_lambda_poly_lstsq = calculate_function_values_from_polynomial(polynomial_lstsq_true, X_train_lambda, force_complete_poly_representation=True, list_of_monomial_identifiers=list_of_monomial_identifiers)
                y_valid_real_lambda_poly_lstsq = calculate_function_values_from_polynomial(polynomial_lstsq_true, X_valid_lambda, force_complete_poly_representation=True, list_of_monomial_identifiers=list_of_monomial_identifiers)  
                y_test_real_lambda_poly_lstsq = calculate_function_values_from_polynomial(polynomial_lstsq_true, X_test_lambda, force_complete_poly_representation=True, list_of_monomial_identifiers=list_of_monomial_identifiers)                    
                
            pred_list.append({'lambda_index': lambda_index,
                              'y_train_real_lambda': y_train_real_lambda, 
                              'y_train_pred_lambda': y_train_pred_lambda, 
                              'y_train_pred_lambda_poly_lstsq': y_train_pred_lambda_poly_lstsq,
                              #y_train_real_lambda_poly_lstsq,
                              'X_train_lambda': X_train_lambda, 
                              'y_valid_real_lambda': y_valid_real_lambda,
                              'y_valid_pred_lambda': y_valid_pred_lambda, 
                              'y_valid_pred_lambda_poly_lstsq': y_valid_pred_lambda_poly_lstsq,
                              #y_valid_real_lambda_poly_lstsq,
                              'X_valid_lambda': X_valid_lambda, 
                              'y_test_real_lambda': y_test_real_lambda, 
                              'y_test_pred_lambda': y_test_pred_lambda, 
                              'y_test_pred_lambda_poly_lstsq': y_test_pred_lambda_poly_lstsq, 
                              #y_test_real_lambda_poly_lstsq,
                              'X_test_lambda': X_test_lambda})
    
            scores_train, std_train, mean_train = evaluate_lambda_net('TRAIN', X_train_lambda, y_train_real_lambda, y_train_pred_lambda, y_train_pred_lambda_poly_lstsq, y_train_real_lambda_poly_lstsq)
            scores_valid, std_valid, mean_valid = evaluate_lambda_net('VALID', X_valid_lambda, y_valid_real_lambda, y_valid_pred_lambda, y_valid_pred_lambda_poly_lstsq, y_valid_real_lambda_poly_lstsq)
            scores_test, std_test, mean_test = evaluate_lambda_net('TEST', X_test_lambda, y_test_real_lambda, y_test_pred_lambda, y_test_pred_lambda_poly_lstsq, y_test_real_lambda_poly_lstsq)

            scores_std = {}
            for aDict in (std_train, std_valid, std_test):
                scores_std.update(aDict)
            scores_mean = {}
            for aDict in (mean_train, mean_valid, mean_test):
                scores_mean.update(aDict)

            scores_list_single_epoch =  [lambda_index,
                                         scores_train,
                                          scores_valid,
                                          scores_test,
                                          scores_std,
                                          scores_mean]        
                  
            scores_list.append(scores_list_single_epoch)
       

        
    if printing and polynomial is not None:        
        for i, (weights_for_epoch, polynomial_lstsq_pred_for_epoch, polynomial_lstsq_true_for_epoch) in enumerate(zip(weights, polynomial_lstsq_pred_list, polynomial_lstsq_true_list)):
            
            directory = './data/weights/weights_' + paths_dict['path_identifier_lambda_net_data'] + '/'
            
            if each_epochs_save == None or each_epochs_save==epochs_lambda:
                path_weights = directory + 'weights_epoch_' + str(epochs_lambda).zfill(3) + '.txt'
            else:
                index = (i+1)*each_epochs_save if each_epochs_save==1 else i*each_epochs_save if i > 1 else each_epochs_save if i==1 else 1
                path_weights = directory + 'weights_epoch_' + str(index).zfill(3) + '.txt'                       
            
                
            with open(path_weights, 'a') as text_file: 
                text_file.write(str(lambda_index))
                text_file.write(', ' + str(current_seed))
                for i, value in enumerate(polynomial.values): 
                    text_file.write(', ' + str(value))   
                for value in polynomial_lstsq_pred_for_epoch:
                    text_file.write(', ' + str(value))
                for value in polynomial_lstsq_true_for_epoch:
                    text_file.write(', ' + str(value))
                for layer_weights, biases in pairwise(weights_for_epoch):    #clf.get_weights()
                    for neuron in layer_weights:
                        for weight in neuron:
                            text_file.write(', ' + str(weight))
                    for bias in biases:
                        text_file.write(', ' + str(bias))
                text_file.write('\n')

                text_file.close() 
            
            
        path_X_data = directory + 'lambda_X_test_data.txt'
        path_y_data = directory + 'lambda_y_test_data.txt'             

        with open(path_X_data, 'a') as text_file: 
            text_file.write(str(lambda_index))

            for row in X_test_lambda:
                for value in row:
                    text_file.write(', ' + str(value))
            text_file.write('\n')

            text_file.close()                

        with open(path_y_data, 'a') as text_file: 
            text_file.write(str(lambda_index))          
            for value in y_test_real_lambda.flatten():
                text_file.write(', ' + str(value))
            text_file.write('\n')

            text_file.close()                    


            
    if return_model:
        return (lambda_index, current_seed, polynomial, polynomial_lstsq_pred_list, polynomial_lstsq_true_list), scores_list, pred_list, history, model
    elif return_history:
        return (lambda_index, current_seed, polynomial, polynomial_lstsq_pred_list, polynomial_lstsq_true_list), scores_list, pred_list, history, #polynomial_lstsq_pred_list, polynomial_lstsq_true_list#, weights, history
    else:
        return (lambda_index, current_seed, polynomial, polynomial_lstsq_pred_list, polynomial_lstsq_true_list), scores_list, pred_list#, weights
    