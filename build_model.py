# ----------------------------------------------------------------------------
# Created By  : Bortch - JBS
# Created Date: 09/01/2021
# version ='16.0'
# source = https://github.com/bortch/second_hand_UK_car_challenge
# modification for kaggle
# ---------------------------------------------------------------------------
import numpy as np
from os.path import join, isfile

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

from joblib import dump, load
import json

from itertools import product

import warnings
import time
import constants as cnst

warnings.filterwarnings("ignore")



def get_pipeline_params_search_domain():
    return {
        'transformer__poly': {'transformer__poly__degree': [1, 2, 3],
                            'transformer__poly__interaction_only': [True, False],
                            'transformer__poly__include_bias': [True, False], },
        'transformer__mpg_pipe': {'transformer__mpg_pipe__discretize__n_bins': [6, 10],
                                'transformer__mpg_pipe__discretize__encode': ['onehot', 'ordinal'],
                                'transformer__mpg_pipe__discretize__strategy': ['uniform', 'quantile', 'kmeans'], },
        'transformer__tax_pipe': {'transformer__tax_pipe__discretize__n_bins': [8, 9, 10],
                                'transformer__tax_pipe__discretize__encode': ['onehot', 'ordinal'],
                                'transformer__tax_pipe__discretize__strategy': ['uniform', 'quantile', 'kmeans'], },
        'transformer__engine_size_pipe': {'transformer__engine_size_pipe__discretize__n_bins': [2, 3, 4],
                                        'transformer__engine_size_pipe__discretize__encode': ['onehot', 'ordinal'],
                                        'transformer__engine_size_pipe__discretize__strategy': ['uniform', 'quantile', 'kmeans'], },
        'transformer__year_pipe': {'transformer__year_pipe__discretize__n_bins': [3, 10, 11],
                                'transformer__year_pipe__discretize__encode': ['onehot', 'ordinal'],
                                'transformer__year_pipe__discretize__strategy': ['uniform', 'quantile', 'kmeans']}
    }

def get_estimator_params_search_domain():
    return {
        'random_forest': {'random_forest__max_depth': [40, 50, 100],
                        'random_forest__min_samples_split': np.arange(2, 10, 2),
                        'random_forest__max_features': ['auto', 'sqrt', 'log2', None],
                        'random_forest__min_samples_leaf':np.arange(1,10,3),
                        }
    }

def get_transformer(verbose=False):
    if verbose:
        print("\nCreating Columns transformers")

    transformers_ = [
        ("poly", PolynomialFeatures(degree=2,
                                    interaction_only=False,
                                    include_bias=False),
         make_column_selector(dtype_include=np.number)),

        ("mpg_pipe", Pipeline(steps=[
            ('discretize', KBinsDiscretizer(n_bins=6,
                                            encode='onehot', strategy='uniform'))
        ], verbose=verbose), ['mpg']),

        ("tax_pipe", Pipeline(steps=[
            ('discretize',  KBinsDiscretizer(n_bins=9,
                                             encode='onehot', strategy='quantile'))
        ], verbose=verbose), ['tax']),

        ("engine_size_pipe", Pipeline(steps=[
            ('discretize',  KBinsDiscretizer(n_bins=3,
                                             encode='onehot', strategy='uniform'))
        ], verbose=verbose), ['engine_size']),

        ('year_pipe', Pipeline(steps=[
            ('discretize', KBinsDiscretizer(
                n_bins=10, encode='ordinal', strategy='quantile'))
        ], verbose=verbose), ['year']),

        ('model_pipe', Pipeline(steps=[
            ('OHE', OneHotEncoder(handle_unknown='ignore', sparse=False)),
            ('OE', OrdinalEncoder())
        ], verbose=verbose), ['model']),

        ('brand_pipe', Pipeline(steps=[
            ('OHE', OneHotEncoder(handle_unknown='ignore', sparse=False)),
            ('OE', OrdinalEncoder())
        ], verbose=verbose), ['brand']),

        ('transmission_pipe', Pipeline(steps=[
            ('OHE', OneHotEncoder(handle_unknown='ignore', sparse=False)),
            ('OE', OrdinalEncoder())
        ], verbose=verbose), ['transmission']),
         
        ('fuel_type_pipe', Pipeline(steps=[
            ('OHE', OneHotEncoder(handle_unknown='ignore', sparse=False)),
            ('OE', OrdinalEncoder()),
        ], verbose=verbose), ['fuel_type']),
    ]

    transformer = ColumnTransformer(
        transformers_, remainder='passthrough', verbose=verbose)
    return transformer

def extract_features(data, features=['all']):
    X = data.copy()
    if 'all' in features:
        features = ['model_count', 'age',
                    'mpy_mpy', 'tax_per_year',
                    'mileage_per_year', 'mpg_per_year',
                    'engine_per_year']
    # adding feature
    model_count = X.groupby('model')['model'].transform('count')
    occ = model_count/X.shape[0]
    if 'model_count' in features:
        X['model_count'] = model_count

    age = X['year'].max()-X['year']
    age[age < 1] = 1
    if 'age' in features:
        X['age'] = age

    m_a = X['mileage']/age
    if 'mileage_per_year' in features:
        X['mileage_per_year'] = m_a

    mpg_a = X['mpg']/age
    if 'mpg_per_year' in features:
        X['mpg_per_year'] = mpg_a

    t_a = X['tax']/age
    if 'tax_per_year' in features:
        X['tax_per_year'] = t_a

    e_a = X['engine_size']/age
    if 'engine_per_year' in features:
        X['engine_per_year'] = e_a

    mmte = (X['mileage']+X['mpg']+X['tax']+X['engine_size'])/occ
    if 'mpy_mpy' in features:
        X['mpy_mpy'] = (m_a/mmte+mpg_a/mmte+t_a/mmte+e_a/mmte)

    #X.drop('age',axis=1, inplace=True)
    #X['galon_per_year'] = X['mpg']/X['mileage_per_year']
    #X['galon_per_year'] = X['mileage_per_year']/X['mpg']
    #X.drop('mileage_per_year',axis=1, inplace=True)
    #X['tax_per_mileage'] = X['tax']/X['mileage']
    #X['tax_per_mileage'] = X['mileage']/X['tax']
    #X['litre_per_mileage'] = X['engine_size']/X['mileage']
    #X['litre_per_mileage'] = X['mileage']/X['engine_size']
    #X['litre_per_galon'] = X['engine_size']/X['galon_per_year']
    return X

def get_model_pipeline(model_path_to_load=None, verbose=False, warm_start=False, transformers=None):

    if (model_path_to_load is not None) and isfile(model_path_to_load):
        model = load(model_path_to_load)
        return model
    else:
        if transformers:
            transformers_ = transformers
        else:
            transformers_ = get_transformer(verbose=verbose)

        nb_estimators = 10
        steps = [
            ("features_extraction", FunctionTransformer(
                extract_features, validate=False)),
            ("transformer", transformers_),
            ("random_forest", RandomForestRegressor(
                n_estimators=nb_estimators,
                max_features=None,
                min_samples_split=6,
                max_depth=50,
                n_jobs=-1,
                warm_start=warm_start,  # Optimise computation during GridSearchCV
                verbose=verbose
            ))
        ]
        pipeline = Pipeline(steps=steps, verbose=verbose)
        return pipeline

def get_default_pipeline_params(ordered_categories):
    params = {
        "features_extraction__kw_args": {'features': ["all"]},
        # -----------
        # numerical
        # ___________
        #"transformer__poly": 'passthrough',
        #"transformer__mpg_pipe": 'passthrough',
        #"transformer__tax_pipe": 'passthrough',
        #"transformer__engine_size_pipe": 'passthrough',
        #"transformer__year_pipe": 'passthrough',
        # -------------
        # categorical
        # -------------
        # *** model
        "transformer__model_pipe__OHE": 'passthrough',
        "transformer__model_pipe__OE__categories": [ordered_categories['model']],
        # *** brand
        "transformer__brand_pipe__OHE": 'passthrough',
        "transformer__brand_pipe__OE__categories": [ordered_categories['brand']],
        # *** transmission
        "transformer__transmission_pipe__OHE": 'passthrough',
        "transformer__transmission_pipe__OE__categories": [ordered_categories['transmission']],
        # *** fuel_type
        "transformer__fuel_type_pipe__OHE": 'passthrough',
        "transformer__fuel_type_pipe__OE__categories": [ordered_categories['fuel_type']]
    }
    return params

def dump_model(model, as_filename, verbose=False):
    model_filename = f'{as_filename}.joblib'
    file_path = join(cnst.MODEL_DIR_PATH, model_filename)
    dump(model, file_path)
    if verbose:
        print(f"Model {as_filename} saved @ {file_path}")
    return file_path

def dump_params(params, as_filename, verbose=False):
    encoded_params = encode_params(params)
    file_path = join(cnst.MODEL_DIR_PATH, f"{as_filename}.json")
    with open(file_path, 'w') as file:
        json.dump(encoded_params, file)
    if verbose:
        print(f"Model's params {as_filename} saved @ {file_path}")
    return file_path

def encode_params(params, verbose=False):
    encoded_params = {}
    for key, p in params.items():
        if isinstance(p,dict):
            for k, value in p.items():
                if verbose:
                    print(f"{k}:{value}")
            if isinstance(value, np.int64):
                encoded_params[key] = int(value)
            elif isinstance(value, np.float64):
                encoded_params[key] = float(value)
            else:
                encoded_params[key] = value
        else:
            encoded_params[key]=p
    return encoded_params

def get_base_model(model_filename, X,y,ordered_categories, verbose=False):
    # load model if already existing   
    model_file_path = join(cnst.MODEL_DIR_PATH, model_filename)
    if isfile(model_file_path):  
        if verbose:
            print(f"Loading {model_filename}")
        model = load(model_file_path)
    else:
        # create the model
        if verbose:
            print(f'Creating {model_filename}')
            
        model = get_model_pipeline(verbose=verbose)
        params = get_default_pipeline_params(ordered_categories)

        # Create an fit a base model
        model.set_params(**params)
        model.fit(X, y)

        # save the model
        #dump_model(model, model_filename.split('.')[0], verbose=verbose)
    return model

def evaluate_model(model, X, y, verbose=False):
    if verbose:
        print(f"\nModel Evaluation")
    y_prediction = model.predict(X)
    y_exp = np.exp(y)
    y_prediction_exp = np.exp(y_prediction)
    rmse = np.sqrt(mean_squared_error(y_exp, y_prediction_exp))
    if verbose:
        print(f"RMSE: {rmse}")
    return y_prediction_exp, y_exp, rmse

def evaluate_params(model, params, X, y, current_score=np.Inf, verbose=False):
    if verbose:
        print("\nEvaluate Params")
    best_params_dict = {}
    model_ = model
    best_score = current_score
    mse = make_scorer(mean_squared_error, greater_is_better=False)
    for key, param in params.items():
        if verbose:
            print(f"\nSearch best param for '{key}' with {params}")
        best_params_dict[key] = 'passthrough'
        if param != 'passthrough':
            grid = GridSearchCV(model_, param_grid=param,
                                cv=10, scoring=mse,
                                verbose=verbose,
                                n_jobs=-1
                                )
            grid.fit(X, y)
            # evaluate model
            if verbose:
                print(f"Current score {grid.best_score_}, last best score {best_score}")
            if grid.best_score_>= best_score:
                # there's a better score
                model_ = grid.best_estimator_
                best_params_dict[key] = grid.best_params_
                best_score = grid.best_score_
        if verbose:
            print('Best Score',best_score,'\n',best_params_dict[key])
    return model_, best_params_dict, best_score

def evaluate_combination_of_params(model, params, X_train, y_train, current_score=np.Inf, verbose=False):
    if verbose:
        print("\nEvaluate params combination")
    
    # init best params, best score & best model
    best_params = {}
    best_model = model
    best_score = current_score # lower is better
    
    to_skip_previous_param_key = []
    # init scroring function
    mse = make_scorer(mean_squared_error, greater_is_better=False)
    # evaluate params one at times:
    for key, param in params.items():
        # skip the evaluation:
        # if params belong to current solution
        if key in best_params.keys():
            if verbose:
                print(f"{key} already in solution")
            continue
        # if it still the same param as in the previous permutation
        if key in to_skip_previous_param_key:
            if verbose:
                print(f"{key} evaluation skipped as already computed")
            to_skip_previous_param_key.clear()
            break
        if verbose:
            print(f"\nSearch best param for '{key}'")
        
        # build the current grid_param to be optimized
        param_grid={}
        if param == 'passthrough':
            param_grid[key]= [param]
        else:
            for p in param:
                param_grid[p]=param[p]
        # override key that belongs to the previous solution
        for key, param in best_params.items():
            if isinstance(param, list):
                param_grid[key]=param
            else:
                param_grid[key]= [param]
        if verbose:
            print('Current Param Grid to evaluate:',param_grid)
        grid = GridSearchCV(model, param_grid=param_grid,
                    cv=5, scoring=mse,
                    verbose=False,
                    n_jobs=-1,
                    # pre_dispatch=1
                    )                   
        grid.fit(X_train, y_train)
        # if it is a better score
        # keep track of the optimized parameters
        if grid.best_score_>=current_score:
            if verbose:
                print(f'actual best score {best_score}, better solution found:{grid.best_score_}')
            best_model = grid.best_estimator_
            best_params = {**best_params,**grid.best_params_}
            best_score = grid.best_score_
            #to_skip_previous_param_key.pop(key)
        else:
            # else itisn't a good start
            if verbose:
                print(f"Add {key} to be skipped with params: \n{param}\n")
            to_skip_previous_param_key.append(key)
            break
    #print('Best Score',best_score,'\n',best_params[key])
    return best_model, best_params, best_score

def get_combinations_of_params(params_dict):
    nb_params = len(params_dict)
    combination = {}
    mask = product(range(2), repeat=nb_params)
    for m in mask:
        if 1 in m:
            for index, key in enumerate(params_dict):
                if m[index] < 1:
                    combination[key] = 'passthrough'
                else:
                    combination[key] = params_dict[key]
            yield combination

def get_best_pipeline_params(model, X_val, y_val, score, verbose):
    best_score = score
    best_params = {}
    best_model = model
    params_dict = get_pipeline_params_search_domain()
    for param in get_combinations_of_params(params_dict):
        model_, params_, _ = evaluate_combination_of_params(model, param, X_val, y_val,current_score=best_score, verbose=verbose)
        _, _, param_score = evaluate_model(model_, X_val, y_val, verbose=verbose)
        if verbose:
            print(f"\nScore for the current param's combination: {param_score}, last score {best_score}")
        if param_score < best_score:
            best_model = model_
            best_params = params_
            best_score = param_score
            if verbose:
                print("Solution found:")
                print("-\tScore", best_score)
                print("-\tParams", best_params)
    print(f"\nBest score found:", best_score)
    print(f"-\tBest params:", best_params)
    model_name = f'model_pipeline_params_{best_score:.3f}'
    dump_model(model=best_model, as_filename=model_name,verbose=True)
    dump_params(params=best_params, as_filename=model_name, verbose=True)
    return best_model,best_params,best_score

def get_best_estimator_params(model, X_val, y_val, score, verbose):
    best_score = score
    best_model, best_params, _ = evaluate_params(model=model, 
                                                params=get_estimator_params_search_domain(), 
                                                X = X_val, 
                                                y = y_val, 
                                                verbose=verbose)
    # compute estimator score found
    _, _, estimator_score = evaluate_model(best_model, X_val, y_val, verbose=verbose)
    # if the score is better we dump the model and save the params as a json 
    # else we keep the existing one
    if(estimator_score >= best_score):
        if verbose:
            print(f"\nBest estimator's params")
            print(f"-\t Reference score: {best_score}")
            print(f"-\t Best score: {estimator_score}")
            print(f"-\t Best params: {best_params}")
        # dump the model and its estimator's best parameters
        model_name = f'model_estimator_{estimator_score:.3f}'
        dump_model(best_model, model_name,verbose=True)
        dump_params(best_params,model_name, verbose=True)
    else:
        if verbose:
            print(f"\nNo better solution found")
            print(f"-\t Reference score: {best_score}")
            print(f"-\t Optimisation Estimator score (RMSLE): {estimator_score}")
        model_name = f'model_estimator_{best_score:.3f}'
        dump_model(best_model, model_name,verbose=True)
    return best_model,best_params,best_score

def get_best_model(model_filename, X_train, y_train, X_val, y_val, ordered_categories, verbose=False):
    timer_start = time.perf_counter()
    model_name = model_filename.split('.')[0]
    # Get a fitted model or create it
    model = get_base_model(model_filename, X_train, y_train, ordered_categories, verbose=verbose)
    print(f'Model creation duration:{time.perf_counter()-timer_start } sec')
    timer_start = time.perf_counter()
    # Evaluate score of the model using the validation test 
    _, _, ref_score = evaluate_model(model, X_val, y_val, verbose=verbose)
    print(f'Model evaluation duration:{time.perf_counter()-timer_start } sec')
    timer_start = time.perf_counter()
    # Search for best pipeline params
    best_model,_,pipeline_params_score = get_best_pipeline_params(model, X_val, y_val, ref_score, verbose)    
    print(f'Best pipeline params search duration:{time.perf_counter()-timer_start } sec')
    timer_start = time.perf_counter()
    # Search for best estimator params
    best_model,best_params,best_score = get_best_estimator_params(best_model, X_val, y_val, pipeline_params_score, verbose)
    print(f"Best estimator's params search duration:{time.perf_counter()-timer_start } sec")
    return best_model,best_params,best_score
