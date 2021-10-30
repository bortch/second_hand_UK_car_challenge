# ----------------------------------------------------------------------------
# Created By  : Bortch - JBS
# Created Date: 09/01/2021
# version ='1.0'
# source = https://github.com/bortch/second_hand_UK_car_challenge
# ---------------------------------------------------------------------------
"""This script contains functions used to prepare the data for the datasets. 
It includes, among others: 
  - saving and loading .csv files
  - cleaning of variables (treatment of duplicates, scaling, removal of unused values) 
  - transformation and imputation of outliers
"""
from os.path import join, isfile
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from scipy.stats import zscore

import constants as cnst


def get_numerical_columns(data):
    return list(data.select_dtypes(include=[np.number]).columns.values)


def get_categorical_columns(data):
    return list(data.select_dtypes(include=['object', 'category']).columns.values)


def set_as_categorical(data, columns=None, verbose=False):
    if verbose:
        print("Change columns dtype into categorical")
    if not isinstance(columns, list):
        get_columns = get_categorical_columns
    else:
        def get_columns(): return columns

    def transform_dtype(series):
        return pd.Categorical(series, categories=series.unique().tolist())

    return set_dtype_as(src_data=data,
                        get_columns_name_callback=get_columns,
                        dtype_transformer_callback=transform_dtype,
                        verbose=verbose)


def set_as_numerical(data, columns=None, verbose=False):
    if verbose:
        print("Change columns dtype into numerical")
    if not isinstance(columns, list):
        get_columns = get_numerical_columns
    else:
        def get_columns(_): return columns

    return set_dtype_as(src_data=data,
                        get_columns_name_callback=get_columns,
                        dtype_transformer_callback=pd.to_numeric,
                        verbose=verbose)

def get_ordered_categories(data, by):
    df = data.copy()
    categories = {}
    columns = get_categorical_columns(df)
    for cat in columns:
        ordered_df = df[[cat, by]]
        ordered_df = ordered_df.groupby(cat).agg('mean').reset_index()
        ordered_df.sort_values(
            by, ascending=True, inplace=True, ignore_index=True)
        categories[cat] = []
        for c in ordered_df[cat].values:
            categories[cat].append(c)
    return categories

def train_val_test_split(X, y, test_size, train_size, val_size, random_state=None, verbose=False):
    if isinstance(X, pd.DataFrame):
        X = X.reset_index(drop=True)
    if isinstance(y, pd.Series):
        y = y.reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=test_size/(test_size + val_size), random_state=random_state, shuffle=False)
    if verbose:
        print("\nSplitting into train, val and test sets")
        print(f"\tX_train: {X_train.shape}\n\tX_val: {X_val.shape}\n\tX_test: {X_test.shape}\n\ty_train: {y_train.shape}\n\ty_val: {y_val.shape}\n\ty_test: {y_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def set_dtype_as(src_data, get_columns_name_callback, dtype_transformer_callback, verbose=False):
    df = src_data.copy()
    columns = get_columns_name_callback(df)
    for c in columns:
        df[c] = dtype_transformer_callback(df[c])
        if verbose:
            print(f"\t{c}: {df[c].dtype}")
    return df


def clean_variables(data, target='price', verbose=False):
    if verbose:
        print("\nRemoving duplicate entries and noisy features:")
    df = data.copy()
    df = set_as_categorical(df,verbose=verbose)

    # remove duplicate
    if verbose:
        print("Drop duplicate")
    df.drop_duplicates(inplace=True)

    df = set_as_numerical(df,verbose=verbose)

    # scale
    if verbose:
        print("Scaling numerical feature")
    num_columns = get_numerical_columns(df)
    num_columns.remove(target)
    # Standardisation
    std_scaler = StandardScaler(with_mean=False)
    df[num_columns] = std_scaler.fit_transform(df[num_columns])
    # Normalisation
    scaler = MinMaxScaler()
    df[num_columns] = scaler.fit_transform(df[num_columns])
    # remove unhandled categories
    if verbose:
        print("Remove unhandled categories")
    df = df[df['transmission'] != 'Other']
    df = df[(df['fuel_type'] != 'Other')]
    df = df[(df['fuel_type'] != 'Electric')]
    # log target
    if verbose:
        print("Replace target by Log(target)")
    #df[target] = np.log(df[target])
    return df


def drop_outliers(data):
    return outliers_transformer(data, drop=True)


def nan_outliers(data):
    return outliers_transformer(data)


def outliers_transformer(data, drop=False, verbose=False):
    if verbose:
        print('\nTransform outliers')
    df = data.copy()
    columns = get_numerical_columns(df)
    columns.remove('price')
    thresh = 3
    if drop:
        outliers = df[columns].apply(lambda x: np.abs(
            zscore(x, nan_policy='omit')) > thresh).any(axis=1)
        if verbose:
            print(f"\tDroping outliers")
        df.drop(df.index[outliers], inplace=True)
    else:
        outliers = df[columns].apply(lambda x: np.abs(
            zscore(x, nan_policy='omit')) > thresh)
        # replace value from outliers by nan
        if verbose:
            print(f"\ttagging outliers")
        for c in outliers.columns.to_list():
            df.loc[outliers[c], c] = np.nan
    return df


def numerical_imputer(data, n_neighbors=10, weights='distance', fit_set=None, imputer_type=None, verbose=False):
    if verbose:
        print('\nImputing missing numerical value')
    df = data.copy()
    # print("df",df.info())
    columns = get_numerical_columns(df)
    if 'price' in columns:
        columns.remove('price')
    has_nan = df.isnull().values.any()
    if verbose:
        print(f"\t{columns} has NAN? {has_nan}")
    if(has_nan):
        if verbose:
            print("\tNAN found, imputing ...")
        if imputer_type == 'KNN':
            if verbose:
                print('\tusing KNNImputer')
            imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        else:
            if verbose:
                print('\tusing IterativeImputer')
            imputer = IterativeImputer(random_state=0)
        if isinstance(fit_set, pd.DataFrame):
            if verbose:
                print('\tfit imputer using fit_set:', fit_set.shape)
            imputer.fit(fit_set[columns])
            imputed = imputer.transform(df[columns])
        else:
            imputed = imputer.fit_transform(df[columns])
        for i, c in enumerate(columns):
            df[c] = imputed[:, i]
    if verbose:
        print("\tImputation done?", not df.isnull().values.any())
    return df

def save_prepared_dataset(df, filename):
    dest_file_path = join(cnst.PREPARED_DATASET_PATH, filename)
    df.to_csv(dest_file_path)
    print(f"{filename} data saved @ {dest_file_path}")


def load_prepared_dataset(filename):
    file_path = join(cnst.PREPARED_DATASET_PATH, filename)
    if isfile(file_path):
        df = pd.read_csv(file_path, index_col=0)
        return df
    else:
        return None