import pandas as pd

PATH_TRAIN_DATA = "data/train.csv"
PATH_TEST_FEATURES = "data/test-full.csv"
PATH_TEST_TARGET = "data/full_submission.csv"

def get_train_data(path_train_data=PATH_TRAIN_DATA):
    df_train = pd.read_csv(path_train_data, index_col="Id")
    X_train = df_train.iloc[:,:-1]
    y_train = df_train.iloc[:,-1]
    return X_train, y_train

def get_test_data(path_test_features=PATH_TEST_FEATURES):
    X_test = pd.read_csv(path_test_features, index_col="Id")
    return X_test