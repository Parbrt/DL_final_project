import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from src.utils.data_prep import *

def process_data():
    """
        process data and create csv file with the prepared data

    """

    df = pd.read_csv('data/csgo_round_snapshots.csv')

    df_without_unused_weapons = del_unused_weapons(df)
    df_one_hot_col = one_hot_col(df_without_unused_weapons,'map')
    df_final = name2bin(df_one_hot_col,'round_winner')


    y_cat = df_final['round_winner']
    y_reg = df_final[['ct_money','ct_health']]
    X = df_final.drop(columns=['round_winner','ct_money','ct_health'])

    X_train, X_test, y_train_reg, y_test_reg, y_train_cat, y_test_cat = train_test_split(
    X, y_reg, y_cat, test_size=0.2, random_state=42
    )

    scaler_X = preprocessing.StandardScaler()
    scaler_y = preprocessing.StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_reg_scaled = scaler_y.fit_transform(y_train_reg)
    y_test_reg_scaled = scaler_y.transform(y_test_reg)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    cols_reg = ['ct_money', 'ct_health']
    y_train_reg_scaled = pd.DataFrame(y_train_reg_scaled, columns=cols_reg)
    y_test_reg_scaled = pd.DataFrame(y_test_reg_scaled, columns=cols_reg)

    X_train_scaled.to_csv('data/processed/X_train.csv', index=False)
    X_test_scaled.to_csv('data/processed/X_test.csv', index=False)

    y_train_reg_scaled.to_csv('data/processed/y_train_reg_scaled.csv', index=False)
    y_test_reg_scaled.to_csv('data/processed/y_test_reg_scaled.csv', index=False)

    y_train_cat.to_csv('data/processed/y_train_cat.csv', index=False)
    y_test_cat.to_csv('data/processed/y_test_cat.csv', index=False)
    
    return True

def get_data():
    """
        read the csv files and return DataFrame

        output:
            (pandas: DataFrame): x_train, x_test, y_train_reg, y_test_reg, y_train_cat, y_test_cat
    """
    x_train = pd.read_csv("data/processed/X_train.csv")
    x_test = pd.read_csv("data/processed/X_test.csv")
    
    y_train_reg = pd.read_csv("data/processed/y_train_reg_scaled.csv")
    y_test_reg = pd.read_csv("data/processed/y_test_reg_scaled.csv")

    y_train_cat = pd.read_csv("data/processed/y_train_cat.csv")
    y_test_cat = pd.read_csv("data/processed/y_test_cat.csv")
    
    return x_train, x_test, y_train_reg, y_test_reg, y_train_cat, y_test_cat

def get_scaler():
    """
        output:
            scaler
    """

    df = pd.read_csv('data/csgo_round_snapshots.csv')

    df_without_unused_weapons = del_unused_weapons(df)
    df_one_hot_col = one_hot_col(df_without_unused_weapons,'map')
    df_final = name2bin(df_one_hot_col,'round_winner')


    y_cat = df_final['round_winner']
    y_reg = df_final[['ct_money','ct_health']]
    X = df_final.drop(columns=['round_winner','ct_money','ct_health'])

    X_train, X_test, y_train_reg, y_test_reg, y_train_cat, y_test_cat = train_test_split(
    X, y_reg, y_cat, test_size=0.2, random_state=42
    )

    scaler_y = preprocessing.StandardScaler()
    

    scaler_y.fit(y_train_reg)
    
    return scaler_y
