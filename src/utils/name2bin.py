import pandas as pd

def name2bin(df, col_name,cat_1 = "CT",cat_2 = "T"):
    df = df.replace({cat_1:1, cat_2:0})
    return df

