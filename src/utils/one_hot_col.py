import pandas as pd

def one_hot_col(df,col_name):
    """
        change colonne to multiple colomne one hot encoded

        input:
            (pandas: DataFrame): df
            (String): col_name: name of the column to encod

        output:
            (pandas: DataFrame): df_encoded
    """
    df_encoded = pd.get_dummies(df,columns=[col_name])
    df_encoded = df_encoded.replace({True:1, False:0})
    return df_encoded
