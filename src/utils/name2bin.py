import pandas as pd

def name2bin(df, col_name,cat_1 = "CT",cat_2 = "T"):
    """
        replace categorical names in a column by 0 and 1

        input:
            (pandas: DataFrame): df
            (String): col_name
            (String): cat_1, cat_2: the categorical names to change
        
        output:
            (pandas: DataFrame): df
    """
    
    df = df.replace({cat_1:1, cat_2:0})
    return df

