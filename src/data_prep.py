import pandas as pd

def del_unused_weapons(df, treshold = 0.01):
    df_unused = df.copy()
    weapons_cols = [col for col in df.columns if 'weapon' in col]
    unused_weapons = []
    for col in weapons_cols:
        presence_rate = df[col].mean()
        if presence_rate < treshold:
            unused_weapons.append(col)

    print(f"unused columns:{unused_weapons}")
    cols_to_drop = unused_weapons
    for col in cols_to_drop:
        if col in unused_weapons:
            df_unused = df_unused.drop(col, axis = 1)
    return df_unused

def aggregate_weapons(df):
    df_agreg = df.copy()

    shotguns = ['t_weapon_nova', 't_weapon_xm1014', 't_weapon_sawedoff', 't_weapon_mag7',
                'ct_weapon_nova', 'ct_weapon_xm1014', 'ct_weapon_sawedoff', 'ct_weapon_mag7']

    heavy_mg = ['t_weapon_m249', 't_weapon_negev',
                'ct_weapon_m249', 'ct_weapon_negev']

    df_agreg['t_category_shotguns'] = df_agreg[[c for c in shotguns if c.startswith('t_')]].sum(axis=1)
    df_agreg['t_category_heavy'] = df_agreg[[c for c in heavy_mg if c.startswith('t_')]].sum(axis=1)

    df_agreg['ct_category_shotguns'] = df_agreg[[c for c in shotguns if c.startswith('ct_')]].sum(axis=1)
    df_agreg['ct_category_heavy'] = df_agreg[[c for c in heavy_mg if c.startswith('ct_')]].sum(axis=1)

    cols_to_drop = shotguns + heavy_mg
    cols_to_drop = [c for c in cols_to_drop if c in df_agreg.columns]

    df_agreg = df_agreg.drop(columns=cols_to_drop)

    return df_agreg

def one_hot_col(df,col_name):
    df_encoded = pd.get_dummies(df,columns=[col_name])
    df_encoded = df_encoded.replace({True:1, False:0})
    return df_encoded

def name2bin(df, col_name,cat_1 = "CT",cat_2 = "T"):
    df = df.replace({cat_1:1, cat_2:0})
    return df