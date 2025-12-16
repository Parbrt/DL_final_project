import pandas as pd


def del_empty_colums(df):
    col = df.columns

    t = []
    for i in col:
        t.append(df[i])

    temp = []
    for i in range(len(t)):
        if t[i] == 1:
            temp.append(i)
            df.drop(i, inplace=True)
            print(f"colonnes supprimées: {df.columns[i]}")

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

def del_unused_weapons(df):
    df_unused = df.copy()
    ct_weapons = ['ct_ak47']