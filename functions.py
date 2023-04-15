import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def scale_and_ohe(vector):
    numeric_cols = vector.select_dtypes([int, float])
    categorical_cols = vector.select_dtypes('O')

    scaler = StandardScaler().fit(numeric_cols)
    ohe = OneHotEncoder(sparse = False, drop = 'first', handle_unknown='ignore').fit(categorical_cols)

    categoricals_bin = pd.DataFrame(data = ohe.transform(categorical_cols),columns = ohe.get_feature_names_out() )
    numeric_scaled = pd.DataFrame(data = scaler.transform(numeric_cols), columns = scaler.get_feature_names_out())

    return pd.concat([numeric_scaled, categoricals_bin], axis = 1)

def countplot_with_percentages(data, x_col, title_name, ax=None, hue=None):
    if ax is None:
        ax = plt.gca()
    
    plot = sns.countplot(data=data, x=x_col, ax=ax, hue=hue)
    plot.set_title(title_name)

    total = len(data)
    for p in plot.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2,
                height + 10,  # Aquí se ajusta el espacio entre el porcentaje y la barra
                '{:1.2f}%'.format(100 * height / total),
                ha="center")

    plt.show()

def clean_numeric_cols(df, col_list):
    for col in col_list:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

def find_tricky_classes(df, var, var_ob, threshold):

    for clas_s in df[var].value_counts().keys():
        try:
            if 1 in list(df[df[var] == clas_s][var_ob].value_counts().keys()):
                indx = list(df[df[var] == clas_s][var_ob].value_counts().keys()).index(1)
                freq_arrest = list(df[df[var] == clas_s][var_ob].value_counts('%'))[indx]
                if freq_arrest > threshold:
                    print(f"{clas_s}, {freq_arrest}, frecuencia de la clase:{df[var].value_counts()[clas_s]}")
        except ValueError:
            continue
        
def find_no_tricky(df, var, var_ob):
    for clas_s in df[var].value_counts().keys():
        if 1 not in list(df[df[var] == clas_s][var_ob].value_counts().keys()):
                    print(clas_s,'--------' ,df[var].value_counts()[clas_s])
                    

def clean_columns(df, col_dict):
    for col, val in col_dict.items():
        if type(val) == str:
            df[col] = np.where(df[col] == ' ', val, df[col])
        if val == 0 or val == 1:
            df[col] = np.where(df[col] == ' ', 0, 1)
            

def check_vars(df, reg):
    for col in list(df.filter(regex='^'+reg+ '_', axis=1).columns):
        fr = df[df[col] == 1]['arstmade'].value_counts('%')
        fr1 = df[df[col] == 1]['arstmade'].value_counts()
        print(f'La variable {col} ocurre {fr1[0]+fr1[1]} veces de las cuales el {round(fr[1]*100, 2)}% son arrestos')
        
def make_ult_var(name, valuable_combinations):
    vars_names = []
    for x in valuable_combinations:
        var_name = "_".join(list(x))
        
        vars_names.append(var_name)
        df[var_name] = np.where((df[x[0]] == 1) & (df[x[1]] == 1) & (df[x[2]] == 1), 1, 0)
        df_test[var_name] = np.where((df_test[x[0]] == 1) & (df_test[x[1]] == 1) & (df_test[x[2]] == 1), 1, 0)

    df[name] = np.where(df[vars_names].isin([1]).any(axis=1), 1, 0)
    df.drop(columns=vars_names, axis=1, inplace=True)
    
    df_test[name] = np.where(df_test[vars_names].isin([1]).any(axis=1), 1, 0)
    df_test.drop(columns=vars_names, axis=1, inplace=True)
    
def make_ult_var_2(name, valuable_combinations):
    
    vars_names = []
    
    for x in valuable_combinations:
        var_name = "_".join(list(x))
        
        vars_names.append(var_name)
        df[var_name] = np.where((df[x[0]] == 1) & (df[x[1]] == 1) & (df[x[2]] == 1) & (df[x[3]] == 1), 1, 0)
        df_test[var_name] = np.where((df_test[x[0]] == 1) & (df_test[x[1]] == 1) & (df_test[x[2]] == 1) & (df_test[x[3]] == 1), 1, 0)
        
    df[name] = np.where(df[vars_names].isin([1]).any(axis=1), 1, 0)
    df.drop(columns=vars_names, axis=1, inplace=True)
    
    df_test[name] = np.where(df_test[vars_names].isin([1]).any(axis=1), 1, 0)
    df_test.drop(columns=vars_names, axis=1, inplace=True)
    
    
def interest_vars(df, var, ov, threshold):
    df_var = df[df[var] == 1]
    print(f'En un data set donde solo ocurre la variable dada {var} Frecuencia:{len(df_var)}')
    print('')
    for col in df_var:
        if len(df_var[col].value_counts()) == 2:
            if 1 and 0 in list(df_var[col].value_counts().keys()):
                try:
                    if df_var[df_var[col] == 1][ov].value_counts('%')[1] > threshold:
                        freq = df_var[col].value_counts()[1]
                        arsts = df_var[df_var[col] == 1][ov].value_counts('%')[1]
                        print(f'La variable {col} ocurre {freq}/{len(df[df[var]==1])} veces de las cuales {round(arsts*100, 2)}% son {ov}')
                except KeyError:
                    continue
                
def create_mask(df, df_test, custom_var, columns):
    df[custom_var] = np.where(df[columns].isin([1]).any(axis=1), 1, 0)
    df_test[custom_var] = np.where(df_test[columns].isin([1]).any(axis=1), 1, 0)