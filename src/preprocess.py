import pandas as pd

def load_data(path):
    df = pd.read_csv(path, sep=';')
    return df

def clean_data(df):
    df = df.replace('-', '')

    # edad numérica
    df['Edad en años'] = pd.to_numeric(df['Edad en años'], errors='coerce')
    df['Edad en años'] = df['Edad en años'].fillna(df['Edad en años'].median())

    # sexo texto
    df['Sexo (Desc)'] = df['Sexo (Desc)'].fillna('Unknown')

    return df

def group_rare_classes(df, min_samples=10):
    counts = df['GRD'].value_counts()
    rare = counts[counts < min_samples].index
    df['GRD_grouped'] = df['GRD'].apply(lambda x: 'OTHER' if x in rare else x)
    return df

def get_feature_columns(df):
    features = ['Edad en años', 'Sexo (Desc)']

    diag_cols = [c for c in df.columns if 'Diag' in c]
    proc_cols = [c for c in df.columns if 'Proced' in c]

    features.extend(diag_cols)
    features.extend(proc_cols)

    return features
