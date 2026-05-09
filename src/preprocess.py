import pandas as pd

def load_data(path):
    df = pd.read_csv(path, sep=';', low_memory=False)
    return df

def clean_data(df):

    df = df.replace('-', '')

    if 'Edad en años' in df.columns:
        df['Edad en años'] = pd.to_numeric(
            df['Edad en años'],
            errors='coerce'
        )

        df['Edad en años'] = df['Edad en años'].fillna(
            df['Edad en años'].median()
        )

    if 'Sexo (Desc)' in df.columns:
        df['Sexo (Desc)'] = df['Sexo (Desc)'].fillna('Unknown')

    return df

def filter_ultra_rare_classes(df, min_samples=5):

    counts = df['GRD'].value_counts()

    valid_classes = counts[counts >= min_samples].index

    df = df[df['GRD'].isin(valid_classes)]

    return df

def group_rare_classes(df, min_samples=30):

    counts = df['GRD'].value_counts()

    rare = counts[counts < min_samples].index

    df['GRD_grouped'] = df['GRD'].apply(
        lambda x: 'OTHER' if x in rare else x
    )

    return df

def get_feature_columns(df):

    features = [
        'Edad en años',
        'Sexo (Desc)'
    ]

    diag_cols = [c for c in df.columns if 'Diag' in c][:3]

    proc_cols = [c for c in df.columns if 'Proced' in c][:2]

    features.extend(diag_cols)
    features.extend(proc_cols)

    return [c for c in features if c in df.columns]