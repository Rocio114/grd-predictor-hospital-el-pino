import pandas as pd


def load_data(path):

    df = pd.read_csv(
        path,
        sep=';',
        low_memory=False
    )

    return df


def clean_data(df):

    # reemplazar guiones vacíos
    df = df.replace('-', 'UNKNOWN')
    df = df.replace('', 'UNKNOWN')

    # limpiar columnas de texto
    for col in df.select_dtypes(include='object').columns:

        df[col] = (
            df[col]
            .fillna('UNKNOWN')
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # edad
    if 'Edad en años' in df.columns:

        df['Edad en años'] = pd.to_numeric(
            df['Edad en años'],
            errors='coerce'
        )

        df['Edad en años'] = df['Edad en años'].fillna(
            df['Edad en años'].median()
        )

        # crear grupo etario
        df['Grupo_Edad'] = pd.cut(
            df['Edad en años'],
            bins=[-1, 18, 40, 65, 120],
            labels=[
                'NIÑO',
                'ADULTO_JOVEN',
                'ADULTO',
                'ADULTO_MAYOR'
            ]
        )

        df['Grupo_Edad'] = (
            df['Grupo_Edad']
            .cat.add_categories(['UNKNOWN'])
            .fillna('UNKNOWN')
            .astype(str)
        )

    # sexo
    if 'Sexo (Desc)' in df.columns:

        df['Sexo (Desc)'] = df['Sexo (Desc)'].fillna(
            'UNKNOWN'
        )

    return df


def filter_ultra_rare_classes(df, min_samples=15):

    counts = df['GRD'].value_counts()

    valid_classes = counts[
        counts >= min_samples
    ].index

    df = df[
        df['GRD'].isin(valid_classes)
    ]

    return df


def group_rare_classes(df, min_samples=80):

    df = df.copy()

    # 1. separar código y nombre
    df[['GRD_code', 'GRD_name']] = df['GRD'].str.split(' - ', n=1, expand=True)

    df['GRD_name'] = df['GRD_name'].fillna(df['GRD_code'])

    # 2. contar por código
    counts = df['GRD_code'].value_counts()

    rare = counts[counts < min_samples].index

    # 3. agrupación
    df['GRD_grouped'] = df['GRD_code'].apply(
        lambda x: x if x not in rare else 'OTHER'
    )

    # 4. construir label_map automático (CLAVE)
    label_map = df[['GRD_code', 'GRD_name']].drop_duplicates()

    label_map = dict(
        zip(label_map['GRD_code'], label_map['GRD_name'])
    )

    # agregar OTHER manualmente
    label_map['OTHER'] = 'Otros diagnósticos'

    return df, label_map


def get_feature_columns(df):

    features = [

        # variables demográficas
        'Edad en años',
        'Grupo_Edad',
        'Sexo (Desc)',

        # diagnósticos principales
        'Diag 01 Principal (cod+des)',
        'Diag 02 Secundario (cod+des)',
        'Diag 03 Secundario (cod+des)',

        # procedimientos principales
        'Proced 01 Principal (cod+des)',
        'Proced 02 Secundario (cod+des)'
    ]

    # solo devolver columnas existentes
    selected = [
        c for c in features
        if c in df.columns
    ]

    return selected
    
