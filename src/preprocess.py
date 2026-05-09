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

    # 1. extraer familia (código antes del guion)
    df['GRD_family'] = df['GRD'].str.split(' - ').str[0]

    # 2. contar por familia
    family_counts = df['GRD_family'].value_counts()

    # 3. familias raras
    rare_families = family_counts[
        family_counts < min_samples
    ].index

    # 4. agrupar raras como OTHER (pero a nivel familia)
    df['GRD_grouped'] = df['GRD_family'].apply(
        lambda x: x if x not in rare_families else 'OTHER'
    )

    return df


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
    
