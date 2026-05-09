import os
import json
import pandas as pd

from src.preprocess import (
    load_data,
    clean_data,
    filter_ultra_rare_classes,
    group_rare_classes,
    get_feature_columns
)

from src.train_model import (
    split_data,
    train_model,
    train_decision_tree,
    train_random_forest
)

from src.evaluate_model import evaluate


def main():

    path = "data/dataset_elpino.csv"

    # CARGA Y PREPROCESAMIENTO

    df = load_data(path)
    df = clean_data(df)
    df = filter_ultra_rare_classes(df)

    df, label_map = group_rare_classes(df)

    # GUARDAR LABEL MAP

    os.makedirs("reports", exist_ok=True)

    with open("reports/label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)

    # DISTRIBUCIÓN DE CLASES

    class_dist = (
        df['GRD_grouped']
        .value_counts()
        .reset_index()
    )

    class_dist.columns = ['GRD', 'Count']

    class_dist.to_csv(
        "reports/class_distribution.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # FEATURES Y TARGET

    features = get_feature_columns(df)

    X = df[features]
    y = df["GRD_grouped"]


    # ENCODING PARA SKLEARN

    X_encoded = X.copy()

    for col in X_encoded.columns:
        if X_encoded[col].dtype.name in ['object', 'category']:
            X_encoded[col] = (
                X_encoded[col]
                .astype(str)
                .fillna("UNKNOWN")
                .astype('category')
                .cat.codes
            )


    # CATBOOST CATEGÓRICAS

    cat_features = [
        i for i, col in enumerate(X.columns)
        if X[col].dtype.name in ['object', 'category']
    ]

    # SPLIT DATA

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    (
        X_train_enc,
        X_val_enc,
        X_test_enc,
        y_train_enc,
        y_val_enc,
        y_test_enc
    ) = split_data(X_encoded, y)

    # 1. CATBOOST

    print("\n========================")
    print("TRAINING CATBOOST")
    print("========================")

    cat_model = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        cat_features
    )

    cat_results = evaluate(
        cat_model,
        X_test,
        y_test,
        model_name="catboost",
        label_map=label_map
    )


    # 2. DECISION TREE
    
    print("\n========================")
    print("TRAINING DECISION TREE")
    print("========================")

    dt_model = train_decision_tree(
        X_train_enc,
        y_train_enc
    )

    dt_results = evaluate(
        dt_model,
        X_test_enc,
        y_test_enc,
        model_name="decision_tree",
        label_map=label_map
    )

    # 3. RANDOM FOREST

    print("\n========================")
    print("TRAINING RANDOM FOREST")
    print("========================")

    rf_model = train_random_forest(
        X_train_enc,
        y_train_enc
    )

    rf_results = evaluate(
        rf_model,
        X_test_enc,
        y_test_enc,
        model_name="random_forest",
        label_map=label_map
    )

    # COMPARACIÓN FINAL

    comparison_df = pd.DataFrame([
        {
            "Model": "CatBoost",
            "Accuracy": cat_results[0],
            "Weighted_F1": cat_results[1],
            "Macro_F1": cat_results[2]
        },
        {
            "Model": "DecisionTree",
            "Accuracy": dt_results[0],
            "Weighted_F1": dt_results[1],
            "Macro_F1": dt_results[2]
        },
        {
            "Model": "RandomForest",
            "Accuracy": rf_results[0],
            "Weighted_F1": rf_results[1],
            "Macro_F1": rf_results[2]
        }
    ])

    comparison_df.to_csv(
        "reports/model_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========================")
    print("MODEL COMPARISON")
    print("========================")
    print(comparison_df)

    # =========================
    # MEJOR MODELO
    # =========================
    best_model_name = comparison_df.loc[
        comparison_df["Macro_F1"].idxmax(),
        "Model"
    ]

    print("\n========================")
    print(f"BEST MODEL: {best_model_name}")
    print("========================")

    if best_model_name == "CatBoost":
        best_model = cat_model

    elif best_model_name == "DecisionTree":
        best_model = dt_model

    elif best_model_name == "RandomForest":
        best_model = rf_model


    # GUARDAR MODELO

    os.makedirs("models", exist_ok=True)

    if best_model_name == "CatBoost":
        best_model.save_model("models/best_model.cbm")
    else:
        import joblib
        joblib.dump(best_model, "models/best_model.pkl")


if __name__ == "__main__":
    main()