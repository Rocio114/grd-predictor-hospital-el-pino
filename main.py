import os
from src.preprocess import load_data, clean_data, group_rare_classes, get_feature_columns
from src.train_model import split_data, train_model
from src.evaluate_model import evaluate

def main():

    path = "data/dataset_elpino.csv"

    df = load_data(path)
    df = clean_data(df)
    df = group_rare_classes(df)

    features = get_feature_columns(df)

    X = df[features]
    y = df["GRD_grouped"]

    cat_features = [i for i,col in enumerate(X.columns) if X[col].dtype == 'object']

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train, cat_features)

    evaluate(model, X_test, y_test)

    os.makedirs("models", exist_ok=True)
    model.save_model("models/final_grd_model.cbm")

    print("\nModel saved in models/final_grd_model.cbm")

if __name__ == "__main__":
    main()
