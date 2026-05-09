from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def split_data(X, y):

    # 80% train
    # 10% validation
    # 10% test

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
    )


def train_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
    cat_features
):

    model = CatBoostClassifier(
        iterations=80,
        depth=8,
        learning_rate=0.15,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        auto_class_weights="Balanced",
        verbose=20,
        random_seed=42,
        early_stopping_rounds=20
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid)
    )

    return model

def train_decision_tree(X_train, y_train):

    model = DecisionTreeClassifier(
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    return model

def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    return model