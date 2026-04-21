from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

def train_model(X_train, y_train, cat_features):
    model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.08,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        verbose=50
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features
    )

    return model
