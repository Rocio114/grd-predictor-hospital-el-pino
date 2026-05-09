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
        iterations=150,
        depth=6,
        learning_rate=0.1,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        verbose=20,
        random_seed=42,
        allow_writing_files=False
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features
    )

    return model