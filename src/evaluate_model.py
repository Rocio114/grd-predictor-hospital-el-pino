import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate(model, X_test, y_test):

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    f1 = f1_score(
        y_test,
        preds,
        average='weighted'
    )

    report = classification_report(
        y_test,
        preds
    )

    print("\nRESULTS")
    print("Accuracy:", round(acc, 4))
    print("Weighted F1:", round(f1, 4))

    print("\nClassification Report:")
    print(report)

    os.makedirs("reports", exist_ok=True)

    # guardar métricas
    with open("reports/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Weighted F1: {f1}\n")

    # guardar classification report
    with open("reports/classification_report.txt", "w") as f:
        f.write(report)

    # matriz de confusión
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Real")

    plt.tight_layout()

    plt.savefig("reports/confusion_matrix.png")

    # importancia de variables
    feature_importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": model.get_feature_importance()
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=feature_importance.head(10),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 10 Feature Importance")

    plt.tight_layout()

    plt.savefig("reports/top_features.png")

    return acc, f1
