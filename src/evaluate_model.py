import os
from collections import Counter

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

    # predicciones
    preds = model.predict(X_test)

    # métricas principales
    acc = accuracy_score(y_test, preds)

    f1 = f1_score(
        y_test,
        preds,
        average='weighted'
    )

    # classification report como diccionario
    report = classification_report(
        y_test,
        preds,
        zero_division=0,
        output_dict=True
    )

    # dataframe del reporte
    report_df = (
        pd.DataFrame(report)
        .transpose()
    )

    print("\nRESULTS")
    print("Accuracy:", round(acc, 4))
    print("Weighted F1:", round(f1, 4))

    print("\nClassification Report:")
    print(report_df.round(2))

    # crear carpeta
    os.makedirs("reports", exist_ok=True)

    # guardar métricas
    with open(
        "reports/metrics.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(f"Accuracy: {acc}\n")
        f.write(f"Weighted F1: {f1}\n")

    # guardar classification report CSV
    report_df.to_csv(
        "reports/classification_report.csv",
        encoding="utf-8-sig"
    )

    # MATRIZ DE CONFUSIÓN MEJORADA

    top_n = 20

    # clases más frecuentes
    most_common = [
        x[0]
        for x in Counter(y_test).most_common(top_n)
    ]

    # filtrar top clases
    indices = [
        i
        for i, y in enumerate(y_test)
        if y in most_common
    ]

    y_true_filtered = [
        y_test.iloc[i]
        for i in indices
    ]

    y_pred_filtered = [
        preds[i]
        for i in indices
    ]

    # matriz normalizada
    cm = confusion_matrix(
        y_true_filtered,
        y_pred_filtered,
        labels=most_common,
        normalize='true'
    )

    # guardar matriz como CSV
    cm_df = pd.DataFrame(
        cm,
        index=most_common,
        columns=most_common
    )

    cm_df.to_csv(
        "reports/confusion_matrix.csv",
        encoding="utf-8-sig"
    )

    # gráfico matriz
    plt.figure(figsize=(18, 14))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=most_common,
        yticklabels=most_common
    )

    plt.title(
        "Normalized Confusion Matrix - Top 20 Classes"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Real")

    plt.xticks(
        rotation=90,
        fontsize=8
    )

    plt.yticks(
        rotation=0,
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        "reports/confusion_matrix_top20_normalized.png"
    )

    plt.close()

    # FEATURE IMPORTANCE

    feature_importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": model.get_feature_importance()
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    # guardar CSV
    feature_importance.to_csv(
        "reports/feature_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # gráfico top 15
    plt.figure(figsize=(12, 8))

    sns.barplot(
        data=feature_importance.head(15),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 15 Feature Importance")

    plt.tight_layout()

    plt.savefig(
        "reports/top_features.png"
    )

    plt.close()

    # TOP CLASES POR F1

    metrics_only = report_df.drop(
        ['accuracy', 'macro avg', 'weighted avg'],
        errors='ignore'
    )

    metrics_only = metrics_only.sort_values(
        by='f1-score',
        ascending=False
    )

    # guardar top métricas
    metrics_only.to_csv(
        "reports/class_metrics_sorted.csv",
        encoding="utf-8-sig"
    )

    # gráfico top 15 F1
    plt.figure(figsize=(12, 10))

    sns.barplot(
        data=metrics_only.head(15),
        x='f1-score',
        y=metrics_only.head(15).index
    )

    plt.title("Top 15 Classes by F1 Score")

    plt.tight_layout()

    plt.savefig(
        "reports/top_f1_classes.png"
    )

    plt.close()

    return acc, f1