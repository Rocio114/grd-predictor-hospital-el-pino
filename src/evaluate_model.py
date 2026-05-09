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


def evaluate(
    model,
    X_test,
    y_test,
    model_name="model",
    label_map=None
):

    # CARPETA DE SALIDA

    output_dir = f"reports/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # PREDICCIONES

    preds = model.predict(X_test)

    preds = pd.Series(preds.reshape(-1)).astype(str)
    y_test = pd.Series(y_test).astype(str)

    # MÉTRICAS GLOBALES

    acc = accuracy_score(y_test, preds)

    f1_weighted = f1_score(
        y_test,
        preds,
        average='weighted'
    )

    f1_macro = f1_score(
        y_test,
        preds,
        average='macro'
    )

    # CLASSIFICATION REPORT

    report = classification_report(
        y_test,
        preds,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    # 🔥 HACER LEGIBLE (MAPEO DE CLASES)
    if label_map is not None:
        report_df.index = report_df.index.map(
            lambda x: label_map.get(x, x)
        )

    # PRINT RESULTADOS

    print(f"\nRESULTS - {model_name.upper()}")
    print("Accuracy:", round(acc, 4))
    print("Weighted F1:", round(f1_weighted, 4))
    print("Macro F1:", round(f1_macro, 4))

    # GUARDAR MÉTRICAS

    with open(
        f"{output_dir}/metrics.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Weighted F1: {f1_weighted}\n")
        f.write(f"Macro F1: {f1_macro}\n")

    report_df.to_csv(
        f"{output_dir}/classification_report.csv",
        encoding="utf-8-sig"
    )

    # MATRIZ DE CONFUSIÓN

    preds_labels = preds.copy()
    y_labels = y_test.copy()

    if label_map is not None:
        preds_labels = preds.map(label_map).fillna(preds)
        y_labels = y_test.map(label_map).fillna(y_test)

    top_n = 20

    most_common = [
        x[0]
        for x in Counter(y_labels).most_common(top_n)
    ]

    indices = [
        i for i, y in enumerate(y_labels)
        if y in most_common
    ]

    y_true_filtered = [
        y_labels.iloc[i]
        for i in indices
    ]

    y_pred_filtered = [
        preds_labels.iloc[i]
        for i in indices
    ]

    cm = confusion_matrix(
        y_true_filtered,
        y_pred_filtered,
        labels=most_common,
        normalize='true'
    )

    cm_df = pd.DataFrame(
        cm,
        index=most_common,
        columns=most_common
    )

    cm_df.to_csv(
        f"{output_dir}/confusion_matrix.csv",
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(18, 14))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=most_common,
        yticklabels=most_common
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Real")

    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/confusion_matrix.png"
    )

    plt.close()

    # FEATURE IMPORTANCE

    if hasattr(model, "get_feature_importance"):
        importances = model.get_feature_importance()

    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    else:
        importances = None

    if importances is not None:

        feature_importance = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": importances
        })

        feature_importance = feature_importance.sort_values(
            by="Importance",
            ascending=False
        )

        feature_importance.to_csv(
            f"{output_dir}/feature_importance.csv",
            index=False,
            encoding="utf-8-sig"
        )

        plt.figure(figsize=(12, 8))

        sns.barplot(
            data=feature_importance.head(15),
            x="Importance",
            y="Feature"
        )

        plt.title(f"Top Features - {model_name}")

        plt.tight_layout()

        plt.savefig(
            f"{output_dir}/top_features.png"
        )

        plt.close()

    # TOP F1 CLASSES

    metrics_only = report_df.drop(
        ['accuracy', 'macro avg', 'weighted avg'],
        errors='ignore'
    )

    metrics_only = metrics_only.sort_values(
        by='f1-score',
        ascending=False
    )

    metrics_only.to_csv(
        f"{output_dir}/class_metrics_sorted.csv",
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(12, 10))

    sns.barplot(
        data=metrics_only.head(15),
        x='f1-score',
        y=metrics_only.head(15).index
    )

    plt.title(f"Top F1 Classes - {model_name}")

    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/top_f1_classes.png"
    )

    plt.close()


    # RETURN MÉTRICAS

    return acc, f1_weighted, f1_macro