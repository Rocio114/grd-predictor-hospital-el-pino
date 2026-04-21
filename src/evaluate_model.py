from sklearn.metrics import accuracy_score, f1_score, classification_report

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')

    print("\nRESULTS")
    print("Accuracy:", round(acc,4))
    print("Weighted F1:", round(f1,4))

    return acc, f1
