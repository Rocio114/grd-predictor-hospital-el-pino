# GRD Predictor Hospital El Pino

Machine Learning project focused on predicting Diagnosis Related Groups (GRD) using clinical patient records from Hospital El Pino.

---

## Project Structure

```bash
grd-predictor-hospital-el-pino/
│
├── data/
│   └── dataset_elpino.csv
│
├── models/
│   └── best_model.cbm
│
├── reports/
│   ├── catboost/
│   ├── decision_tree/
│   ├── random_forest/
│   ├── class_distribution.csv
│   └── model_comparison.csv
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── catboost_info/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

##Features Used

The model uses structured clinical and administrative data:

Age
Sex
Diagnostic codes
Procedure codes

---

## Machine Learning Models

The project evaluates three models:

CatBoostClassifier
Decision Tree
Random Forest

Each model is trained and evaluated independently.

---

# Model Selection Strategy

The final model is selected automatically based on:

Weighted F1-score

The model with the best performance is saved as the final model.

---

# CatBoost Configuration
Iterations: 80
Depth: 8
Learning rate: 0.15
Loss function: MultiClass
Evaluation metric: TotalF1

---

# Outputs Generated
 Reports
reports/class_distribution.csv
reports/model_comparison.csv
Model-specific evaluation folders:
reports/catboost/
reports/decision_tree/
reports/random_forest/

---

# Final Model
models/best_model.cbm (CatBoost selected as best model in current run)

---

## Installation

Clone repository:

```bash
git clone <repository-url>
cd grd-predictor-hospital-el-pino
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Project

```bash
python main.py
```

---
