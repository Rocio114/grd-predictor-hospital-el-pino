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
│   └── final_grd_model.cbm
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Features Used

The model uses:

- Age
- Sex
- Diagnostic codes
- Procedure codes

---

## Model

Algorithm used:

- CatBoostClassifier

Configuration:

- 150 iterations
- depth = 6
- learning_rate = 0.1

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

## Output

The trained model will be saved in:

```bash
models/final_grd_model.cbm
```

---

## Authors

- Rocío