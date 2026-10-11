# South German Credit MLOps

# South German Credit MLOps

Credit risk classification using Scikit-learn and MLflow.

## 1. Project Overview

This project implements a reproducible machine learning pipeline to predict credit risk using the South German Credit dataset.

The project follows MLOps principles, including modular Python development, object-oriented programming, configurable hyperparameters, experiment tracking, model evaluation, and version control.

**Technologies:** Python 3.11, Scikit-learn, Pandas, NumPy, Matplotlib, MLflow, and Git.

## 2. Dataset

The South German Credit dataset contains 1,000 observations and 20 predictor variables.

The target variable, `credit_risk`, is transformed so that:

- `0`: Non-risky credit
- `1`: Risky credit

The dataset is divided using stratified sampling and a fixed random seed:

- Development: 700 observations (70%)
- Final test: 300 observations (30%)

For hyperparameter comparison, the development set is further divided into 560 training and 140 validation observations.

**Dataset location:** `data/raw/SouthGermanCredit.asc`

The dataset is not included in the repository. It must be obtained separately and placed at the indicated path.

## 3. Machine Learning Pipeline

The project uses Logistic Regression within a Scikit-learn Pipeline.

Preprocessing is implemented with `ColumnTransformer`:

- Numerical features: `MinMaxScaler`
- Nominal categorical features: `OneHotEncoder`
- Ordinal features: `OrdinalEncoder`

The preprocessing pipeline is fitted only on training data to avoid data leakage.

## 4. Project Structure

```text
south-german-credit-mlops/
├── data/raw/                         # Dataset (not tracked by Git)
├── south_german_credit_mlops/
│   ├── dataset.py                    # Dataset loading
│   ├── features.py                   # Preprocessing
│   └── modeling/
│       └── train.py                  # Training, evaluation, MLflow
├── requirements.txt                  # Python dependencies
├── pyproject.toml                     # Project configuration
├── README.md                          # Documentation
└── .gitignore                         # Git exclusions
```

## 5. Installation

Clone the repository:

```bash
git clone https://github.com/NetsaRom/south-german-credit-mlops.git
cd south-german-credit-mlops
```

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Place the dataset in `data/raw/SouthGermanCredit.asc` before running the training script.

## 6. Training and Experiment Tracking

The training script accepts the following hyperparameters:

| Argument | Default | Description |
|---|---|---|
| `--C` | `1.0` | Inverse regularization strength |
| `--max-iter` | `2000` | Maximum solver iterations |
| `--class-weight` | `balanced` | Class weighting: `balanced` or `none` |
| `--seed` | `42` | Random seed |
| `--final` | Disabled | Train on all development data and evaluate on test |

Run a validation experiment:

```bash
python3 -m south_german_credit_mlops.modeling.train --C 1.0 --max-iter 2000 --class-weight balanced --seed 42
```

Run the selected final configuration:

```bash
python3 -m south_german_credit_mlops.modeling.train --C 1.0 --max-iter 2000 --class-weight balanced --seed 42 --final
```

**Note:** The final test evaluation should only be performed after hyperparameter selection using validation data.

MLflow records hyperparameters, evaluation metrics, confusion matrix artifacts, and the trained Scikit-learn pipeline.

To open the MLflow interface, execute:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then visit `http://127.0.0.1:5000` and open the experiment `South_German_Credit_Experiments`.

## 7. Experimental Results

Five hyperparameter configurations were evaluated using the same stratified data split.

The selected configuration was:

- `C = 1.0`
- `max_iter = 2000`
- `class_weight = balanced`
- `seed = 42`

**Selected validation results:**

| Metric | Value |
|---|---:|
| Accuracy | 0.7500 |
| Precision | 0.5686 |
| Recall | 0.6905 |
| F1-score | 0.6237 |

**Final test results:**

| Metric | Value |
|---|---:|
| Accuracy | 0.7167 |
| Precision | 0.5210 |
| Recall | 0.6889 |
| F1-score | 0.5933 |

The final test confusion matrix was:

```text
[[153, 57],
 [ 28, 62]]
```

The model correctly identified 62 of the 90 risky credit cases in the test set. However, 28 risky cases were classified as non-risky.

## 8. Reproducibility

Reproducibility is supported through fixed random seeds, stratified data splitting, Scikit-learn pipelines, documented dependencies, and MLflow experiment tracking.

Repeated validation runs with identical arguments produced the same metrics and confusion matrix.

## 9. Limitations

The dataset contains only 1,000 observations, and the classes are imbalanced. The model also produces false negatives, which are particularly important in credit risk assessment.

Additional validation, threshold optimization, calibration, fairness analysis, and monitoring would be required before considering real-world deployment.

## 10. Author

NetsaRom
