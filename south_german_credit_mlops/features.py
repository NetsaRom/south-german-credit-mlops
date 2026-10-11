
"""
Preprocesamiento de variables para South German Credit.

Este módulo define los grupos de variables y construye
un ColumnTransformer de Scikit-learn.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)


# Variables numéricas
NUMERIC_FEATURES = [
    "duration",
    "amount",
    "installment_rate",
    "residence_since",
    "age",
    "number_credits",
    "people_liable",
]

# Variables ordinales
ORDINAL_FEATURES = [
    "status",
    "savings",
    "employment_duration",
    "property",
]

# Variables nominales
CATEGORICAL_FEATURES = [
    "credit_history",
    "purpose",
    "personal_status_sex",
    "other_debtors",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]


def build_preprocessor() -> ColumnTransformer:
    """
    Construye el preprocesamiento para el modelo.

    Returns
    -------
    ColumnTransformer
        Transformador de variables numéricas,
        ordinales y nominales.
    """

    # Pipeline para variables numéricas
    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
        ]
    )

    # Pipeline para variables nominales
    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    # Pipeline para variables ordinales
    ordinal_pipeline = Pipeline(
        steps=[
            ("ordinal", OrdinalEncoder()),
        ]
    )

    # Combinar todas las transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("nom", categorical_pipeline, CATEGORICAL_FEATURES),
            ("ord", ordinal_pipeline, ORDINAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor
