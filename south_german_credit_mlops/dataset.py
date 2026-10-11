
"""Carga y preparación de South German Credit."""

from pathlib import Path

import pandas as pd


COLUMN_NAMES = [
    "status", "duration", "credit_history", "purpose",
    "amount", "savings", "employment_duration",
    "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age",
    "other_installment_plans", "housing", "number_credits",
    "job", "people_liable", "telephone", "foreign_worker",
    "credit_risk",
]


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Cargar el dataset y prepara la variable objetivo."""

    df = pd.read_csv(file_path, sep=r"\s+")

    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Se esperaban {len(COLUMN_NAMES)} columnas, "
            f"pero se encontraron {df.shape[1]}."
        )

    df.columns = COLUMN_NAMES

    # Mismo tratamiento utilizado en nuestro notebook.
    df["credit_risk"] = 1 - df["credit_risk"]

    return df
