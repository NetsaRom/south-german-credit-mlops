
"""Entrenamiento del modelo South German Credit."""

from sklearn.model_selection import train_test_split

from south_german_credit_mlops.dataset import load_data
from south_german_credit_mlops.features import build_preprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import argparse
import mlflow
import tempfile
from pathlib import Path
import mlflow.sklearn

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


class CreditRiskModel:
    """Modelo de clasificación de riesgo crediticio."""

    def __init__(
        self,
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.C,
                        max_iter=self.max_iter,
                        class_weight=self.class_weight,
                        random_state=self.random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def train(self, X_train, y_train):
        """Entrena el pipeline."""
        self.pipeline.fit(X_train, y_train)

    def predict(self, X):
        """Genera predicciones."""
        return self.pipeline.predict(X)

    
    def evaluate(self, X_test, y_test):
        """Evalúa el modelo con datos de prueba."""

        y_pred = self.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, y_pred, zero_division=0
            ),
        }

        matrix = confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        )

        return metrics, matrix

    
    def save_confusion_matrix(self, matrix, output_path):
        """Guarda la matriz de confusión como imagen PNG."""

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=[0, 1],
        )

        display.plot(cmap="Blues", values_format="d")

        plt.title("South German Credit - Confusion Matrix")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()



def parse_args():
    """Lee los hiperparámetros desde la terminal."""

    parser = argparse.ArgumentParser(
        description="Entrenamiento de South German Credit"
    )

    parser.add_argument(
        "--C", type=float, default=1.0,
        help="Inverso de la regularización"
    )

    parser.add_argument(
        "--max-iter", type=int, default=2000,
        help="Número máximo de iteraciones"
    )

    parser.add_argument(
        "--class-weight",
        type=str,
        choices=["balanced", "none"],
        default="balanced",
        help="Ponderación de clases"
    )

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla aleatoria"
    )

    parser.add_argument(
    "--final",
    action="store_true",
    help="Entrena con los 700 registros de desarrollo y evalúa en test",
    )
    return parser.parse_args()


def main():

    args = parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment("South_German_Credit_Experiments")

    # 1. Cargar el dataset
    df = load_data("data/raw/SouthGermanCredit.asc")

    # 2. Separar variables predictoras y objetivo
    X = df.drop(columns=["credit_risk"])
    y = df["credit_risk"]

    # 3. Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=args.seed,
        stratify=y,
    )

    # Conservar los 700 registros para el entrenamiento final
    X_development = X_train.copy()
    y_development = y_train.copy()

    # Separar validación del conjunto de entrenamiento
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=args.seed,
        stratify=y_train,
    )

    # Seleccionar datos según el modo de ejecución
    if args.final:
        X_fit = X_development
        y_fit = y_development
        X_eval = X_test
        y_eval = y_test
        evaluation_type = "test"
    else:
        X_fit = X_train
        y_fit = y_train
        X_eval = X_val
        y_eval = y_val
        evaluation_type = "val"

    # 4. Crear nuestro modelo
    model = CreditRiskModel(
    C=args.C,
    max_iter=args.max_iter,
    class_weight=(
        None if args.class_weight == "none"
        else args.class_weight
    ),
    random_state=args.seed,
    )

    print("Entrenamiento:", X_train.shape)
    print("Validación:", X_val.shape)
    print("Prueba reservada:", X_test.shape)

    run_name = (
        "final_test"
        if args.final
        else f"C_{args.C}_weight_{args.class_weight}"
    )

    with mlflow.start_run(run_name=run_name):

        # Registrar hiperparámetros
        mlflow.log_params({
            "C": args.C,
            "max_iter": args.max_iter,
            "class_weight": args.class_weight,
            "seed": args.seed,
        })

        # Entrenar modelo
        model.train(X_fit, y_fit)

        # Evaluar modelo
        metrics, matrix = model.evaluate(X_eval, y_eval)

        # Identificar las métricas como métricas de validación
        metrics = {
            f"{evaluation_type}_{name}": value
            for name, value in metrics.items()
        }


        # Registrar métricas
        mlflow.log_metrics(metrics)


        # Crear y registrar la matriz de confusión
        with tempfile.TemporaryDirectory() as temp_dir:
            matrix_path = Path(temp_dir) / "confusion_matrix.png"

            model.save_confusion_matrix(matrix, matrix_path)

            mlflow.log_artifact(str(matrix_path))

        
        # Registrar el pipeline entrenado en MLflow
        mlflow.sklearn.log_model(
            sk_model=model.pipeline,
            name="credit_risk_model",
        )


        # Mostrar resultados
        print("\n--- MÉTRICAS DEL MODELO ---")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\n--- MATRIZ DE CONFUSIÓN ---")
        print(matrix)




if __name__ == "__main__":
    main()
