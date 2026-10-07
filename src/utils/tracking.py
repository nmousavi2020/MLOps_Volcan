import mlflow
import mlflow.sklearn

MLFLOW_URI = "http://localhost:5000"


def start_run(run_name):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Volcanic_Mass_Prediction")
    return mlflow.start_run(run_name=run_name)


def log_params(params):
    mlflow.log_params(params)


def log_metrics(metrics):
    mlflow.log_metrics(metrics)


def log_model(model, name="volcanic_mass_model"):
    mlflow.sklearn.log_model(
        model,
        name=name,
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )