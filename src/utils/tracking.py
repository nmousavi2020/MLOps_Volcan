import mlflow


def start_run(run_name):
    mlflow.set_experiment("Volcanic_Mass_Prediction")
    return mlflow.start_run(run_name=run_name)


def log_params(params):
    mlflow.log_params(params)


def log_metrics(metrics):
    mlflow.log_metrics(metrics)