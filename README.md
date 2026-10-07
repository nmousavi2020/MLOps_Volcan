# MLOps_Volcan

An end-to-end MLOps pipeline for estimating volcanic eruption mass using **Gradient Boosting Regression Trees (GBRT)** with logarithmic target transformation.

The project combines machine learning, experiment tracking, model registry, automated testing, Docker, and Docker Compose into a reproducible workflow.

## Scientific Reference

Mousavi, N., Fullea, J., & Mousavi, S. M. (2026). *A machine learning approach for volcanic eruption mass estimation.* Journal of Geophysical Research: Machine Learning and Computation, 3, e2026JH001264.

**DOI:** https://doi.org/10.1029/2026JH001264

If you use this software in academic work, please cite the publication above.

Citation information is also provided in `CITATION.cff`.

---

## Project Overview

The pipeline estimates volcanic eruption mass from geophysical and related predictor variables.

Because eruption masses span several orders of magnitude, the target variable is transformed into logarithmic space before model training.

The workflow includes:

1. Data loading and feature selection
2. Feature standardization
3. Logarithmic target transformation
4. GBRT model training
5. K-fold cross-validation
6. Quantile regression for extreme eruptions
7. Bias correction
8. Prediction generation
9. Model evaluation
10. Visualization
11. MLflow experiment tracking
12. MLflow model registration
13. Automated testing with GitHub Actions
14. Containerized execution with Docker
15. Multi-container orchestration with Docker Compose

---

## MLOps Architecture

```text
                    ┌─────────────────────┐
                    │      Input Data     │
                    │ global.csv          │
                    │ gris_features.csv   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Data Loader      │
                    │ Feature Selection   │
                    │ Standardization     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     ML Pipeline     │
                    │                     │
                    │ Log Transformation  │
                    │ GBRT                │
                    │ K-Fold CV           │
                    │ Quantile Regression │
                    │ Bias Correction     │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │    MLflow       │         │     Outputs     │
        │                 │         │                 │
        │ Experiments     │         │ Predictions     │
        │ Metrics         │         │ Diagnostic      │
        │ Parameters      │         │ Plots           │
        │ Model Registry  │         └─────────────────┘
        └─────────────────┘

        Docker / Docker Compose
                 │
        ┌────────┴────────┐
        ▼                 ▼
   MLflow Server      ML Pipeline
```

---

## Repository Structure

```text
MLOps_Volcan/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── Log_Transformed_ML.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── metrics.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── tracking.py
│
├── tests/
│   ├── __init__.py
│   └── test_config.py
│
├── plots/
│   ├── Observed_vs_Predicted_KFold_Log.png
│   ├── predicted_mass_large.png
│   ├── predicted_mass_small.png
│   ├── training_mass_large.png
│   └── training_mass_small.png
│
├── predictions/
│   └── predicted_mass_test.csv
│
├── global.csv
├── gris_features.csv
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── CITATION.cff
├── LICENSE
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Machine Learning Model

The implementation uses `GradientBoostingRegressor` from **scikit-learn**.

### GBRT Parameters

```python
GBRT_PARAMS = dict(
    loss="huber",
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=4,
    max_features=0.3,
    subsample=1.0,
    random_state=42,
)
```

For extreme eruptions, quantile regression uses:

```python
loss="quantile"
alpha=0.8
```

The configured quantile is:

```python
QUANTILES = [0.8]
```

---

## Log-Target Transformation

The target variable is transformed using:

```python
y_log = np.log(y + EPS)
```

with:

```python
EPS = 1e-6
```

Predictions are transformed back to the original scale using:

```python
y_pred = np.exp(y_pred_log) - EPS
```

This transformation helps accommodate the strongly skewed distribution and large dynamic range of volcanic eruption masses.

---

## Data Processing

The predictor variables are standardized using `StandardScaler`.

The training and test datasets are loaded by:

```text
src/data/loader.py
```

The feature columns are selected by excluding:

```python
["Mass", "lon", "lat"]
```

The geographic coordinates are retained separately for the prediction output.

---

## Quantile Regression

A separate quantile GBRT model is used for extreme eruptions above:

```python
EXTREME_THRESHOLD = 1.5
```

The configured quantile is:

```python
alpha = 0.8
```

This component is intended to improve the treatment of high-mass eruptions.

It is **not equivalent to a complete probabilistic uncertainty model or a P10–P90 prediction interval**.

---

## Bias Correction

A global multiplicative bias correction is applied after prediction:

```python
bias = np.mean(y_train) / np.mean(y_pred)
y_pred *= bias
```

This adjusts the overall prediction scale relative to the mean observed training mass.

---

## Configuration

Central configuration parameters are stored in:

```text
src/config.py
```

Current configuration includes:

```python
N_SPLITS = 2
EPS = 1e-6
EXTREME_THRESHOLD = 1.5
QUANTILES = [0.8]
RANDOM_STATE = 42
TRAIN_FILE = "global.csv"
TEST_FILE = "gris_features.csv"
```

---

## MLflow Experiment Tracking

The project uses **MLflow** for experiment tracking.

Tracked information includes:

* Model parameters
* Cross-validation metrics
* Experiment runs
* Trained models

The MLflow experiment is:

```text
Volcanic_Mass_Prediction
```

The final GBRT model is registered in the MLflow Model Registry as:

```text
volcanic_mass_model
```

A `champion` alias is used for the selected model version.

---

## Docker

The ML pipeline can be executed inside a Docker container.

Build the image:

```bash
docker build -t mlops-volcan .
```

Run the pipeline:

```bash
docker run --rm mlops-volcan
```

The Docker image uses Python 3.12.

---

## Docker Compose

Docker Compose provides the MLflow tracking server and ML pipeline as separate services.

Start MLflow:

```bash
docker compose up -d mlflow
```

Run the ML pipeline:

```bash
docker compose run --rm mlpipeline
```

Check running services:

```bash
docker compose ps
```

The MLflow UI is available at:

```text
http://localhost:5000
```

The pipeline communicates with the MLflow service through the Docker Compose service name:

```text
http://mlflow:5000
```

---

## Automated Testing

Tests are implemented using **pytest**.

Run tests locally:

```bash
pytest
```

Development dependencies are defined in:

```text
requirements-dev.txt
```

The project also uses GitHub Actions to automatically run the test suite on pushes and pull requests.

The workflow is located at:

```text
.github/workflows/tests.yml
```

---

## Installation

Python 3.12 is used for the project environment.

Install the main dependencies:

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

---

## Running the Pipeline

From the repository root:

```bash
python src/Log_Transformed_ML.py
```

For the containerized MLOps workflow:

```bash
docker compose up -d mlflow
docker compose run --rm mlpipeline
```

The pipeline will:

1. Load the datasets.
2. Select model features.
3. Standardize predictor variables.
4. Transform the target into logarithmic space.
5. Perform K-fold cross-validation.
6. Train the final GBRT model.
7. Apply quantile regression to extreme predictions.
8. Apply bias correction.
9. Log experiment information to MLflow.
10. Register the final model.
11. Generate predictions.
12. Generate diagnostic plots.

---

## Input Data

### Training Dataset

```text
global.csv
```

Required columns include:

* `Mass`
* `lon`
* `lat`
* Predictor/feature columns

The first CSV column is treated as the index.

### Test Dataset

```text
gris_features.csv
```

Required columns include:

* Predictor columns compatible with the training dataset
* `lon`
* `lat`

A `Mass` column is not required for the test dataset.

---

## Outputs

### Predictions

The generated prediction file is:

```text
predictions/predicted_mass_test.csv
```

Columns:

| Column           | Description                      |
| ---------------- | -------------------------------- |
| `lon`            | Longitude                        |
| `lat`            | Latitude                         |
| `predicted_mass` | Predicted volcanic eruption mass |

The mass unit is **Gt (gigatonnes)**, assuming the input `Mass` variable is provided in Gt.

### Diagnostic Plots

The pipeline generates:

```text
plots/Observed_vs_Predicted_KFold_Log.png
plots/training_mass_small.png
plots/training_mass_large.png
plots/predicted_mass_small.png
plots/predicted_mass_large.png
```

---

## Evaluation Metrics

The cross-validation procedure reports:

* R²
* Log-space R²
* RMSE
* MAE
* Median Absolute Error
* Median Relative Error
* Mean Log Error
* Normalized error metrics

Log-space metrics are particularly relevant because volcanic eruption masses have a broad dynamic range.

---

## Reproducibility

The workflow uses:

```python
random_state=42
```

For reproducible results, maintain consistent:

* Input datasets
* Feature definitions
* Python version
* Dependency versions
* Model parameters
* Preprocessing procedures

---

## Data and Scientific Considerations

Prediction quality depends on the quality and representativeness of the training dataset.

The model should be applied carefully to samples that are substantially outside the feature distribution represented by the training data.

Predictions should be interpreted in the context of:

* Training-data coverage
* Feature distributions
* Model assumptions
* Data quality
* Scientific context

The quantile-regression component should not be interpreted as a complete probabilistic uncertainty model or as a P10–P90 confidence interval.

---

## Citation

If you use this software or methodology in a publication, thesis, report, or other academic work, please cite:

```bibtex
@article{Mousavi2026VolcanicMass,
  author  = {Mousavi, N. and Fullea, J. and Mousavi, S. M.},
  title   = {A machine learning approach for volcanic eruption mass estimation},
  journal = {Journal of Geophysical Research: Machine Learning and Computation},
  volume  = {3},
  pages   = {e2026JH001264},
  year    = {2026},
  doi     = {10.1029/2026JH001264}
}
```

The repository also contains a `CITATION.cff` file.

---

## License

Copyright © 2026 N. Mousavi, J. Fullea, and S. M. Mousavi.

See the `LICENSE` file for the terms governing use, reproduction, modification, and distribution.

---

## Acknowledgments

This software was developed in support of research on machine-learning-based volcanic eruption mass estimation.

For the scientific methodology and detailed research context, please refer to the associated publication.
