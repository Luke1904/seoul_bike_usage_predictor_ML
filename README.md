# SWIM - Seoul Weather-Informed Modelling

This project predicts hourly Seoul bike demand as a continuous regression
problem. It compares baselines, linear regressions, separate weekday/weekend
periodic regressions, and random-forest regressors.

## Design

- "src/preprocessing.py" owns cleaning, the chronological split, and Fourier
  hour features (sin/cos terms for three harmonics).
- "src/models.py" owns reusable regression models and feature definitions.
- "src/experiment.py" provides the BikeDemandExperiment class that owns one
  shared train/validation/test split and runs the baseline/linear suite.
- "src/random_forest.py" tunes only RandomForestRegressor models; there is no
  classification target or classification code.
- "src/generate_report_results.py" combines the linear and forest metrics and
  saves report-ready tables and figures.

The dataset has one canonical location: "src/data/SeoulBikeData.csv".
Generated CSV files are stored in "results/", and figures are stored in
"results/figures/".

## Environment setup

From the repository root:

~~~bash
conda env create -f environment.yml
conda activate seoul-bike-predictor
~~~

Create the environment only once. On later runs, only activate it.

## Run the project

Run the interactive exploratory workflow:

~~~bash
python src/main.py
~~~

Run the full reproducible report workflow:

~~~bash
python src/random_forest.py
python src/season_improvement.py
python src/generate_report_results.py
~~~

The random-forest grid search is the slow step and can take several minutes.
Run it again when the data, features, split, or tuning grid changes. The final
generator reads "results/rf_results.csv" and creates:

- "results/linear_model_results.csv"
- "results/model_comparison.csv"
- "results/figures/model_comparison.png"
- "results/figures/weather_error_reduction.png"
- "results/figures/october_prediction_week.png"

The seasonal script creates
"results/figures/seasonal_percentage_improvement.png".
