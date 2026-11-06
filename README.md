# Rossmann Store Sales Forecasting

A notebook-based data science project exploring daily sales patterns
across Rossmann stores and comparing historical baselines with
machine-learning models.

The project follows a practical workflow: data understanding, data
cleaning, exploratory data analysis (EDA), baseline modeling,
machine-learning experiments, and chronological model validation.

**Repository:**
[masoud-vosoughi/rossmann-sales-forecasting](https://github.com/masoud-vosoughi/rossmann-sales-forecasting)\
**Dataset:** [Rossmann Store Sales ---
Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales)

------------------------------------------------------------------------

## Project Overview

The goal of this project is to understand the factors associated with
daily store sales and evaluate whether machine-learning models can
improve on a simple historical baseline.

The analysis uses the Rossmann training data and store metadata:

  ------------------------------------------------------------------------
  Dataset                                       Size Description
  --------------------- ---------------------------- ---------------------
  `train.csv`             1,017,209 rows × 9 columns Daily sales
                                                     observations for
                                                     individual stores

  `store.csv`                1,115 rows × 10 columns Store
                                                     characteristics,
                                                     competition, and
                                                     promotion metadata
  ------------------------------------------------------------------------

The training data covers **January 1, 2013 to July 31, 2015** and
includes **1,115 stores**.

This repository is an exploratory and modeling project. It evaluates
historical validation periods; it does **not** currently generate a
Kaggle test-set submission or a production-ready six-week forecasting
service.

## Project Workflow

1.  **Data Understanding** --- inspect dataset structure, date coverage,
    missing values, duplicate records, and store metadata.
2.  **Data Cleaning** --- standardize data types, preserve meaningful
    missing values, validate the cleaned data, and save Parquet files.
3.  **Exploratory Data Analysis** --- examine sales patterns across
    promotions, weekdays, holidays, store characteristics, competition,
    and time.
4.  **Baseline Modeling** --- compare simple historical averages and
    select a benchmark.
5.  **Machine-Learning Modeling** --- evaluate linear regression,
    decision trees, and CatBoost feature configurations.
6.  **Model Evaluation** --- compare the selected CatBoost model with
    the baseline across four chronological monthly validation folds.

## Repository Structure

The notebooks expect the following local data layout. The raw CSV files
are downloaded separately, and the processed Parquet files are generated
by Notebook 02.

``` text
rossmann-sales-forecasting/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── store.csv
│   └── processed/
│       ├── train_clean.parquet
│       └── store_clean.parquet
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_ml_modeling.ipynb
│   └── 06_model_evaluation.ipynb
├── .pre-commit-config.yaml
└── README.md
```

`data/processed/` is created when Notebook 02 saves the cleaned
datasets. Your local repository may contain additional configuration or
environment files not shown above.

## Dataset Setup

1.  Download the **Rossmann Store Sales** dataset from the [Kaggle
    competition
    page](https://www.kaggle.com/competitions/rossmann-store-sales).
2.  Place the two required files in `data/raw/`:

``` text
data/raw/train.csv
data/raw/store.csv
```

3.  The notebooks create the processed files in `data/processed/`:

``` text
data/processed/train_clean.parquet
data/processed/store_clean.parquet
```

Only `train.csv` and `store.csv` are required for the current notebook
workflow. The Kaggle `test.csv` file is not used by these experiments.

Keep the dataset in its original form. Notebook 02 creates working
copies and writes cleaned outputs separately.

## Environment Setup

The notebooks use Python and the following main packages:

-   pandas
-   NumPy
-   Matplotlib
-   scikit-learn
-   CatBoost
-   PyArrow
-   JupyterLab

Create and activate a virtual environment from the repository root:

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the notebook dependencies:

``` bash
python -m pip install pandas numpy matplotlib scikit-learn catboost pyarrow jupyterlab
```

Start JupyterLab:

``` bash
jupyter lab
```

For a reproducible setup, package versions should be pinned in a
`requirements.txt` file or another environment specification. The
commands above install the main packages used by the notebooks but do
not pin their versions.

## Running the Notebooks

Run the notebooks in numerical order from the repository root:

  -------------------------------------------------------------------------------
  Notebook                        Purpose                 Input / Output
  ------------------------------- ----------------------- -----------------------
  `01_data_understanding.ipynb`   Inspect raw sales and   Reads
                                  store data              `data/raw/train.csv`
                                                          and
                                                          `data/raw/store.csv`

  `02_data_cleaning.ipynb`        Clean and validate the  Reads raw CSVs; writes
                                  datasets                two Parquet files to
                                                          `data/processed/`

  `03_eda.ipynb`                  Explore sales patterns  Reads cleaned Parquet
                                  and relationships       files

  `04_baseline_model.ipynb`       Build and evaluate      Reads cleaned training
                                  historical baselines    data

  `05_ml_modeling.ipynb`          Compare                 Reads cleaned sales and
                                  machine-learning models store data
                                  and features

  `06_model_evaluation.ipynb`     Validate the selected   Reads cleaned sales and
                                  model across multiple   store data
                                  months
  -------------------------------------------------------------------------------

**Important:** Run Notebook 02 before Notebooks 03--06 because those
notebooks depend on the processed Parquet files.

## Data Preparation and Analysis

### Data Cleaning

The cleaning workflow includes:

-   Standardizing `StateHoliday` values.
-   Converting `Date` to a datetime type.
-   Converting integer-like metadata fields to nullable integer types.
-   Standardizing store category columns.
-   Reviewing missing competition and Promo2 metadata rather than
    filling unknown values without evidence.
-   Saving cleaned datasets as Parquet files.
-   Reloading the saved files and checking that shapes, data types, and
    values are preserved.

The raw CSV files are not modified by the cleaning notebook.

### Data Quality Considerations

The analysis identified a notable coverage limitation:

-   **180 stores** have no observations from **July 1 to December 31,
    2014** (184 calendar days).
-   Store 988 starts on January 2, 2013, one day after the overall
    dataset start date.
-   Store metadata contains missing competition and Promo2 fields. Some
    Promo2 fields are expected to be missing for stores where
    `Promo2 = 0`.

These limitations are retained and considered when interpreting the
analysis. In particular, year-over-year comparisons for July--December
2014 are not fully like-for-like across stores.

### Exploratory Data Analysis

The EDA notebooks examine:

-   Open and closed store observations.
-   Sales and customer patterns on promotional and non-promotional days.
-   Differences across weekdays and store types.
-   State and school holidays.
-   Seasonal and monthly sales patterns.
-   Competition distance and opening-date metadata.
-   Promo2 schedules.
-   Sales distributions and potential outliers.

For open-store observations, mean sales are approximately **5,929** on
non-promotion days and **8,228** on promotion days. In the within-store
weekday comparison, all 1,115 stores have higher average sales on
promotional weekdays; the mean store-level difference is approximately
**39.95%**.

These are descriptive associations, not causal estimates. Differences
may also reflect store characteristics, timing, and other factors.

## Modeling Approach

### Prediction Target and Evaluation Scope

-   **Target:** `Sales`
-   **Modeling population:** observations where `Open = 1`
-   **Excluded predictor:** `Customers`, because customer counts may not
    be available when a forecast is made.
-   **Primary metrics:** Mean Absolute Error (MAE) and Root Mean Squared
    Error (RMSE).

MAE measures the average absolute prediction error. RMSE gives more
weight to larger errors. Both are reported in the dataset's sales units.

### Historical Baseline

Three simple baseline approaches are compared:

-   Store-level average sales.
-   Average sales by `Store + DayOfWeek`.
-   Average sales by `Store + DayOfWeek + Promo`.

The strongest tested baseline uses the historical mean for each
`Store + DayOfWeek + Promo` group, with a store-level mean as a fallback
when a group has no training observations.

### Machine-Learning Experiments

The experiments include:

-   Linear Regression with one-hot encoded categorical features.
-   Linear Regression with store-weekday historical information.
-   Decision Tree Regression.
-   CatBoost with store, weekday, and promotion features.
-   CatBoost with store metadata and competition features.
-   CatBoost with Promo2 features.
-   CatBoost with historical sales features.

The selected CatBoost model uses store and calendar categories,
promotion status, competition information, and historical sales
summaries. Historical features include:

-   Mean sales over the previous 30 days.
-   Mean sales over the previous 90 days.
-   Historical mean sales for the same store and weekday.

Historical features are constructed using observations before the start
of each forecast month. Sales from the month being predicted are not
used to construct that month's historical features.

## Model Results

### July 2015 Validation

The selected model is **CatBoost + Competition + Historical Sales**.

  Model                          MAE         RMSE
  --------------------- ------------ ------------
  Historical Baseline         732.18     1,038.49
  Selected CatBoost       **718.09**   **991.79**

The selected CatBoost model reduces July open-store MAE by approximately
**1.92%** compared with the historical baseline.

### Multi-Month Validation

Notebook 06 evaluates the selected model on April, May, June, and July
2015. Each monthly fold is trained only on observations dated before the
start of that validation month.

  Validation month     Baseline MAE   CatBoost MAE   MAE improvement
  ------------------ -------------- -------------- -----------------
  April 2015                 972.65         959.77             1.32%
  May 2015                   843.40         805.88             4.45%
  June 2015                  748.15         713.03             4.69%
  July 2015                  732.18         718.09             1.92%

Across all four validation months combined:

  Model                          MAE           RMSE
  --------------------- ------------ --------------
  Historical Baseline         820.21       1,217.09
  Selected CatBoost       **795.60**   **1,148.76**

CatBoost has lower MAE in all four monthly folds. Its combined MAE is
approximately **3.00%** lower than the historical baseline.

### How to Interpret These Results

The results are encouraging, but they should be interpreted within the
scope of the experiments:

-   April--July 2015 are development validation periods, not a final
    untouched test set.
-   These periods were also considered during the broader
    model-development process.
-   The project reports MAE and RMSE. The official Kaggle competition
    uses RMSPE, so these results are not directly comparable with Kaggle
    leaderboard scores.
-   The project does not currently generate predictions for `test.csv`.

## Limitations and Future Work

The current project can be extended in several ways:

1.  Reserve a final future time period that is not used for feature or
    model selection.
2.  Add RMSPE using the official competition evaluation rules.
3.  Investigate the effect of the missing July--December 2014
    observations on store-level and year-over-year comparisons.
4.  Test additional calendar and holiday features using chronological
    validation.
5.  Verify that competition metadata would have been available at each
    historical forecast date.
6.  Tune CatBoost hyperparameters and compare additional models.
7.  Add a prediction workflow for the Kaggle test data, with features
    constructed only from information available at prediction time.
8.  Pin package versions and document the environment for
    reproducibility.

## References

-   [Rossmann Store Sales --- Kaggle
    Competition](https://www.kaggle.com/competitions/rossmann-store-sales)
-   [Kaggle Evaluation
    Metric](https://www.kaggle.com/competitions/rossmann-store-sales/overview/evaluation)

------------------------------------------------------------------------

*This project is intended for learning, experimentation, and portfolio
demonstration. It is not a production forecasting system.*
