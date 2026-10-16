# Scikit-learn: decision matrix and experimental ML workflow

> Compact Obsidian reference · **Supervised ML (regression)** · grounded in seven provided course notebooks. The preprocessing principles apply to classification, but **classification estimators and evaluation metrics are not taught in these notebooks**. The end-to-end implementation below therefore targets regression. All code snippets share the same data variables.

## 0. Core loop: minimal EDA → model → diagnostics → experiment

1. **Mandatory data audit:** types, missing values, invalid entries, duplicates, ranges and target distribution.
2. **Split Train/Test before learned transformations.** Never use Test to select features, transformations or hyperparameters.
3. **Baseline Pipeline:** imputation, encoding, scaling, `LinearRegression`. Record train RMSE and CV mean ± standard deviation.
4. **Diagnose intermediate results:** large train and CV errors suggest underfitting; small train error but large CV error suggests overfitting; unstable coefficients suggest possible collinearity; extreme values suggest comparing robust preprocessing.
5. **Controlled experiments:** `PolynomialFeatures` degrees 1/2/3, `Ridge` and `Lasso` alpha, `ElasticNet` alpha and l1_ratio, `SelectKBest` number k. Use identical folds and metric.
6. **Tune the complete Pipeline** on Train with `GridSearchCV` or `RandomizedSearchCV`; selectors and transformations must be included.
7. **Evaluate Test once** using the CV-selected estimator; inspect residuals and the CV-to-Test gap. Changing the model after looking at Test invalidates its role as an untouched final test.

## 1. Decision matrix: input data → preprocessing

| Observation | Check | Decision / API |
|---|---|---|
| Missing numerical data | `isna().sum()` | `SimpleImputer(strategy="median")` or `"mean"` |
| Missing categories | `value_counts(dropna=False)` | `SimpleImputer(strategy="most_frequent")` or `"constant"` |
| Special missing marker `999` | `unique()`, business meaning | convert to NaN or `SimpleImputer(missing_values=999)` |
| Nominal categories | `value_counts()` | `OneHotEncoder(handle_unknown="ignore")` |
| Ordinal categories | domain-defined ordering | `OrdinalEncoder(categories=[...])` |
| Different scales | `describe()` | `StandardScaler` or `MinMaxScaler` |
| Extreme values | boxplots and `zscore()` | investigate; compare `RobustScaler` with `StandardScaler` |
| Right-skewed features | histogram, `skew()` | `FunctionTransformer(np.log1p)` for nonnegative inputs |
| Nonlinear relationship | scatter, residuals, MI | `PolynomialFeatures`, optionally `KBinsDiscretizer` |
| Constant columns | `nunique()` | `VarianceThreshold` |
| Redundant predictors | correlation / VIF | test `SelectKBest`, `RFE`, `SequentialFeatureSelector`, `Lasso` |
| Pearson ≈ 0 but nonlinear signal | mutual information | `mutual_info_regression`; do not drop automatically |
| Duplicate records | `duplicated()` | verify and remove erroneous duplicates *before the split* |
| Mixed types or units in a field | `dtypes`, `unique()` | meaning-aware pandas cleaning; `FunctionTransformer` for consistent rules |

> Ordinary `sklearn.Pipeline` transformers cannot remove training rows while simultaneously updating `X` and `y`. The outlier notebook removes rows separately. The `|z| > 3` heuristic is meaningful mainly for approximately normal distributions and does not imply automatic deletion.

## 2. Decision matrix: intermediate results → next experiment

| Diagnostic result | Possible explanation | Next experiment | Track |
|---|---|---|---|
| High train **and** CV RMSE | underfitting, insufficient features | `PolynomialFeatures(degree=2/3)`, weaker regularization | CV RMSE, train–CV gap |
| Low train RMSE; much higher CV RMSE | overfitting | `Ridge(alpha↑)`, `Lasso(alpha↑)`, lower polynomial degree, `SelectKBest(k↓)` | CV RMSE and fold variability |
| CV scores vary strongly by fold | unstable model or dataset composition | fixed shuffled `KFold`, regularization, inspect fold differences | CV mean ± std |
| Coefficients vary in magnitude/sign | potential collinearity | `Ridge`, `Lasso`, correlation, VIF | coefficient std and CV RMSE |
| Huge polynomial coefficients | instability with expanded feature space | scale after `PolynomialFeatures`, add `Ridge` | CV RMSE and `coef_` |
| Many exact zero Lasso coefficients | L1 feature sparsity | compare alpha and `SelectKBest(k)` | nonzero count and CV RMSE |
| Weak Pearson but positive MI | possible nonlinear relation | polynomial features or binning | CV RMSE |
| Removing outliers worsens validation | genuine informative extreme values | restore rows; compare `RobustScaler` | fairly comparable CV RMSE |
| Strong CV but weak final Test score | distribution shift, luck or over-selection | analyze why without tuning against Test | recorded CV vs Test |

**Recommended experiment order:** linear baseline → polynomial features → Ridge → Lasso → SelectKBest → ElasticNet → combined tuning. This is a practical heuristic, not a claim about which model wins.

## 3. Common dataset setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

TARGET = "price"
NUM = ["area", "age"]
CAT = ["location", "type"]
X, y = df[NUM + CAT], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

print(df.dtypes, df.duplicated().sum())
print(X_train.isna().sum(), X_train[NUM].describe())
print(X_train[NUM].skew(), y_train.describe())
```

## 4. Baseline Pipeline and KFold CV

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocess = ColumnTransformer([
    ("num", num_pipe, NUM), ("cat", cat_pipe, CAT)
])
base = Pipeline([
    ("preprocess", preprocess), ("model", LinearRegression())
])

def report_cv(estimator, X=X_train, y=y_train):
    scores = -cross_val_score(
        estimator, X, y, cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    print(f"CV RMSE = {scores.mean():.3f} ± {scores.std():.3f}")
    return scores

report_cv(base)
```

The negative scorer is a scikit-learn convention; negating its result yields positive RMSE. Compare mean **and standard deviation**, not merely the best score.

## 5. Preprocessing toolbox: drop-in templates

```python
from sklearn.preprocessing import (
    MinMaxScaler, RobustScaler, OrdinalEncoder, KBinsDiscretizer,
    FunctionTransformer, PolynomialFeatures
)

StandardScaler()
MinMaxScaler()
RobustScaler()

OrdinalEncoder(categories=[["low", "medium", "high"]])

FunctionTransformer(np.log1p, feature_names_out="one-to-one")  # x >= 0
KBinsDiscretizer(n_bins=5, encode="onehot", strategy="quantile")

poly_num = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scale", StandardScaler()),
])

```

```python
poly_preprocess = ColumnTransformer([
    ("num", poly_num, NUM), ("cat", cat_pipe, CAT)
])
```

```python
from sklearn.base import BaseEstimator, TransformerMixin
class MyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
# Pipeline([("custom", MyTransformer()), ...])
```

Use `OrdinalEncoder` only when order is genuinely meaningful. Keep ordered and nominal features in different `ColumnTransformer` branches. Apply log transforms only when valid for every incoming observation. A custom transformer is a scaffold, not a ready-made cleaning algorithm.

## 6. Experiment A: polynomial degree × Ridge / Lasso / ElasticNet

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV

candidate = Pipeline([
    ("preprocess", poly_preprocess),
    ("model", Ridge())
])

param_grid = [
    {"preprocess__num__poly__degree": [1, 2, 3],
     "model": [Ridge(max_iter=20000)],
     "model__alpha": [0.01, 0.1, 1, 10, 100]},
    {"preprocess__num__poly__degree": [1, 2, 3],
     "model": [Lasso(max_iter=20000)],
     "model__alpha": [0.001, 0.01, 0.1, 1]},
    {"preprocess__num__poly__degree": [1, 2],
     "model": [ElasticNet(max_iter=20000)],
     "model__alpha": [0.01, 0.1, 1],
     "model__l1_ratio": [0.2, 0.5, 0.8]},
]
search = GridSearchCV(
    candidate, param_grid, cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1, return_train_score=True
)
search.fit(X_train, y_train)
results = pd.DataFrame(search.cv_results_)
cols = ["params", "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]
print(results[cols].sort_values("rank_test_score").head(10))
print("best", search.best_params_, "CV RMSE", -search.best_score_)

results["train_rmse"] = -results["mean_train_score"]
results["cv_rmse"] = -results["mean_test_score"]
results["gap"] = results["cv_rmse"] - results["train_rmse"]
```

**Interpret the intermediate `cv_results_`:** `train_rmse` substantially below `cv_rmse` indicates a potential generalization gap; both high suggests insufficient model flexibility or weak features; `std_test_score` measures cross-fold variability. Hyperparameters use the `step__parameter` naming scheme.

## 7. Experiment B: feature selection inside CV (`SelectKBest`)

```python
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

select_pipe = Pipeline([
    ("preprocess", preprocess),
    ("select", SelectKBest(score_func=f_regression)),
    ("model", Ridge())
])
sel_search = GridSearchCV(
    select_pipe,
    {"select__k": [3, 5, "all"],
     "model__alpha": [0.1, 1, 10]},
    cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
)
sel_search.fit(X_train, y_train)
print(sel_search.best_params_, -sel_search.best_score_)
selected = sel_search.best_estimator_
feature_names = selected.named_steps["preprocess"].get_feature_names_out()
mask = selected.named_steps["select"].get_support()
print(feature_names[mask])

# SelectKBest(score_func=mutual_info_regression, k=5)
```

`f_regression` tests univariate linear associations, while `mutual_info_regression` can capture nonlinear association. Feature selection *before* CV leaks information from validation folds. The `select__k` candidate must not exceed the number of transformed features (the string `"all"` is safe).

## 8. Coefficient stability across CV folds

```python
from sklearn.base import clone

cat_levels = [sorted(X_train[c].dropna().unique()) for c in CAT]
fixed_cat = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(categories=cat_levels,
                          handle_unknown="ignore", sparse_output=False)),
])
fixed_pre = ColumnTransformer([("num", num_pipe, NUM), ("cat", fixed_cat, CAT)])
fixed_pipe = Pipeline([("preprocess", fixed_pre), ("model", Ridge(alpha=1))])

folds = []
for tr, va in cv.split(X_train, y_train):
    fitted = clone(fixed_pipe).fit(X_train.iloc[tr], y_train.iloc[tr])
    names = fitted.named_steps["preprocess"].get_feature_names_out()
    weights = fitted.named_steps["model"].coef_
    folds.append(pd.Series(weights, index=names))
weights_by_fold = pd.concat(folds, axis=1)
coef_stats = pd.DataFrame({
    "mean": weights_by_fold.mean(axis=1),
    "std": weights_by_fold.std(axis=1),
    "sign_changes": weights_by_fold.gt(0).any(axis=1)
                    & weights_by_fold.lt(0).any(axis=1)
})
print(coef_stats.sort_values("std", ascending=False))

best = search.best_estimator_
print(best.named_steps["model"].coef_, best.named_steps["model"].intercept_)
print(best.named_steps["preprocess"].get_feature_names_out())
```

Linear estimator coefficients are available **after `fit()`**, not as a default training-time trajectory. To compare coefficients across folds, use the same output feature layout; fixed category vocabularies make `OneHotEncoder` outputs comparable. Compare magnitude, sign flips and CV RMSE together. With feature selection, apply the selector’s `get_support()` to output feature names.

## 9. Targeted diagnostics: outliers, associations, alternative selection

```python
import matplotlib.pyplot as plt
from scipy.stats import zscore, f_oneway, chi2_contingency
from sklearn.feature_selection import (
    VarianceThreshold, RFE, SequentialFeatureSelector
)
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

X_train["area"].hist(); plt.show()
X_train.boxplot(column="area"); plt.show()
z = zscore(X_train["area"].dropna())
print("suspected outliers:", (abs(z) > 3).sum())

a = X_train[NUM].fillna(X_train[NUM].median())
print(a.corr(method="pearson"))
print(a.corr(method="spearman"))
print(pd.Series(mutual_info_regression(a, y_train, random_state=42), index=NUM))

# Pipeline([("preprocess", preprocess), ("variance", VarianceThreshold()), ...])

RFE(LinearRegression(), n_features_to_select=3)
SequentialFeatureSelector(Ridge(), n_features_to_select=3,
                          direction="forward", cv=cv)

print(pd.Series([variance_inflation_factor(a.to_numpy(), i)
                 for i in range(a.shape[1])], index=a.columns))

print(sm.OLS(y_train, sm.add_constant(a)).fit().pvalues)

parts = [y_train[X_train["location"] == c]
         for c in X_train["location"].dropna().unique()]
print(f_oneway(*parts))

table = pd.crosstab(X_train["location"], X_train["type"])
chi2, p, dof, expected = chi2_contingency(table)
n = table.to_numpy().sum()
v = np.sqrt((chi2/n) / min(table.shape[0]-1, table.shape[1]-1))
print(p, v)
```

Here, `zscore` detects candidates, not automatic errors. Correlations and mutual information are diagnostics, not hard deletion thresholds. The VIF and statsmodels sections assume appropriately prepared numeric inputs. `f_oneway` requires its statistical assumptions to be considered; chi-square also has expected-count requirements. `f_classif` and `chi2` are mentioned in the feature-selection notebook as classification-related scoring choices, but a complete classification training workflow is outside the provided course scope. For `chi2`, values must be nonnegative.

## 10. Randomized tuning and the final untouched Test

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform
from sklearn.metrics import root_mean_squared_error, r2_score

random_search = RandomizedSearchCV(
    Pipeline([("preprocess", poly_preprocess),
              ("model", ElasticNet(max_iter=20000))]),
    param_distributions={
        "preprocess__num__poly__degree": [1, 2, 3],
        "model__alpha": loguniform(1e-3, 10),
        "model__l1_ratio": uniform(0.05, 0.9),
    },
    n_iter=25, cv=cv, scoring="neg_root_mean_squared_error",
    random_state=42, n_jobs=-1, return_train_score=True
)
random_search.fit(X_train, y_train)

final_estimator = search.best_estimator_
pred = final_estimator.predict(X_test)
print("Final test RMSE:", root_mean_squared_error(y_test, pred))
print("Final test R²:", r2_score(y_test, pred))
residuals = y_test - pred
print(residuals.describe())
```

Choose among `search`, `sel_search` and `random_search` **using training cross-validation alone**. Only then execute a single final Test evaluation. Do not use the Test score to retune the final model.

## 11. Experiment log (copy for each dataset)

| Trial | Mean CV RMSE | CV RMSE std | Train RMSE | Feature count | Hypothesis / observation |
|---|---:|---:|---:|---:|---|
| Linear baseline | | | | | |
| Polynomial degree 2 | | | | | |
| Ridge alpha=… | | | | | |
| Lasso alpha=… | | | | | |
| SelectKBest k=… | | | | | |
| ElasticNet alpha=…, l1_ratio=… | | | | | |

**Course sources:** `01_feature_engineering_basics.ipynb`, `03_pipelines.ipynb`, `01_bias_variance_guide.ipynb`, `02_regularization_guide.ipynb`, `03_feature_selection_guide.ipynb`, `04_detecting_outliers_ru.ipynb`, `1_cross_validation_hyperparameter_tuning.ipynb`. Checking duplicates and resolving inconsistent units are data-quality safeguards rather than a dedicated ML algorithm presented in the notebooks. Adapt column names, valid feature ranges, and statistical assumptions to your data.
