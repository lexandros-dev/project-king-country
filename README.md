# King County House Prices Project

A one-day project with real housing data. You predict the sale price of houses in King County (Seattle and its surroundings) with a regression model: combine two tables, clean the data, explore it, build a preprocessing pipeline, and tune and compare two models against a baseline.

![Map of King County with each of the 21,597 house sales of 2014 and 2015 drawn as a dot at its location and coloured by its sale price. The most expensive houses lie around Bellevue, Mercer Island and the shores of Lake Washington, and prices fall towards the south of the county](assets/king_county_price_map.png)

Each dot is one house sale, coloured by its price; the colour scale stops at \$1.5 million, so the few more expensive houses appear in the same yellow.

## Learning Objectives

By the end of this project, you should be able to:

- Combine two related tables after checking the merge key, and split the data into a training and a test set before exploring it.
- Build a preprocessing `Pipeline` with a `ColumnTransformer` that encodes the categorical columns stored as numbers, and tune two models with cross-validation.
- Compare every model with a mean and a rule-based baseline in one table of training and test scores, and describe where the best model makes its largest errors.

## Learning Path

| File | Description |
|---|---|
| [**01 - Loading the Data**](01_loading_the_data.ipynb) | Load both tables, check the merge key, combine them and save the result. |
| [**02 - King County Project**](02_king_county_project.ipynb) | The project: the data description and the task in eight steps. It loads the data saved by notebook 01. |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**Assets**](assets/) | The diagram of the project's steps (Mermaid source in `assets/diagrams/`) and the map of the house sales. |
| [**Data**](data/) | The two source tables, `king_county_house_sales.csv` and `king_county_house_details.csv`. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a **placeholder**. Replace it including the `< >` brackets with your own value. For example, `cd <repo-name>` becomes `cd my-king-county-project`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies and creates a virtual environment in (`.venv/`).

```bash
cd <repo-name>
uv sync
```

---

### 5. Open the Notebooks

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

## References & Further Reading

- [**scikit-learn: Pipelines and composite estimators**](https://scikit-learn.org/stable/modules/compose.html): Combining preprocessing and models with `Pipeline` and `ColumnTransformer`.
- [**scikit-learn: Tuning the hyperparameters of an estimator**](https://scikit-learn.org/stable/modules/grid_search.html): Grid search, randomized search and how they use cross-validation.
- [**scikit-learn: Polynomial features**](https://scikit-learn.org/stable/modules/preprocessing.html#polynomial-features): Expanding features into squares and interactions.
- [**Common pitfalls in the interpretation of coefficients of linear models**](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html): Why correlated features make coefficients unstable.
- [**House Sales in King County, USA (Kaggle)**](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction): The original dataset.
