> ℹ️ **About this file**
>
> - This is a rewritten version of [../README.md](../README.md).
> - All paths point one folder up, for example `../assets/` and `../data/`.
> - A solution path in German is in [../summary/loesungsweg.ipynb](../summary/loesungsweg.ipynb).

# King County House Prices Project

A one-day project with real housing data.

- You predict the sale price of houses in King County, Seattle and its surroundings.
- You use a **regression** model.
- You combine two tables, clean the data and explore it.
- You build a preprocessing **pipeline**.
- You tune two models and compare them with a **baseline**.

![Map of King County with each of the 21,597 house sales of 2014 and 2015 drawn as a dot at its location and coloured by its sale price. The most expensive houses lie around Bellevue, Mercer Island and the shores of Lake Washington, and prices fall towards the south of the county.](../assets/king_county_price_map.png)

- Each dot is one house sale, coloured by its price.
- The colour scale stops at 1.5 million dollars.
- So the few more expensive houses appear in the same yellow.

## 📖 New terms

| Term | In one sentence |
| :--- | :--- |
| **regression** | A **regression** predicts a number, here the sale price of a house. |
| **pipeline** | A **pipeline** chains preprocessing steps and a model into one object. |
| **baseline** | A **baseline** is a simple model that your real models have to beat. |
| **cross-validation** | **Cross-validation** scores a model on several parts of the training set in turn. |
| **template** | A **template** is a GitHub repository that you copy to start your own. |
| **placeholder** | A **placeholder** is text in angle brackets that you replace with your own value. |

## 🎯 Learning objectives

By the end of this project, you can:

1. Combine two related tables after checking the merge key. Then split the data into a training set and a test set before you explore it.
2. Build a preprocessing **pipeline** with a `ColumnTransformer`. It encodes the categorical columns that are stored as numbers. Then tune two models with **cross-validation**.
3. Compare every model with a mean **baseline** and a rule-based **baseline** in one table of training and test scores. Then describe where the best model makes its largest errors.

## 🗺️ Learning path

| File | Description |
| :--- | :--- |
| [**01 Loading the Data**](01_loading_the_data.ipynb) | Load both tables, check the merge key, combine them and save the result. |
| [**02 King County Project**](02_king_county_project.ipynb) | The project: the data description and the task in eight steps. It loads the data saved by notebook 01. |
| [**Solution path**](../summary/loesungsweg.ipynb) | One possible solution for both notebooks, step by step, in German. |
| [**Summary**](../summary/zusammenfassung.ipynb) | A short summary in German for review, with a glossary. |

### Additional folders and files

| File or folder | Description |
| :--- | :--- |
| [**Assets**](../assets/) | The diagram of the project steps and the map of the house sales. The Mermaid source is in `../assets/diagrams/`. |
| [**Data**](../data/) | The two source tables `king_county_house_sales.csv` and `king_county_house_details.csv`. |
| [**pyproject.toml**](../pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](../uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Text in angle brackets like `<repo-name>` is a **placeholder**. Replace it, including the brackets, with your own value. For example, `cd <repo-name>` becomes `cd my-king-county-project`.

### 1. Create the repository from the template

Click **Use this template** on GitHub. When you create the repository:

1. Set yourself as the **Owner**.
2. Choose a repository name.
3. Turn off **Include all branches**.
4. Click **Create repository**.

> [!IMPORTANT]
> If you work in pairs or groups, only **one person** does this step.

---

### 2. Add collaborators, for pairs and groups only

If you work with teammates:

1. Open the repository on GitHub.
2. Go to **Settings → Collaborators**.
3. Add your teammates as collaborators.
4. Share the repository link with your team.

Your teammates accept the invitation before they continue.

---

### 3. Clone the repository

Copy the SSH URL from the **Code** button on GitHub. Then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL looks like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the project folder and install the dependencies

This installs all dependencies. It also creates a virtual environment in `.venv/`.

```bash
cd <repo-name>
uv sync
```

---

### 5. Open the notebooks

> [!NOTE]
> Open VS Code from the project root. Then it finds the environment from `uv sync` automatically.

Start VS Code in the project root folder:

```bash
code .
```

Then open a notebook. Select the Python environment from `uv sync` as the kernel.

## References and further reading

- [**scikit-learn: Pipelines and composite estimators**](https://scikit-learn.org/stable/modules/compose.html): how to combine preprocessing and models with `Pipeline` and `ColumnTransformer`.
- [**scikit-learn: Tuning the hyperparameters of an estimator**](https://scikit-learn.org/stable/modules/grid_search.html): grid search, random search and how they use **cross-validation**.
- [**scikit-learn: Polynomial features**](https://scikit-learn.org/stable/modules/preprocessing.html#polynomial-features): how to expand features into squares and interactions.
- [**Common pitfalls in the interpretation of coefficients of linear models**](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html): why correlated features make coefficients unstable.
- [**House Sales in King County, USA on Kaggle**](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction): the original dataset.
