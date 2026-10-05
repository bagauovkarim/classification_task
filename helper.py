import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler


def load_clean_data(path="bank-additional-full.csv"):
    df = pd.read_csv(path, sep=";")
    
    df["y"] = np.where(df["y"] == "yes", 1, 0)
    
    df = df.drop(columns=["duration"])

    df["pdays"] = np.where(df["pdays"] != 999, 1, 0)

    df = df.drop_duplicates()

    df = df.replace("unknown", np.nan)

    return df


def get_preprocessor():

    numeric_cols = ["age", "campaign", "pdays", "previous",
                    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                    "euribor3m", "nr.employed"]
    onehot_cols = ["job", "marital", "poutcome", "day_of_week", "month"]
    ord_col = ["education"]
    binary_cols = ["default", "housing", "loan", "contact"]

    ohe_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    ord_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(categories=[
            ["illiterate", "basic.4y", "basic.6y", "basic.9y",
             "high.school", "professional.course", "university.degree"]
        ])),
    ])

    bin_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(categories=[
            ["no", "yes"],              # default
            ["no", "yes"],              # housing
            ["no", "yes"],              # loan
            ["cellular", "telephone"],  # contact
        ])),
    ])

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric_cols),
        ("ohe", ohe_pipe, onehot_cols),
        ("ord", ord_pipe, ord_col),
        ("bin", bin_pipe, binary_cols),
    ])

    return preprocessor


def evaluate_models_cv(models, X, y, preprocessor, cv, scoring=None, sampler=None):
    if scoring is None:
        scoring = ["roc_auc", "f1_macro", "precision_macro", "recall_macro", "accuracy"]
    results = {}
    for name, model in models:
        steps = []
        steps.append(("prep", clone(preprocessor)))
        if sampler is not None:
            steps.append(("sampler", clone(sampler)))
        steps.append(("model", clone(model)))
        pipe = ImbPipeline(steps)
        cv_res = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        results[name] = {m: cv_res["test_" + m].mean() for m in scoring}

    return pd.DataFrame(results).T



def get_preprocessor_with_exclude(exclude=None):
    exclude = set(exclude or [])
    numeric_cols = ["age", "campaign", "pdays", "previous",
                    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                    "euribor3m", "nr.employed"]
    onehot_cols = ["job", "marital", "poutcome", "day_of_week", "month"]
    ord_col = ["education"]
    binary_cols = ["default", "housing", "loan", "contact"]

    numeric_cols = [c for c in numeric_cols if c not in exclude]
    onehot_cols = [c for c in onehot_cols if c not in exclude]
    ord_col = [c for c in ord_col if c not in exclude]
    binary_cols = [c for c in binary_cols if c not in exclude]

    ohe_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    ord_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(categories=[
            ["illiterate", "basic.4y", "basic.6y", "basic.9y",
             "high.school", "professional.course", "university.degree"]
        ])),
    ])

    bin_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(categories=[
            ["no", "yes"],              # default
            ["no", "yes"],              # housing
            ["no", "yes"],              # loan
            ["cellular", "telephone"],  # contact
        ])),
    ])

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric_cols),
        ("ohe", ohe_pipe, onehot_cols),
        ("ord", ord_pipe, ord_col),
        ("bin", bin_pipe, binary_cols),
    ])

    return preprocessor


def get_preprocessor_scaled():

    numeric_cols = ["age", "campaign", "pdays", "previous",
                    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                    "euribor3m", "nr.employed"]
    onehot_cols = ["job", "marital", "poutcome", "day_of_week", "month"]
    ord_col = ["education"]
    binary_cols = ["default", "housing", "loan", "contact"]

    ohe_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    ord_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(categories=[
            ["illiterate", "basic.4y", "basic.6y", "basic.9y",
             "high.school", "professional.course", "university.degree"]
        ])),
    ])

    bin_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(categories=[
            ["no", "yes"],              # default
            ["no", "yes"],              # housing
            ["no", "yes"],              # loan
            ["cellular", "telephone"],  # contact
        ])),
    ])

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("ohe", ohe_pipe, onehot_cols),
        ("ord", ord_pipe, ord_col),
        ("bin", bin_pipe, binary_cols),
    ])

    return preprocessor