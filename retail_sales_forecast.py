"""
Retail Sales Forecast
=====================
Converted from Retail_Sales_Forecast.ipynb

Pipeline:
  1. Preprocessing   - load, merge, clean the store/sales/features datasets
  2. Method 1         - impute MarkDown1-5, CPI, Unemployment, then predict Weekly_Sales
  3. Method 2         - drop MarkDown1-5, impute CPI, Unemployment, then predict Weekly_Sales
  4. Inference sample - reload the pickled models and predict on a sample row

Run with:  python retail_sales_forecast.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,

)
from xgboost import XGBRegressor

# ======================================================
# CONFIGURATION
# ======================================================
COMPARE_MODELS = False

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

# Directory where trained models get pickled (relative, cross-platform)
DATASET_DIR = "dataset"
MODEL_DIR = "model"


STORE_URL = os.path.join(DATASET_DIR, "stores_data_set.csv")
SALES_URL = os.path.join(DATASET_DIR, "sales_data_set.csv")
FEATURES_URL = os.path.join(DATASET_DIR, "Features_data_set.csv")
SQL_URL = os.path.join(DATASET_DIR, "df_sql.csv")


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------
def load_raw_data():
    """Load the three raw source datasets."""
    df_store = pd.read_csv(STORE_URL)
    df_sales = pd.read_csv(SALES_URL)
    df_feature = pd.read_csv(FEATURES_URL)

    print("df_store null values:\n", df_store.isnull().sum(), "\n")
    print("df_sales null values:\n", df_sales.isnull().sum(), "\n")
    print("df_feature null values:\n", df_feature.isnull().sum(), "\n")

    return df_store, df_sales, df_feature


def merge_datasets(df_store, df_sales, df_feature):
    """Combine store/sales/feature data and reconcile the date ranges."""
    # combine store and sales dataframe into a single dataframe based on 'Store'
    df1 = pd.merge(df_sales, df_store, on="Store", how="inner")

    # combine store and feature dataframe into a single dataframe based on 'Store'
    df2 = pd.merge(df_store, df_feature, on="Store", how="inner")

    # create unique column (diff) for the combination of store and date
    df1["diff"] = df1["Store"].astype(str) + "-" + df1["Date"]
    df2["diff"] = df2["Store"].astype(str) + "-" + df2["Date"]

    # df1 covers 2010 to 2012-Oct, df2 covers 2010 to 2013.
    # Split df2 into (2010 to 2012-Oct) and (2012-Nov to 2013).
    df1_list = df1["diff"].to_list()

    df2_inlist = df2[df2["diff"].isin(df1_list)].reset_index(drop=True)
    df2_notinlist = df2[~df2["diff"].isin(df1_list)].reset_index(drop=True)

    print("df2 / df2_inlist / df2_notinlist shapes:",
          df2.shape, df2_inlist.shape, df2_notinlist.shape)
    print("Unique 'diff' counts:",
          df2["diff"].nunique(), df2_inlist["diff"].nunique(), df2_notinlist["diff"].nunique())

    # merge df1 with df2 (2010 to 2012-Oct) based on 'diff'
    df3 = pd.merge(df1, df2_inlist, on="diff", how="inner")

    # drop duplicate columns created by the merge and rename the kept ones
    df3.drop(columns=["Store_y", "Type_y", "Size_y", "Date_y", "IsHoliday_y"], inplace=True)
    df3.rename(
        columns={
            "Store_x": "Store",
            "Date_x": "Date",
            "IsHoliday_x": "IsHoliday",
            "Type_x": "Type",
            "Size_x": "Size",
        },
        inplace=True,
    )

    # df2_notinlist (2012-Nov to 2013) is missing 'Dept', so build it out
    # by cross-joining every (Store, Dept) combination present in df_sales.
    s = df_sales[["Store", "Dept"]].drop_duplicates(subset=["Store", "Dept"]).reset_index(drop=True)
    df4 = pd.merge(s, df2_notinlist, on="Store", how="outer")

    # concatenate both halves into a single dataframe (2010 to 2013)
    df5 = pd.concat([df3, df4]).reset_index(drop=True)

    return df5


def clean_and_engineer(df5):
    """Type conversion, feature engineering, and null/outlier handling."""
    # datatype conversion
    df5["Date"] = df5["Date"].apply(lambda x: x.replace("/", "-"))
    df5["Date"] = pd.to_datetime(df5["Date"], format="%d-%m-%Y", errors="coerce")

    # encode categorical features into numerical
    df5["IsHoliday"] = df5["IsHoliday"].map({True: 1, False: 0})
    df5["Type"] = df5["Type"].map({"A": 1, "B": 2, "C": 3})

    # drop 'diff' column and sort ascending by date/store/dept
    df5.drop(columns=["diff"], inplace=True)
    df5 = df5.sort_values(by=["Date", "Store", "Dept"]).reset_index(drop=True)

    # split Date into Day / Month / Year, then drop Date
    df5["Day"] = df5["Date"].dt.day
    df5["Month"] = df5["Date"].dt.month
    df5["Year"] = df5["Date"].dt.year
    df5.drop(columns=["Date"], inplace=True)

    # rearrange column order
    df5 = df5[
        [
            "Day", "Month", "Year", "Store", "Dept", "Type", "Weekly_Sales", "Size",
            "IsHoliday", "Temperature", "Fuel_Price", "MarkDown1", "MarkDown2",
            "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment",
        ]
    ]

    print("dtypes:\n", df5.dtypes, "\n")
    print("null counts:\n", df5.isnull().sum(), "\n")
    print(df5.describe().T, "\n")

    # negative Weekly_Sales values are invalid -> convert to NaN
    print("negative Weekly_Sales rows:", (df5["Weekly_Sales"] <= 0).sum())
    df5["Weekly_Sales"] = df5["Weekly_Sales"].apply(lambda x: np.nan if x <= 0 else x)

    print(df5.describe().T, "\n")
    print("null counts after cleanup:\n", df5.isnull().sum(), "\n")

    # MarkDown1-5 have a huge number of nulls compared to other features.
    # Create a flag: 1 if any MarkDown value is present, 0 otherwise.
    df5["markdown"] = df5[["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]].notnull().any(axis=1).astype(int)

    print("Weekly_Sales mean by markdown flag:\n", df5.groupby("markdown")["Weekly_Sales"].mean(), "\n")

    plot_correlation(df5.drop(columns=["Day", "Month", "Year", "Store", "Dept", "markdown"]).dropna().corr(),
                      "df5 Correlation Heatmap", figsize=(15, 5))

    # rebuild the unique 'diff' key used to re-merge predicted columns later
    df5["diff"] = (
        df5["Day"].astype(str) + df5["Month"].astype(str) + df5["Year"].astype(str)
        + "-" + df5["Store"].astype(str) + "-" + df5["Dept"].astype(str)
    )

    print("unique value counts per feature:\n", df5.nunique(), "\n")

    return df5


def plot_correlation(corr_df, title, figsize=(15, 5)):
    plt.figure(figsize=figsize)
    sns.heatmap(corr_df, annot=True, fmt=".2f")
    plt.title(title)
    plt.savefig("plots/correlation.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------------
# Shared ML helpers
# ---------------------------------------------------------------------------
def algorithm_train_test_accuracy(x_train, x_test, y_train, y_test, algorithm):
    """Fit `algorithm` and return train/test R2 scores."""
    model = algorithm().fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    accuracy = {
        "algorithm": algorithm.__name__,
        "R2_train": r2_train,
        "R2_test": r2_test,
    }
    return accuracy


def ml_regression(df, null_features, label):
    """
    Train several regressors to compare performance, then use RandomForest
    (best performer) to impute the missing values of `label`.

    - df: input dataframe (must contain a 'diff' key column)
    - null_features: other columns with nulls to drop before training
    - label: target column whose missing values will be predicted
    """
    df = df.drop(columns=null_features)

    df_null = df[df[label].isnull()].reset_index(drop=True)
    df_notnull = df[df[label].notnull()].reset_index(drop=True)

    x = df_notnull.drop(columns=[label, "diff"])
    y = df_notnull[label]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # compare algorithms
    if COMPARE_MODELS:
        print(algorithm_train_test_accuracy(
            x_train, x_test, y_train, y_test, DecisionTreeRegressor))
        print(algorithm_train_test_accuracy(
            x_train, x_test, y_train, y_test, ExtraTreesRegressor))
        print(algorithm_train_test_accuracy(
            x_train, x_test, y_train, y_test, RandomForestRegressor))
        print(algorithm_train_test_accuracy(
            x_train, x_test, y_train, y_test, AdaBoostRegressor))
        print(algorithm_train_test_accuracy(
            x_train, x_test, y_train, y_test, GradientBoostingRegressor))
        print(algorithm_train_test_accuracy(
            x_train, x_test, y_train, y_test, XGBRegressor))

    # RandomForest performs best -> use it to impute
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ).fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {
        "R2": r2,
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "Root Mean Squared Error": rmse,
    }
    print(metrics)

    y_pred_null = model.predict(df_null.drop(columns=[label, "diff"]))
    df_null[label] = pd.DataFrame(y_pred_null)

    df_final = pd.concat([df_null, df_notnull], axis=0, ignore_index=True)
    return df_final

'''
# ---------------------------------------------------------------------------
# 2. Method 1 - with MarkDowns
# ---------------------------------------------------------------------------
def method1_with_markdown(df5):
    print("\n===== Method 1: WITH MarkDowns =====\n")

    df_m1 = df5.copy()
    df_m1.drop(columns=["markdown"], inplace=True)

    # predict each MarkDown1-5 column separately
    df_markdown1 = ml_regression(
        df_m1, ["Weekly_Sales", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment"], "MarkDown1"
    )
    df_markdown2 = ml_regression(
        df_m1, ["Weekly_Sales", "MarkDown1", "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment"], "MarkDown2"
    )
    df_markdown3 = ml_regression(
        df_m1, ["Weekly_Sales", "MarkDown1", "MarkDown2", "MarkDown4", "MarkDown5", "CPI", "Unemployment"], "MarkDown3"
    )
    df_markdown4 = ml_regression(
        df_m1, ["Weekly_Sales", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown5", "CPI", "Unemployment"], "MarkDown4"
    )
    df_markdown5 = ml_regression(
        df_m1, ["Weekly_Sales", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "CPI", "Unemployment"], "MarkDown5"
    )

    # swap in the newly predicted MarkDown columns
    df_m1 = df_m1.drop(columns=["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"])
    df_m1 = pd.merge(df_m1, df_markdown1[["MarkDown1", "diff"]], on="diff", how="inner")
    df_m1 = pd.merge(df_m1, df_markdown2[["MarkDown2", "diff"]], on="diff", how="inner")
    df_m1 = pd.merge(df_m1, df_markdown3[["MarkDown3", "diff"]], on="diff", how="inner")
    df_m1 = pd.merge(df_m1, df_markdown4[["MarkDown4", "diff"]], on="diff", how="inner")
    df_m1 = pd.merge(df_m1, df_markdown5[["MarkDown5", "diff"]], on="diff", how="inner")

    # predict CPI
    df_cpi = ml_regression(df_m1, ["Weekly_Sales", "Unemployment"], "CPI")
    df_m1 = df_m1.drop(columns=["CPI"])
    df_m1 = pd.merge(df_m1, df_cpi[["CPI", "diff"]], on="diff", how="inner")

    # predict Unemployment
    df_unemployment = ml_regression(df_m1, ["Weekly_Sales"], "Unemployment")
    df_m1 = df_m1.drop(columns=["Unemployment"])
    df_m1 = pd.merge(df_m1, df_unemployment[["Unemployment", "diff"]], on="diff", how="inner")

    # predict Weekly_Sales (final target)
    df_weekly_sales = ml_regression(df_m1, [], "Weekly_Sales")
    df_m1_weekly_sales = df_weekly_sales.copy()

    print(df_m1_weekly_sales.describe().T, "\n")

    plot_correlation(
        df_m1_weekly_sales.drop(columns=["Day", "Month", "Year", "Store", "Dept"]).dropna().corr(),
        "df_m1 Correlation Heatmap",
        figsize=(15, 5),
    )

    # final train/test split + RandomForest fit used for the saved model
    df_null = df_m1[df_m1["Weekly_Sales"].isnull()].reset_index(drop=True)
    df_notnull = df_m1[df_m1["Weekly_Sales"].notnull()].reset_index(drop=True)

    x = df_notnull.drop(columns=["Weekly_Sales", "diff"])
    y = df_notnull["Weekly_Sales"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ).fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {
        "R2": r2,
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "Root Mean Squared Error": rmse,
    }
    print("Method 1 final metrics:", metrics)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "model1_markdown.pkl"), "wb") as f:
        pickle.dump(model, f)

    return df_m1_weekly_sales
'''

# ---------------------------------------------------------------------------
# 3. Method 2 - without MarkDowns
# ---------------------------------------------------------------------------
def method2_without_markdown(df5):
    print("\n===== Method 2: WITHOUT MarkDowns =====\n")

    df_m2 = df5.copy()
    df_m2.drop(columns=["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "markdown"], inplace=True)

    # predict CPI
    df_m2_cpi = ml_regression(df_m2, ["Weekly_Sales", "Unemployment"], "CPI")
    df_m2 = df_m2.drop(columns=["CPI"])
    df_m2 = pd.merge(df_m2, df_m2_cpi[["CPI", "diff"]], on="diff", how="inner")

    # predict Unemployment
    df_m2_unemployment = ml_regression(df_m2, ["Weekly_Sales"], "Unemployment")
    df_m2 = df_m2.drop(columns=["Unemployment"])
    df_m2 = pd.merge(df_m2, df_m2_unemployment[["Unemployment", "diff"]], on="diff", how="inner")

    # predict Weekly_Sales (final target)
    df_m2_weekly_sales = ml_regression(df_m2, [], "Weekly_Sales")

    print(df_m2_weekly_sales.describe().T, "\n")

    '''plot_correlation(
        df_m2_weekly_sales.drop(columns=["Day", "Month", "Year", "Store", "Dept"]).dropna().corr(),
        "df_m2 Correlation Heatmap",
        figsize=(10, 4),
    )'''
    numeric_df = df_m2_weekly_sales.select_dtypes(include="number")

    plot_correlation(
    numeric_df.corr(),
    "df_m2 Correlation Heatmap",
    figsize=(10,4)
)

    # final train/test split + RandomForest fit used for the saved model
    df_null = df_m2[df_m2["Weekly_Sales"].isnull()].reset_index(drop=True)
    df_notnull = df_m2[df_m2["Weekly_Sales"].notnull()].reset_index(drop=True)

    x = df_notnull.drop(columns=["Weekly_Sales", "diff"])
    y = df_notnull["Weekly_Sales"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ).fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {
        "R2": r2,
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "Root Mean Squared Error": rmse,
    }
    print("Method 2 final metrics:", metrics)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "model2.pkl"), "wb") as f:
        pickle.dump(model, f)

    return df_m2_weekly_sales


# ---------------------------------------------------------------------------
# 4. Inference sample - reload pickled models and predict
# ---------------------------------------------------------------------------
def run_inference_samples():
    print("\n===== Inference samples =====\n")

    model1_path = os.path.join(MODEL_DIR, "model1_markdown.pkl")
    model2_path = os.path.join(MODEL_DIR, "model2.pkl")

    if os.path.exists(model1_path):
        with open(model1_path, "rb") as f1:
            pred_model = pickle.load(f1)
        # columns: Day, Month, Year, Store, Dept, Type, Size, IsHoliday, Temperature,
        #          Fuel_Price, MarkDown1-5, CPI, Unemployment
        sample1 = np.array([[5, 2, 2010, 1, 1, 1, 151315, 0, 42.31, 2.6,
                              7046.9, 6618.9, 166.9, 16055.8, 4671.9, 211.1, 8.1]])
        y_pred1 = pred_model.predict(sample1)
        print("Model 1 (with MarkDowns) sample prediction:", y_pred1[0])

    if os.path.exists(model2_path):
        with open(model2_path, "rb") as f1:
            pred_model = pickle.load(f1)
        # columns: Day, Month, Year, Store, Dept, Type, Size, IsHoliday, Temperature,
        #          Fuel_Price, CPI, Unemployment
        sample2 = np.array([[5, 2, 2010, 1, 1, 1, 151315, 0, 42.31, 2.6, 211.1, 8.1]])
        y_pred2 = pred_model.predict(sample2)
        print("Model 2 (without MarkDowns) sample prediction:", y_pred2[0])


def load_sql_reference_data():
    df_sql = pd.read_csv(SQL_URL)
    print(df_sql.head(2))
    return df_sql


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df_store, df_sales, df_feature = load_raw_data()
    df5_raw = merge_datasets(df_store, df_sales, df_feature)
    df5 = clean_and_engineer(df5_raw)

    
    df_m2_weekly_sales = method2_without_markdown(df5)

    run_inference_samples()
    load_sql_reference_data()

    print("\nDone. Trained models saved under:", os.path.abspath(MODEL_DIR))


if __name__ == "__main__":
    main()