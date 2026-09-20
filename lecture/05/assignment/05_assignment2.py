"""第 5 週 Assignment #2 - 晶圓驗收分類器資料載入骨架。

目前先完成題目資料的載入與基本驗證。Logistic Regression、特徵工程、
標準化、訓練與視覺化會在後續作業實作階段加入。

不使用 scikit-learn。
"""

from pathlib import Path

import pandas as pd


TRAIN_COLUMNS = ["x1", "x2", "y"]
TEST_COLUMNS = ["Wafer ID", "x1", "x2"]


def validate_assignment_data(train_df, test_df):
    """確認兩份 CSV 符合作業指定的欄位與基本資料條件。"""
    if list(train_df.columns) != TRAIN_COLUMNS:
        raise ValueError(
            f"wat_train.csv columns must be {TRAIN_COLUMNS}, "
            f"got {list(train_df.columns)}"
        )
    if list(test_df.columns) != TEST_COLUMNS:
        raise ValueError(
            f"wat_test.csv columns must be {TEST_COLUMNS}, "
            f"got {list(test_df.columns)}"
        )
    if train_df.isna().any().any() or test_df.isna().any().any():
        raise ValueError("Assignment data must not contain missing values")
    if not set(train_df["y"]).issubset({0, 1}):
        raise ValueError("wat_train.csv labels must be 0 or 1")


def load_assignment_data(data_dir=None):
    """從指定目錄讀取並驗證訓練集與待驗收資料。"""
    base_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent
    train_df = pd.read_csv(base_dir / "wat_train.csv")
    test_df = pd.read_csv(base_dir / "wat_test.csv")
    validate_assignment_data(train_df, test_df)
    return train_df, test_df


if __name__ == "__main__":
    training_data, acceptance_data = load_assignment_data()
    print(f"Training data: {training_data.shape[0]} rows, columns={list(training_data.columns)}")
    print(f"Test data: {acceptance_data.shape[0]} rows, columns={list(acceptance_data.columns)}")
    print("Assignment scaffold ready; model implementation is the next step.")
