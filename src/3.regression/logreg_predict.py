import pandas as pd
import os
import sys
from sklearn import preprocessing
import numpy as np
from scipy.stats import pearsonr
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt


class LogregPredict:
  def __init__(self):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    self.csv_dir = os.path.join(parent_dir, 'datasets/')
    if not os.path.exists(self.csv_dir):
        os.makedirs(self.csv_dir)
    self.filename_test = sys.argv[1] if len(sys.argv) == 3 else "dataset_test.csv"
    self.filename_weights = sys.argv[2] if len(sys.argv) == 3 else "weights.csv"
    self.filepath_test = os.path.join(self.csv_dir, self.filename_test)
    self.filepath_weights = os.path.join(self.csv_dir, self.filename_weights)

  
  def normalize_df(self, df):
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(x_scaled, columns=df.columns)
    return df_normalized

  def save_csv_prediction(self, weights, X_columns):
    pass

  def predict(self, X, weights):
    y_pred = np.dot(X, weights)
    return y_pred
  
  def run(self):
    try:
        df_test = pd.read_csv(self.filepath_test)
        df_weights = pd.read_csv(self.filepath_weights)
    except FileNotFoundError:
        print(f"Error: File not found at {self.filepath_test}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    print(f"df_test head: {df_test.head()}")
    df_test = df_test.select_dtypes(include=np.number).drop('Index', axis=1, errors='ignore')
    print(f"df_test head: {df_test.head()}")

    if df_test.empty:
        print("Error: DataFrame is empty after dropping non-numeric columns and NaN values.")
        sys.exit(1)
    # Select only the columns that are in df_weights
    print(f"df_weights columns: {df_weights.columns}")
    df_test = df_test[df_weights.columns]

    scaler = preprocessing.StandardScaler()
    X_test = scaler.fit_transform(df_test)
    print(f"X_test.mean: {X_test.mean(axis=0)}, X_test.std: {X_test.std(axis=0)}")
    print(f"X_test head: {X_test[:5]}")

    weights = df_weights.values

    y_pred = self.predict(X_test, weights)
    print(f"y_pred: {y_pred[:5]}")



    
def main():
  a = LogregPredict()
  a.run()

if __name__ == "__main__":
  main()