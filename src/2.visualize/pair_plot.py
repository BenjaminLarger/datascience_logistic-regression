import pandas as pd
import os
import sys
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix

class PairPlot:
  def __init__(self):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    self.csv_dir = os.path.join(parent_dir, 'datasets/')
    if not os.path.exists(self.csv_dir):
        os.makedirs(self.csv_dir)
    self.filename = sys.argv[1] if len(sys.argv) > 2 else "dataset_train.csv"
    self.filepath = os.path.join(self.csv_dir, self.filename)
  
  def normalize_df(self, df):
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(x_scaled, columns=df.columns)
    return df_normalized

  def find_most_similar_features(self, correlation_matrix):
    corr_unstacked = correlation_matrix.unstack()
    sorted_correlation = corr_unstacked.abs().sort_values(ascending=False)
    sorted_correlation = sorted_correlation[sorted_correlation != 1.0]
    most_similar_pair = sorted_correlation.index[1]
    return most_similar_pair

  def plot_scatter(self, df, feature1, feature2):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=feature1, y=feature2)
        plt.title(f'Scatter plot of {feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

  def run(self):
    try:
        df_raw = pd.read_csv(self.filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {self.filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    df_numeric = df_raw.select_dtypes(include=np.number)
    df_numeric = df_numeric.dropna().reset_index(drop=True)
    if df_numeric.empty:
        print("No numeric data available after dropping NaN values.")

    df_normalized = self.normalize_df(df_numeric)
    if 'Index' in df_normalized.columns:
        df_normalized.drop(columns=['Index'], inplace=True)
    scatter_matrix(df_normalized, alpha=0.2, figsize=(40, 20), diagonal='kde', range_padding=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
  a = PairPlot()
  a.run()


if __name__ == "__main__":
  main()