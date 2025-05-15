import pandas as pd
import os
import sys
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix

class PairPlot:
  def __init__(self):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    self.csv_dir = os.path.join(parent_dir, 'datasets/')
    if not os.path.exists(self.csv_dir):
        os.makedirs(self.csv_dir)
    self.filename = sys.argv[1] if len(sys.argv) == 2 else "dataset_train.csv"
    self.filepath = os.path.join(self.csv_dir, self.filename)
  
  def normalize_df(self, df):
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(x_scaled, columns=df.columns)
    return df_normalized

  def select_relevant_features(self, df, correlation_threshold=0.8):
      """
      Select relevant features based on correlation and domain knowledge.

      Parameters:
      - df: DataFrame containing the normalized data.
      - correlation_threshold: Threshold for considering features as highly correlated.

      Returns:
      - List of selected feature names.
      """
      correlation_matrix = df.corr()
      selected_features = set()

      # Iterate over the correlation matrix to find highly correlated pairs
      for i in range(len(correlation_matrix.columns)):
          for j in range(i):
              if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                  feature1 = correlation_matrix.columns[i]
                  feature2 = correlation_matrix.columns[j]
                  # Select one feature from the pair
                  if feature1 not in selected_features and feature2 not in selected_features:
                      selected_features.add(feature1)
                      print(f"Selected {feature1} over {feature2} due to high correlation.")

      # Add remaining features that are not highly correlated with any other feature
      for feature in correlation_matrix.columns:
          if feature not in selected_features:
              selected_features.add(feature)

      return list(selected_features)

  def run(self):
    try:
        df_raw = pd.read_csv(self.filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {self.filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    df_raw["Hogwarts House"] = df_raw["Hogwarts House"].map({"Slytherin": 0, "Gryffindor": 1, "Ravenclaw": 2, "Hufflepuff": 3})
    df_numeric = df_raw.select_dtypes(include=np.number)
    df_numeric = df_numeric.dropna().reset_index(drop=True)
    if df_numeric.empty:
        print("No numeric data available after dropping NaN values.")
        sys.exit(1)

    df_normalized = self.normalize_df(df_numeric)
    if 'Index' in df_normalized.columns:
        df_normalized.drop(columns=['Index'], inplace=True)
    scatter_matrix(df_normalized, alpha=0.2, figsize=(40, 20), diagonal='kde', range_padding=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    selected_features = self.select_relevant_features(df_normalized)
    print("Selected features based on correlation and domain knowledge:")
    print(selected_features)


def main():
  a = PairPlot()
  a.run()


if __name__ == "__main__":
  main()