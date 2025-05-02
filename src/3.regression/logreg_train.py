import pandas as pd
import os
import sys
from sklearn import preprocessing
import numpy as np
from scipy.stats import pearsonr
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

class LogregTrain:
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

  def select_features(self, df, target_column=None, correlation_threshold=0.5, max_features=10):
      """
      Select the most relevant features based on correlation or variance.
      
      Parameters:
      - df: The dataframe containing the features
      - target_column: If provided, select features based on correlation with this target
      - correlation_threshold: Minimum absolute correlation to keep a feature
      - max_features: Maximum number of features to select
      
      Returns:
      - List of selected feature names
      """
      print("Analyzing feature importance...")

      df = self.normalize_df(df)
      
      if target_column is not None and target_column in df.columns:
          # Select features based on correlation with target
          correlations = {}
          target = df[target_column]
          
          for column in df.columns:
              if column != target_column:
                  corr, _ = pearsonr(df[column], target)
                  correlations[column] = abs(corr)  # Use absolute value of correlation
          
          # Sort features by correlation and select top ones
          sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
          selected_features = [col for col, corr in sorted_features 
                            if corr >= correlation_threshold][:max_features]
          
          print(f"Top features correlated with {target_column}:")
          for feature, corr in sorted_features[:max_features]:
              print(f"  {feature}: {corr:.4f}")
          
      else:
          # If no target, select features with highest variance
          variances = df.var().sort_values(ascending=False)
          selected_features = variances.index[:max_features].tolist()
          
          print("Top features by variance:")
          for feature in selected_features:
              print(f"  {feature}: {variances[feature]:.4f}")
      
      # Add target column to selected features if it exists
      if target_column is not None and target_column not in selected_features:
          selected_features.append(target_column)
          
      return selected_features
  
  def plot_feature_importance(self, selected_features, model):
    plt.figure
    plt.bar(selected_features[:-1], np.abs(model.coef_.mean(axis=0)))
    plt.xticks(rotation=45)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

  def save_csv_weights(self, weights, X_columns):
    Hogwarts_House = ["Slytherin", "Gryffindor", "Ravenclaw", "Hufflepuff"]
    weights_df = pd.DataFrame(weights, columns=X_columns)
    weights_df['Hogwarts House'] = Hogwarts_House
    weights_df.to_csv(os.path.join(self.csv_dir, 'weights.csv'), index=False)
    print(f"Weights saved to {os.path.join(self.csv_dir, 'weights.csv')}")
  
  def train_logistic_regression(self, X_train, y_train):
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)
    return model
  
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
    df_raw['Hogwarts House'] = pd.to_numeric(df_raw['Hogwarts House'], errors='coerce')
    df_raw = df_raw.select_dtypes(include=np.number)
    df_numeric = df_raw.copy()
    df_numeric = df_numeric.dropna().reset_index(drop=True)

    if df_numeric.empty:
        print("No numeric data available after dropping NaN values.")
    selected_features = self.select_features(df_numeric, target_column='Hogwarts House', correlation_threshold=0.4, max_features=10)
    print(f"Selected features: {selected_features}")
    df_numeric = df_numeric[selected_features]
    X_train = df_numeric.drop(columns=['Hogwarts House'])
    X_columns = X_train.columns
    y_train = df_numeric['Hogwarts House']

    model = self.train_logistic_regression(X_train, y_train)

    print(f"Weights (Coefficients): {model.coef_}\nBias (Intercept): {model.intercept_}")

    self.plot_feature_importance(selected_features, model)

    self.save_csv_weights(model.coef_, X_columns)
    

    


def main():
  a = LogregTrain()
  a.run()

if __name__ == "__main__":
  main()