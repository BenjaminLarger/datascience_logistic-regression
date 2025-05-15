import pandas as pd
import os
import sys
from sklearn import preprocessing
import numpy as np
from scipy.stats import pearsonr
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

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
  
  def save_csv_weights(self, weights, bias, X_columns):
    print(f"type of weights: {type(weights)}")
    print(f"type of bias: {type(bias)}")
    print(f"Weights = {weights}")
    # Convert bias to scalars
    houses = ['Slytherin', 'Gryffindor', 'Ravenclaw', 'Hufflepuff']
    bias = [float(b[0]) for b in bias]  # Extract first element from each array
    
    X_columns = list(X_columns)  # Convert to list before appending
    X_columns.append('Bias')
    X_columns.append('Hogwarts House')
    weights_df = pd.DataFrame(weights, index=X_columns).T
    weights_df['Bias'] = bias
    weights_df['Hogwarts House'] = houses
    weights_df.index = houses
    print(f"Weights_df = {weights_df}")
    weights_df.to_csv(os.path.join(self.csv_dir, 'weights2.csv'), index=False)

  def train_one_vs_all(self, X_train, unique_houses, houses):
    all_weights = {}
    models = {}

    for house in unique_houses:
         # Create binary labels for the current house
         y_binary = (houses == house).astype(int)
         
         # Train the model
         logreg = LogisticRegression()
         logreg.fit(X_train, y_binary)

         # Store the model
         models[house] = logreg

         # Extract and store the weights (coefficients and intercept)
         weights = np.append(logreg.intercept_, logreg.coef_[0])
         all_weights[house] = weights
         print(f"Model weights for house {house}: {weights}")
    return all_weights, models

  def plot_feature_importance(self, df_numeric):
    """
    Plot each feature separately to show the mean value for each house.
    """
    print(f"df_numeric head: {df_numeric.head()}")
    Slytherin_data = df_numeric[df_numeric['Hogwarts House'] == 0]
    Gryffindor_data = df_numeric[df_numeric['Hogwarts House'] == 1]
    Ravenclaw_data = df_numeric[df_numeric['Hogwarts House'] == 2]
    Hufflepuff_data = df_numeric[df_numeric['Hogwarts House'] == 3]
    means = {
        'Slytherin': Slytherin_data.mean(),
        'Gryffindor': Gryffindor_data.mean(),
        'Ravenclaw': Ravenclaw_data.mean(),
        'Hufflepuff': Hufflepuff_data.mean()
    }
    means_df = pd.DataFrame(means)
    
    # Get features (excluding Hogwarts House)
    features = [col for col in df_numeric.columns if col != 'Hogwarts House']
    
    # Create a plot for each feature
    fig, axes = plt.subplots(len(features), 1, figsize=(10, 4*len(features)))
    
    for i, feature in enumerate(features):
        feature_data = means_df.loc[feature]
        feature_data.plot(kind='bar', ax=axes[i], color=['green', 'red', 'blue', 'yellow'])
        axes[i].set_title(f"Mean {feature} by House")
        axes[i].set_ylabel("Value")
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
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

    df_raw["Hogwarts House"] = df_raw["Hogwarts House"].map({"Slytherin": 0, "Gryffindor": 1, "Ravenclaw": 2, "Hufflepuff": 3})
    df_raw['Hogwarts House'] = pd.to_numeric(df_raw['Hogwarts House'], errors='coerce')
    df_raw = df_raw.select_dtypes(include=np.number)
    df_numeric = df_raw.copy()
    df_numeric = df_numeric.dropna().reset_index(drop=True)

    if df_numeric.empty:
        print("No numeric data available after dropping NaN values.")
        sys.exit(1)
    selected_features = self.select_features(df_numeric, target_column='Hogwarts House', correlation_threshold=0.4, max_features=10)
    print(f"Selected features: {selected_features}")
    df_numeric = df_numeric[selected_features]
    self.plot_feature_importance(df_numeric)
    X_columns = df_numeric.drop(columns=['Hogwarts House']).columns
    X = df_numeric.drop(columns=['Hogwarts House']).values
    houses = df_numeric['Hogwarts House'].values

    # Standardize the features
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Get the unique houses
    unique_houses = np.unique(houses)

    weights, model = self.train_one_vs_all(X_scaled, unique_houses, houses)

    weights_df = pd.DataFrame(weights).T
    weights_df.columns = ['Bias'] + list(X_columns)
    houses = ['Slytherin', 'Gryffindor', 'Ravenclaw', 'Hufflepuff']
    # weights_df['Hogwarts House'] = unique_houses
    weights_df['Hogwarts House'] = houses
    weights_df.index = unique_houses
    weights_df.to_csv(os.path.join(self.csv_dir, 'weights2.csv'), index=False)
    # print(f"Weights (Coefficients): {weights}")
    # print(f"X columns: {X_columns}")
    # self.save_csv_weights(weights, X_columns)

def main():
  a = LogregTrain()
  a.run()

if __name__ == "__main__":
  main()