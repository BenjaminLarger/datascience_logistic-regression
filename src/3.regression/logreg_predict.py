import pandas as pd
import os
import sys
from sklearn import preprocessing
import numpy as np
from sklearn import preprocessing


class LogregPredict:
  def __init__(self):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    self.csv_dir = os.path.join(parent_dir, 'datasets/')
    if not os.path.exists(self.csv_dir):
        os.makedirs(self.csv_dir)
    self.filename_test = sys.argv[1] if len(sys.argv) == 3 else "dataset_test.csv"
    self.filename_weights = sys.argv[2] if len(sys.argv) == 3 else "weights2.csv"
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
  
  def make_prediction(self, X_test_scaled, df_weights):
    print("--------- MAKE PREDICTION ---------")
    predictions = []
    houses = ['Slytherin', 'Gryffindor', 'Ravenclaw', 'Hufflepuff']
    houses = df_weights.index.tolist()
    print(f"houses: {houses}")
    features = df_weights.drop('Bias', axis=1).drop('Hogwarts House', axis=1).columns.tolist()

    for index, student in enumerate(X_test_scaled):
        house_scores = {}

        for house in houses:
            # Get the weights for the current house
            print(f"house: {house}, df_weights.loc[house, 'Bias']: {df_weights.loc[house, 'Bias']}")
            bias = df_weights.loc[house, 'Bias']
            coefficients = df_weights.loc[house, features].values

            # Calculate the score for the current house
            score = bias + np.dot(student, coefficients)

            # Convert the score to a probability using sigmoid
            probability = 1 / (1 + np.exp(-score))
            house_scores[house] = probability
        predicted_house = max(house_scores, key=house_scores.get)
        predictions.append(predicted_house)
    print(f"Predicted house: {predictions}, len(predictions): {len(predictions)}")
    return predictions

  def run(self):
    try:
        df_test = pd.read_csv(self.filepath_test)
        df_weights = pd.read_csv(self.filepath_weights)
    except FileNotFoundError:
        print(f"Error: File not found at {self.filepath_test} or {self.filepath_weights}.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    print(f"df_test head: {df_test.head()}")
    df_test.drop('Hogwarts House', axis=1, inplace=True)
    df_test.drop('Index', axis=1, inplace=True)
    df_test.dropna(inplace=True)
    # df_test = df_test.reset_index(drop=True)
    df_test = df_test.select_dtypes(include=np.number).drop('Index', axis=1, errors='ignore')
    print(f"df_test head:\n {df_test.head()}")

    if df_test.empty:
        print("Error: DataFrame is empty after dropping non-numeric columns and NaN values.")
        sys.exit(1)
    # Select only the columns that are in df_weights
    print(f"df_weights columns: {df_weights.columns}")
    X_columns = df_weights.drop('Bias', axis=1).drop('Hogwarts House', axis=1).columns.tolist()
    df_test = df_test[X_columns]
    print(f"df_test head:\n {df_test.head()}")

    scaler = preprocessing.StandardScaler()
    X_test_scaled = scaler.fit_transform(df_test)
    # Only 310 rows. Why ?
    print(f"X_test_scaled head:\n {X_test_scaled[:5]}\n X_test_scaled shape: {X_test_scaled.shape}")
    predictions = self.make_prediction(X_test_scaled, df_weights)
    results = pd.DataFrame({
        'Index': df_test.index,
        'Hogwarts House': predictions
    })
    results['Hogwarts House'] = results['Hogwarts House'].map({
        0: 'Slytherin',
        1: 'Gryffindor',
        2: 'Ravenclaw',
        3: 'Hufflepuff'
    })
    results.to_csv(os.path.join(self.csv_dir, 'predictions.csv'), index=False)


def main():
  a = LogregPredict()
  a.run()

if __name__ == "__main__":
  main()