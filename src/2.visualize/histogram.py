import pandas as pd
import os
import sys
from sklearn import preprocessing
import matplotlib.pyplot as plt

class Histogram:
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
  
  def plot_histograms(self, df):
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    plt.figure(figsize=(20, 12))
    
    for i, house in enumerate(houses, 1):
        plt.subplot(2, 2, i)
        house_data = df[df['Hogwarts House'] == house].drop('Hogwarts House', axis=1)
        plt.hist(house_data.values, bins=30, alpha=0.7, label=house, stacked=True)
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title(f'Score distribution for {house}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
  
  def run(self):
    df = pd.read_csv(self.filepath, sep=',')
    df = df.dropna()
    # df = df.drop('Index', axis=1)
    df_numeric = df.dropna()
    df_numeric.reset_index(drop=True, inplace=True)
    hogwarts_column = df_numeric['Hogwarts House']
    df_numeric = df_numeric.select_dtypes(include=['number'])
    print(f"df_numeric is {df_numeric.columns}")
    df_normalized = self.normalize_df(df_numeric)
    score_column = df_normalized.mean(axis=1)
    df_normalized['Hogwarts House'] = hogwarts_column
    print(f"df_normalized columns {df_normalized.columns}")
    df_normalized['score'] = score_column
    df_normalized = df_normalized[['Hogwarts House', 'score']]
    self.plot_histograms(df_normalized)

    print(df_normalized.head())
def main():
  a = Histogram()
  a.run()

if __name__ == "__main__":
  main()