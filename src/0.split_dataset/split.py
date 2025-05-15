import pandas as pd
import os
from sklearn.model_selection import train_test_split
import sys
import numpy as np
class Split:
    def __init__(self):
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        self.csv_dir = os.path.join(parent_dir, 'datasets/')
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
        self.filename = "dataset_train.csv"
        self.filepath = os.path.join(self.csv_dir, self.filename)

    def split_training_data(self, df):
        # Stratify ensure the proportions of the classes are maintained in both sets   
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["Hogwarts House"]
        )
        return train_df, val_df
    
    def run(self):
      try:
          df = pd.read_csv(self.filepath)
      except FileNotFoundError:
          print(f"Error: File not found at {self.filepath}")
          sys.exit(1)
      except Exception as e:
          print(f"Error reading CSV file: {e}")
          sys.exit(1)

      df["Hogwarts House"] = df["Hogwarts House"].map({"Slytherin": 0, "Gryffindor": 1, "Ravenclaw": 2, "Hufflepuff": 3})
      df['Hogwarts House'] = pd.to_numeric(df['Hogwarts House'], errors='coerce')
      df = df.select_dtypes(include=np.number)
      df_numeric = df.copy()
      df_numeric = df_numeric.dropna().reset_index(drop=True)

      if df_numeric.empty:
          print("No numeric data available after dropping NaN values.")
          sys.exit(1)
      df_train, df_test = self.split_training_data(df_numeric)
      
      df_train.to_csv(os.path.join(self.csv_dir, "splitted_dataset_train.csv"), index=False)
      df_test.to_csv(os.path.join(self.csv_dir, "splitted_dataset_test.csv"), index=False)

def main():
    a = Split()
    a.run()
if __name__ == "__main__":
    main()