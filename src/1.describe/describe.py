import pandas as pd
import os
import sys
from math import sqrt
class Describe:
  def __init__(self):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    self.csv_dir = os.path.join(parent_dir, 'datasets/')
    if not os.path.exists(self.csv_dir):
        os.makedirs(self.csv_dir)
    self.filename = sys.argv[1] if len(sys.argv) == 2 else "dataset_train.csv"
    self.filepath = os.path.join(self.csv_dir, self.filename)

  def get_mean(self, column):
    sum = 0
    for i in range(len(column)):
        sum += column[i]
    return sum / len(column)
  
  def get_std(self, column):
    mean = self.get_mean(column)
    sum = 0
    for i in range(len(column)):
        sum += (column[i] - mean) ** 2
    return sqrt(sum / len(column))
  
  def get_min(self, column):
    min = column[0]
    for i in range(len(column)):
        if column[i] < min:
            min = column[i]
    return min
  
  def get_max(self, column):
    max = column[0]
    for i in range(len(column)):
        if column[i] > max:
            max = column[i]
    return max
  
  def quartile_one(self, column):
    sorted_column = sorted(column)
    k = len(sorted_column) // 4
    if len(sorted_column) % 4 < 2:
        return sorted_column[k-1] + sorted_column[k] / 2
    else:
        return sorted_column[k - 1]
    
  def quartile_two(self, column):
    sorted_column = sorted(column)
    k = len(sorted_column) // 2
    if len(sorted_column) % 2 < 2:
        return sorted_column[k-1] + sorted_column[k] / 2
    else:
        return sorted_column[k - 1]
  
  def quartile_three(self, column):
    sorted_column = sorted(column)
    k = len(sorted_column) * 3 // 4
    if len(sorted_column) % 4 < 2:
        return sorted_column[k-1] + sorted_column[k] / 2
    else:
        return sorted_column[k - 1]

  def run(self):
    df = pd.read_csv(self.filepath, sep=',')
    df = df.dropna()
    df = df.drop('Index', axis=1)
    df_numeric = df.dropna().select_dtypes(include=['number'])
    df_numeric.reset_index(drop=True, inplace=True)
    column_width = max(len(f"feature {i}") for i in range(len(df_numeric.columns))) + 3

    def format_value(value):
        return f"{value:.6f}".rjust(column_width)

    sys.stdout.write("".rjust(column_width))
    for i in range(len(df_numeric.columns)):
        sys.stdout.write(f"feature {i}".rjust(column_width))
    sys.stdout.write("\n")

    sys.stdout.write("Count".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(df_numeric.shape[0]))
    sys.stdout.write("\n")

    sys.stdout.write("Mean".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(self.get_mean(df_numeric[col])))
    sys.stdout.write("\n")

    sys.stdout.write("Std".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(self.get_std(df_numeric[col])))
    sys.stdout.write("\n")

    sys.stdout.write("Min".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(self.get_min(df_numeric[col])))
    sys.stdout.write("\n")

    sys.stdout.write("25%".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(self.quartile_one(df_numeric[col])))
    sys.stdout.write("\n")

    sys.stdout.write("50%".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(self.quartile_two(df_numeric[col])))
    sys.stdout.write("\n")

    sys.stdout.write("75%".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(self.quartile_three(df_numeric[col])))
    sys.stdout.write("\n")

    sys.stdout.write("Max".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(self.get_max(df_numeric[col])))
    sys.stdout.write("\n\n")

    # Uncomment the following lines to use pandas built-in methods
      
    sys.stdout.write("Count".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(df_numeric[col].count()))
    sys.stdout.write("\n")
    sys.stdout.write("Mean".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(df_numeric[col].mean()))
    sys.stdout.write("\n")
    sys.stdout.write("Std".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(df_numeric[col].std()))
    sys.stdout.write("\n")
    sys.stdout.write("Min".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(df_numeric[col].min()))
    sys.stdout.write("\n")
    sys.stdout.write("25%".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(df_numeric[col].quantile(0.25)))
    sys.stdout.write("\n")
    sys.stdout.write("50%".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(df_numeric[col].quantile(0.5)))
    sys.stdout.write("\n")
    sys.stdout.write("75%".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(df_numeric[col].quantile(0.75)))
    sys.stdout.write("\n")
    sys.stdout.write("Max".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(df_numeric[col].max()))
    sys.stdout.write("\n")

def main():
  a = Describe()
  a.run()

if __name__ == "__main__":
  main()