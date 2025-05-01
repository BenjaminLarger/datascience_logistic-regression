import pandas as pd
import os
import sys

class Describe:
  def __init__(self):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_dir)
    self.csv_dir = os.path.join(parent_dir, 'datasets/')
    if not os.path.exists(self.csv_dir):
        os.makedirs(self.csv_dir)
    self.filename = sys.argv[1] if len(sys.argv) > 2 else "dataset_train.csv"
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
    return (sum / len(column)) ** 0.5
  
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
  
  def get_quantile(self, column, q):
    sorted_column = sorted(column)
    index = int(q * len(sorted_column))
    return sorted_column[index]

  def run(self):
    df = pd.read_csv(self.filepath, sep=',')
    df = df.dropna()
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
        sys.stdout.write(format_value(self.get_quantile(df_numeric[col], 0.25)))
    sys.stdout.write("\n")

    sys.stdout.write("50%".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(self.get_quantile(df_numeric[col], 0.5)))
    sys.stdout.write("\n")

    sys.stdout.write("75%".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(self.get_quantile(df_numeric[col], 0.75)))
    sys.stdout.write("\n")

    sys.stdout.write("Max".ljust(column_width))
    for col in df_numeric.columns:
        sys.stdout.write(format_value(self.get_max(df_numeric[col])))
    sys.stdout.write("\n\n")

    # Uncomment the following lines to use pandas built-in methods
    #   
    # sys.stdout.write("Count".ljust(column_width))
    # for col in df_numeric.columns:
    #     sys.stdout.write(format_value(df_numeric[col].count()))
    # sys.stdout.write("\n")
    # sys.stdout.write("Mean".ljust(column_width))
    # for col in df_numeric.columns:
    #     sys.stdout.write(format_value(df_numeric[col].mean()))
    # sys.stdout.write("\n")
    # sys.stdout.write("Std".ljust(column_width))
    # for col in df_numeric.columns:
    #     sys.stdout.write(format_value(df_numeric[col].std()))
    # sys.stdout.write("\n")
    # sys.stdout.write("Min".ljust(column_width))
    # for col in df_numeric.columns:
    #     sys.stdout.write(format_value(df_numeric[col].min()))
    # sys.stdout.write("\n")
    # sys.stdout.write("25%".ljust(column_width))
    # for col in df_numeric.columns:
    #     sys.stdout.write(format_value(df_numeric[col].quantile(0.25)))
    # sys.stdout.write("\n")
    # sys.stdout.write("50%".ljust(column_width))
    # for col in df_numeric.columns:
    #     sys.stdout.write(format_value(df_numeric[col].quantile(0.5)))
    # sys.stdout.write("\n")
    # sys.stdout.write("75%".ljust(column_width))
    # for col in df_numeric.columns:
    #     sys.stdout.write(format_value(df_numeric[col].quantile(0.75)))
    # sys.stdout.write("\n")
    # sys.stdout.write("Max".ljust(column_width))
    # for col in df_numeric.columns:
    #     sys.stdout.write(format_value(df_numeric[col].max()))
    # sys.stdout.write("\n")

def main():
  a = Describe()
  a.run()

if __name__ == "__main__":
  main()