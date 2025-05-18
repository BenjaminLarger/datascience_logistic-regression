import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

class Histogram:
  def __init__(self):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    self.csv_dir = os.path.join(parent_dir, 'datasets/')
    if not os.path.exists(self.csv_dir):
        os.makedirs(self.csv_dir)
    self.filename = sys.argv[1] if len(sys.argv) == 2 else "dataset_train.csv"
    self.filepath = os.path.join(self.csv_dir, self.filename)
  
  def find_homogeneous_course(self, df):
    """
    Find which course has the most homogeneous distribution across houses
    using statistical measures of distribution similarity.
    """
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    course_columns = df.select_dtypes(include=['number']).columns
    
    homogeneity_scores = {}
    
    for course in course_columns:
        if course == 'Index':
            continue
            
        # Get distributions for each house for this course
        house_distributions = []
        for house in houses:
            # Get scores for this house and course
            scores = df[df['Hogwarts House'] == house][course].dropna()
            house_distributions.append(scores)
        
        # Calculate homogeneity using variance of means and standard deviations
        means = [dist.mean() for dist in house_distributions if len(dist) > 0]
        stds = [dist.std() for dist in house_distributions if len(dist) > 0]
        
        # A lower score means more homogeneous
        mean_variance = np.var(means)
        std_variance = np.var(stds)
        combined_score = mean_variance + std_variance
        
        homogeneity_scores[course] = combined_score
    
    # Find the most homogeneous course (lowest score)
    most_homogeneous = min(homogeneity_scores, key=homogeneity_scores.get)
    
    print(f"Homogeneity scores (lower is more homogeneous):")
    for course in sorted(homogeneity_scores, key=homogeneity_scores.get):
        print(f"{course}: {homogeneity_scores[course]:.4f}")
        
    print(f"\nThe most homogeneous course is: {most_homogeneous}")
    
    return most_homogeneous, homogeneity_scores

  def plot_course_histogram(self, df, course):
        """
        Plot histogram for each course one by one for better readability,
        with special emphasis on the most homogeneous course.
        """
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        colors = ['#740001', '#ECB939', '#0E1A40', '#1A472A']
        
        # First, plot the most homogeneous course
        plt.figure(figsize=(10, 6))
        plt.suptitle(f'Most Homogeneous Course: {course}', fontsize=16)
        
        for house, color in zip(houses, colors):
            house_data = df[df['Hogwarts House'] == house][course].dropna()
            plt.hist(house_data, bins=20, alpha=0.7, label=house, color=color)
        
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {course} scores across houses')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
        # Then plot each course individually for comparison
        course_columns = [c for c in df.select_dtypes(include=['number']).columns if c != 'Index']
        
        print("\nShowing histograms for each course. Close each plot window to view the next one.")
        print("Compare these distributions to see why the selected course is most homogeneous.\n")
        
        for course_name in course_columns:
            plt.figure(figsize=(10, 6))
            
            for house, color in zip(houses, colors):
                house_data = df[df['Hogwarts House'] == house][course_name].dropna()
                plt.hist(house_data, bins=15, alpha=0.7, label=house, color=color)
            
            is_homogeneous = course_name == course
            title_suffix = " (MOST HOMOGENEOUS)" if is_homogeneous else ""
            
            plt.title(f'Course: {course_name}{title_suffix}')
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            plt.show()
            
  def run(self):
      try:
          df = pd.read_csv(self.filepath, sep=',')
          # Remove rows with missing Hogwarts House values
          df = df[df['Hogwarts House'].notna()]
          
          most_homogeneous, _ = self.find_homogeneous_course(df)
          self.plot_course_histogram(df, most_homogeneous)
          
      except Exception as e:
          print(f"Error occurred: {e}")

if __name__ == "__main__":
    histogram = Histogram()
    histogram.run()