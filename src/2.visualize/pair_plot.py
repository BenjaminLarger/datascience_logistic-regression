import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import f_oneway

class PairPlot:
    def __init__(self):
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        self.csv_dir = os.path.join(parent_dir, 'datasets/')
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
        self.filename = sys.argv[1] if len(sys.argv) == 2 else "dataset_train.csv"
        self.filepath = os.path.join(self.csv_dir, self.filename)

    def analyze_feature_importance(self, X, y):
        """
        Analyze feature importance using multiple methods:
        1. Mutual Information (information gain)
        2. ANOVA F-statistic
        3. Correlation between features
        
        Parameters:
        - X: Feature dataframe
        - y: Target series (Hogwarts House)
        
        Returns:
        - Dictionary of feature importance metrics
        """
        results = {}
        
        # Method 1: Mutual Information (measures dependency between variables)
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_scores_dict = {feature: score for feature, score in zip(X.columns, mi_scores)}
        results['mutual_info'] = {k: v for k, v in sorted(mi_scores_dict.items(), 
                                                          key=lambda item: item[1], 
                                                          reverse=True)}
        
        # Method 2: ANOVA F-statistic for each feature
        f_scores = {}
        houses = y.unique()
        
        for feature in X.columns:
            groups = [X[X.index.isin(y[y == house].index)][feature].values for house in houses]
            f_stat, p_value = f_oneway(*groups)
            f_scores[feature] = {'f_stat': f_stat, 'p_value': p_value}
        
        # Sort by F-statistic (higher means more discriminative power)
        results['anova'] = {k: v for k, v in sorted(f_scores.items(), 
                                                   key=lambda item: item[1]['f_stat'], 
                                                   reverse=True)}
        
        # Method 3: Get correlation matrix
        corr_matrix = X.corr().abs()
        
        # Get average correlation for each feature
        avg_corr = corr_matrix.mean(axis=1)
        results['avg_correlation'] = avg_corr.to_dict()
        
        # Get highly correlated feature pairs
        feature_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > 0.7:  # High correlation threshold
                    feature_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        results['high_correlation_pairs'] = sorted(feature_pairs, key=lambda x: x[2], reverse=True)
        
        return results
    
    def recommend_features(self, analysis_results):
        """
        Recommend features for logistic regression based on analysis results.
        
        Parameters:
        - analysis_results: Dictionary with feature importance metrics
        
        Returns:
        - List of recommended features
        """
        # Start with top features from mutual information
        mi_features = list(analysis_results['mutual_info'].keys())
        anova_features = list(analysis_results['anova'].keys())
        
        # Combine top features from both methods (get top 50% from each)
        top_mi = set(mi_features[:len(mi_features)//2])
        top_anova = set(anova_features[:len(anova_features)//2])
        
        # Union of top features from both methods
        candidate_features = top_mi.union(top_anova)
        
        # Remove one feature from each highly correlated pair
        for feature1, feature2, _ in analysis_results['high_correlation_pairs']:
            if feature1 in candidate_features and feature2 in candidate_features:
                # Keep the one with higher mutual information
                if analysis_results['mutual_info'][feature1] < analysis_results['mutual_info'][feature2]:
                    candidate_features.remove(feature1)
                else:
                    candidate_features.remove(feature2)
        
        return sorted(list(candidate_features))
    
    def plot_pair_plot(self, df, selected_features=None, target_column='Hogwarts House'):
        """
        Create a pair plot visualization with selected features
        
        Parameters:
        - df: DataFrame with data
        - selected_features: List of features to include (optional)
        - target_column: Name of the target column for color coding
        """
        if selected_features is None:
            # If no features specified, use all features
            numeric_features = df.select_dtypes(include=np.number).columns.tolist()
            if 'Index' in numeric_features:
                numeric_features.remove('Index')
            selected_features = numeric_features
        
        # Limit to a reasonable number of features for readability
        if len(selected_features) > 5:
            print(f"Limiting pair plot to top 5 features for readability")
            selected_features = selected_features[:5]
        
        plot_df = df[selected_features + [target_column]].copy()
        
        # Create pair plot
        plt.figure(figsize=(12, 10))
        print("Generating pair plot, this may take a moment...")
        
        sns.pairplot(plot_df, hue=target_column, height=2.5,
                              diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
        
        plt.suptitle('Pair Plot of Selected Features by Hogwarts House', y=1.02, fontsize=16)
        plt.tight_layout()
        # plt.show()
        
        # Also show a correlation heatmap for selected features
        plt.figure(figsize=(10, 8))
        corr_matrix = plot_df[selected_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Selected Features')
        plt.tight_layout()
        plt.show()

    def run(self):
        try:
            # Read the data
            df = pd.read_csv(self.filepath)
            print(f"Successfully loaded data from {self.filepath}")
            
            # Drop rows with missing Hogwarts House
            df = df[df['Hogwarts House'].notna()]
            
            # Get numeric columns for analysis
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if 'Index' in numeric_cols:
                numeric_cols.remove('Index')  # Remove Index if present
            
            # Create a clean dataset for analysis (no NaNs)
            analysis_df = df[numeric_cols + ['Hogwarts House']].dropna()
            print(f"Analyzing {len(analysis_df)} rows with complete data")
            
            # Create feature matrix and target vector
            X = analysis_df[numeric_cols]
            y = analysis_df['Hogwarts House']
            
            # Analyze feature importance
            print("Analyzing feature importance...")
            analysis_results = self.analyze_feature_importance(X, y)
            
            # Show mutual information scores
            print("\nFeature ranking by Mutual Information (higher = more predictive):")
            for i, (feature, score) in enumerate(analysis_results['mutual_info'].items(), 1):
                print(f"{i}. {feature}: {score:.4f}")
            
            # Show ANOVA results
            print("\nFeature ranking by ANOVA F-statistic (higher = more class separation):")
            for i, (feature, stats) in enumerate(analysis_results['anova'].items(), 1):
                print(f"{i}. {feature}: F={stats['f_stat']:.4f}, p={stats['p_value']:.4e}")
            
            # Show highly correlated pairs
            print("\nHighly correlated feature pairs (correlation > 0.7):")
            for feature1, feature2, corr in analysis_results['high_correlation_pairs']:
                print(f"{feature1} & {feature2}: {corr:.4f}")
            
            # Get recommended features
            recommended_features = self.recommend_features(analysis_results)
            print("\nRecommended features for logistic regression:")
            for i, feature in enumerate(recommended_features, 1):
                print(f"{i}. {feature}")
                
            # Create pair plot with only the recommended features
            self.plot_pair_plot(df, recommended_features)
            
            # Also create a full pair plot for comparison
            print("\nWould you like to see the full pair plot with all features? (y/n)")
            response = input().strip().lower()
            if response == 'y':
                self.plot_pair_plot(df)
            
        except FileNotFoundError:
            print(f"Error: File not found at {self.filepath}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

def main():
    plot = PairPlot()
    plot.run()

if __name__ == "__main__":
    main()