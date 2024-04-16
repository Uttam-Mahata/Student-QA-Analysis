import os
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler

# Import the ClusterOptimizer class
from tr import ClusterOptimizer

# Define the directory paths
separate_csv_folder = 'separate_csv_datasets'

# Iterate through each separate CSV dataset for each question
for file_name in os.listdir(separate_csv_folder):
    if file_name.endswith('.csv'):
        # Load the dataset
        dataset_path = os.path.join(separate_csv_folder, file_name)
        dataset = pd.read_csv(dataset_path)

        # Drop unnecessary columns (e.g., Question and Student)
        # if they are present in the dataset
        if 'Question' in dataset.columns:
            dataset.drop(columns=['Question'], inplace=True)
        if 'Student' in dataset.columns:
            dataset.drop(columns=['Student'], inplace=True)

        # Apply StandardScaler to the dataset
        scaled_dataset = StandardScaler().fit_transform(dataset.transpose()).transpose()

        # Convert the scaled dataset back to a DataFrame
        scaled_df = pd.DataFrame(scaled_dataset, columns=dataset.columns)

        # Initialize the ClusterOptimizer with a threshold value
        optimizer = ClusterOptimizer(scaled_df, threshold_value=0.75)

        # Get the cluster labels
        cluster_labels = optimizer.labels

        # Print cluster information
        print(f'Cluster labels for {file_name}:')
        print(cluster_labels)
        print(Counter(cluster_labels))
        print(f'Number of clusters: {optimizer.no_of_clusters}')
