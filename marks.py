import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Import the ClusterOptimizer class
from tr import ClusterOptimizer

# Define the directory paths
separate_csv_folder = 'separate_csv_datasets'
reference_vectors_folder = 'reference_answer_vectors'
output_folder = 'cluster_marks'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the marks range and thresholds
marks_range = range(10, -1, -1)
thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

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
        optimizer = ClusterOptimizer(scaled_df, threshold_value=0.8)

        # Get the cluster centroids
        cluster_centroids = optimizer.get_centroid()

        # Load the reference answer vectors
        reference_vectors_path = os.path.join(reference_vectors_folder, f'{file_name.split("_")[0]}_ref_vector.npy')
        reference_vector = np.load(reference_vectors_path)

        # Calculate cosine similarity between each cluster centroid and reference answer vector
        similarity_scores = cosine_similarity(cluster_centroids, [reference_vector])

        # Assign marks based on similarity scores
        marks = []
        for score in similarity_scores.flatten():
            assigned_mark = next(mark for threshold, mark in zip(thresholds, marks_range) if score >= threshold)
            marks.append(assigned_mark)

        # Add cluster labels and marks to the DataFrame
        result_df = pd.DataFrame({'Cluster': range(len(cluster_centroids)), 'Marks': marks})

        # Save the results to a CSV file
        output_file_path = os.path.join(output_folder, f'{file_name.split(".")[0]}_cluster_marks.csv')
        result_df.to_csv(output_file_path, index=False)

        # Print cluster marks information
        print(f'Cluster marks for {file_name}:')
        print(result_df)
