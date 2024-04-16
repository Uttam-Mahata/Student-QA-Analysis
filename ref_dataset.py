import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the directory paths
answers_folder = 'answers'
output_folder = 'reference_answer_vectors'
reference_csv_file = 'reference_answer_dataset.csv'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize an empty list to store the dataset rows
dataset_rows = []

# Iterate through each question folder
for question_no in range(1, 6):
    question_folder = os.path.join(answers_folder, f'question{question_no}')

    # Read the reference answer from the 'ref.txt' file
    ref_file_path = os.path.join(question_folder, 'ref.txt')
    with open(ref_file_path, 'r', encoding='utf-8') as f:
        ref_answer = f.read().strip()

    # Split the reference answer into sentences
    ref_sentences = [sentence.strip() for sentence in ref_answer.split('.') if sentence.strip()]

    # Vectorize each sentence in the reference answer
    sentence_vectors = [model.encode([sentence])[0] for sentence in ref_sentences]

    # Compute the average vector for the reference answer
    if sentence_vectors:
        ref_answer_vector = np.mean(sentence_vectors, axis=0)
    else:
        # If there are no sentences in the reference answer, use a zero vector
        ref_answer_vector = np.zeros_like(sentence_vectors[0])

    # Save the reference answer vector to a file
    output_file_path = os.path.join(output_folder, f'question{question_no}_ref_vector.npy')
    np.save(output_file_path, ref_answer_vector)

    # Append the row data to the dataset rows list
    row_data = [f'question{question_no}'] + ref_answer_vector.tolist()
    dataset_rows.append(row_data)

# Convert the list of rows into a DataFrame
vector_length = len(dataset_rows[0]) - 1  # Subtract 1 for 'Question' column
dataset_columns = ['Question'] + [f'V{i}' for i in range(1, vector_length + 1)]
dataset = pd.DataFrame(dataset_rows, columns=dataset_columns)

# Save the dataset to a CSV file
dataset.to_csv(reference_csv_file, index=False)

print("Reference answer dataset has been saved successfully.")
