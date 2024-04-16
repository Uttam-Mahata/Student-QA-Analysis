import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Get the length of the sentence vectors
vector_length = len(model.encode(['example'])[0])

# Define the directory paths
answers_folder = 'answers'
output_folder = 'average_vectors'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the CSV file path for storing the dataset
csv_file_path = 'average_vectors_dataset.csv'

# Define a function to compute the average vector for a list of sentences
def compute_average_vector(sentences):
    sentence_embeddings = model.encode(sentences)
    return np.mean(sentence_embeddings, axis=0)

# Initialize an empty list to store the dataset rows
dataset_rows = []

# Iterate through each question folder
for question_no in range(1, 6):
    question_folder = os.path.join(answers_folder, f'question{question_no}')
    output_question_folder = os.path.join(output_folder, f'question{question_no}')
    os.makedirs(output_question_folder, exist_ok=True)

    # Iterate through each student's answer
    for student_no in range(1, 46):
        # Read the student's answer
        answer_file_path = os.path.join(question_folder, f'answer{student_no}.txt')
        with open(answer_file_path, 'r', encoding='utf-8') as f:
            answer_sentences = [sentence.strip() for sentence in f.readlines()]

        # Compute the average vector for the answer
        average_vector = compute_average_vector(answer_sentences)

        # Save the average vector to a file
        output_file_path = os.path.join(output_question_folder, f'answer{student_no}_avg.npy')
        np.save(output_file_path, average_vector)

        # Append the row data to the dataset rows list
        row_data = [f'question{question_no}', f'student{student_no}'] + average_vector.tolist()
        dataset_rows.append(row_data)

# Convert the list of rows into a DataFrame
dataset_columns = ['Question', 'Student'] + [f'V{i}' for i in range(1, vector_length + 1)]
dataset = pd.DataFrame(dataset_rows, columns=dataset_columns)

# Save the dataset to a CSV file
dataset.to_csv(csv_file_path, index=False)

print("Average vectors and dataset have been saved successfully.")
