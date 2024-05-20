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
separate_csv_folder = 'separate_csv_datasets'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

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
        average_vector = np.mean(model.encode(answer_sentences), axis=0)

        # Save the average vector to a file
        output_file_path = os.path.join(output_question_folder, f'answer{student_no}_avg.npy')
        np.save(output_file_path, average_vector)

        # Append the row data to the dataset rows list
        row_data = [f'question{question_no}', f'student{student_no}'] + average_vector.tolist()
        dataset_rows.append(row_data)

# Convert the list of rows into a DataFrame
dataset_columns = ['Question', 'Student'] + [f'V{i}' for i in range(1, vector_length + 1)]
dataset = pd.DataFrame(dataset_rows, columns=dataset_columns)

# Save the complete dataset to a CSV file
complete_csv_file_path = os.path.join(output_folder, 'complete_dataset.csv')
dataset.to_csv(complete_csv_file_path, index=False)
print("Complete dataset CSV file has been saved successfully.")

# Create the separate CSV folder if it doesn't exist
os.makedirs(separate_csv_folder, exist_ok=True)

# Group the dataset by question
grouped_by_question = dataset.groupby('Question')

# Save each group (question) to a separate CSV file
for question, group_df in grouped_by_question:
    # Define the CSV file path for the current question
    csv_file_path = os.path.join(separate_csv_folder, f'{question}_dataset.csv')

    # Save the group DataFrame to a CSV file
    group_df.to_csv(csv_file_path, index=False)

print("Separate CSV datasets for each question have been saved successfully.")
