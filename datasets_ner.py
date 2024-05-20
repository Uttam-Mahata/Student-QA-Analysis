import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the English language model in spaCy for NER
nlp = spacy.load("en_core_web_sm")

# Define the directory paths
answers_folder = 'answers'
output_folder = 'ner_average_vectors'
separate_csv_folder = 'ner_separate_csv_datasets'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define a function to compute the average vector for a list of vectors
def compute_average_vector(vectors):
    return np.mean(vectors, axis=0)

# Initialize an empty list to store the dataset rows
dataset_rows = []

# Iterate through each question folder
for question_no in range(1, 6):
    question_folder = os.path.join(answers_folder, f'question{question_no}')
    output_question_folder = os.path.join(output_folder, f'question{question_no}')
    os.makedirs(output_question_folder, exist_ok=True)

    # Initialize an empty list to store vectors for each answer
    for student_no in range(1, 46):
        answer_vectors = []

        # Read the answer file
        answer_file_path = os.path.join(question_folder, f'answer{student_no}.txt')
        with open(answer_file_path, 'r', encoding='utf-8') as f:
            answer_sentences = [sentence.strip() for sentence in f.readlines()]

        # Vectorize each sentence considering named entity pairs
        for sentence in answer_sentences:
            doc = nlp(sentence)
            entities = [ent.text for ent in doc.ents]

            if len(entities) == 1:
                # Single Named Entity Pair
                entity = entities[0]
                start_idx = sentence.find(entity)
                end_idx = start_idx + len(entity)
                sentence_to_vectorize = sentence[start_idx:end_idx]
                sentence_vector = model.encode([sentence_to_vectorize])[0]
                answer_vectors.append(sentence_vector)
            elif len(entities) > 1:
                # Multiple Named Entity Pairs
                pairs = [(ent1.text, ent2.text) for ent1, ent2 in zip(doc.ents, doc.ents[1:])]
                if pairs:
                    # Case 2: Multiple pairs with no common words between them
                    for pair in pairs:
                        start_idx = sentence.find(pair[0])
                        end_idx = sentence.rfind(pair[1]) + len(pair[1])
                        sentence_to_vectorize = sentence[start_idx:end_idx]
                        sentence_vector = model.encode([sentence_to_vectorize])[0]
                        answer_vectors.append(sentence_vector)
                else:
                    # Case 3: Multiple overlapping pairs
                    start_idx = sentence.find(entities[0])
                    end_idx = sentence.rfind(entities[-1]) + len(entities[-1])
                    sentence_to_vectorize = sentence[start_idx:end_idx]
                    sentence_vector = model.encode([sentence_to_vectorize])[0]
                    answer_vectors.append(sentence_vector)
            else:
                # No Named Entity Pair
                sentence_vector = model.encode([sentence])[0]
                answer_vectors.append(sentence_vector)

        # Compute the average vector for the answer for this question
        average_vector = compute_average_vector(answer_vectors)

        # Save the average vector to a file for this question
        output_file_path = os.path.join(output_question_folder, f'answer{student_no}_average_vector.npy')
        np.save(output_file_path, average_vector)

        # Append the row data to the dataset rows list
        row_data = [f'question{question_no}', f'student{student_no}'] + average_vector.tolist()
        dataset_rows.append(row_data)

# Convert the list of rows into a DataFrame
vector_length = len(dataset_rows[0]) - 2  # Subtract 2 for 'Question' and 'Student' columns
dataset_columns = ['Question', 'Student'] + [f'V{i}' for i in range(1, vector_length + 1)]
dataset = pd.DataFrame(dataset_rows, columns=dataset_columns)

# Save the dataset to a CSV file
csv_file_path = 'ner_complete_dataset.csv'
dataset.to_csv(csv_file_path, index=False)

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

print("Average vectors and dataset have been saved successfully.")
